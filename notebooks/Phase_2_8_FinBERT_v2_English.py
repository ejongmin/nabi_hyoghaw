# ============================================================
# Phase 2.8 · FinBERT v2 English-only full re-inference
# ------------------------------------------------------------
# 입력  : data/processed/finbert_input_candidates.parquet  (7,119,988 rows)
# 모델  : ProsusAI/finbert  (gold standard, English only)
# 전처리: fastText lid.176 재필터 (langid 정밀도 0.62 보완)
# 출력  : data/processed/finbert_v2.parquet
#         (url, title, finbert_neg, finbert_neu, finbert_pos, finbert_score, lang_ft)
# 예상  : T4 ≈ 5h / A100 ≈ 1.5h  (batch=256, MAX_LEN=64)
# ============================================================

# ----- Cell 1 : Drive 연결 + GPU 확인 -------------------------
# Colab 에서는 아래 3줄을 %%cell 로 실행
"""
from google.colab import drive
drive.mount('/content/drive')
import os, torch
PROJECT_DIR = '/content/drive/MyDrive/나비효과'  # 경로 맞게 수정
os.chdir(PROJECT_DIR)
print('cwd :', os.getcwd())
print('GPU :', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
"""

# ----- Cell 2 : 의존성 ----------------------------------------
"""
!pip install -q transformers==4.41.2 torch pandas pyarrow tqdm fasttext-wheel
!wget -q -O lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
"""

# ----- Cell 3 : 로드 + 언어 재필터 ----------------------------
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import fasttext
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

INPUT_PATH  = 'data/processed/finbert_input_candidates.parquet'
OUTPUT_PATH = 'data/processed/finbert_v2.parquet'
LID_PATH    = 'lid.176.bin'

MODEL_NAME = 'ProsusAI/finbert'
BATCH      = 256
MAX_LEN    = 64
LANG_MIN_P = 0.65            # fastText 영어 확률 하한

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# 1) 원본 후보 로드 --------------------------------------------
df = pd.read_parquet(INPUT_PATH, columns=['url', 'slug_text'])
df = df.rename(columns={'slug_text': 'title'})
print(f'[load] {len(df):,} rows')

# 2) fastText 재필터 -------------------------------------------
# langid precision 0.62 → fastText lid.176 로 2차 정화하면 약 4.4~4.6M 로 축소 예상
lid = fasttext.load_model(LID_PATH)

def ft_en(texts, min_p=LANG_MIN_P):
    # fasttext는 한번에 list 처리 가능, 개행 제거 필수
    clean = [(t or '').replace('\n', ' ').strip() for t in texts]
    labels, probs = lid.predict(clean, k=1)
    keep = np.array([
        (lab[0] == '__label__en') and (pr[0] >= min_p)
        for lab, pr in zip(labels, probs)
    ])
    return keep

CHUNK = 200_000
keep_mask = np.zeros(len(df), dtype=bool)
for i in tqdm(range(0, len(df), CHUNK), desc='lid.176 refilter'):
    sl = slice(i, i + CHUNK)
    keep_mask[sl] = ft_en(df['title'].iloc[sl].tolist())

df = df.loc[keep_mask].reset_index(drop=True)
print(f'[lid.176] {len(df):,} English rows (≈{len(df)/7_119_988*100:.1f}%)')

# ----- Cell 4 : FinBERT 배치 추론 -----------------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

id2label = model.config.id2label
# ProsusAI/finbert : {0:'positive', 1:'negative', 2:'neutral'}
pos_ix = next(k for k, v in id2label.items() if 'pos' in str(v).lower())
neg_ix = next(k for k, v in id2label.items() if 'neg' in str(v).lower())
neu_ix = next(k for k, v in id2label.items() if 'neu' in str(v).lower())
print('label idx  pos/neg/neu =', pos_ix, neg_ix, neu_ix)

@torch.inference_mode()
def infer_batch(titles):
    enc = tok(
        titles, padding=True, truncation=True,
        max_length=MAX_LEN, return_tensors='pt'
    ).to(device)
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs[:, [neg_ix, neu_ix, pos_ix]]

# ----- Cell 5 : 스트리밍 실행 + 체크포인트 --------------------
# OOM/timeout 재개를 위해 50만 행 단위로 저장
CKPT_STEP = 500_000
start = 0
if os.path.exists(OUTPUT_PATH):
    done = pd.read_parquet(OUTPUT_PATH, columns=['url'])
    done_set = set(done['url'])
    df = df.loc[~df['url'].isin(done_set)].reset_index(drop=True)
    print(f'[resume] skipping {len(done_set):,} already-done rows, remaining {len(df):,}')

buf_neg, buf_neu, buf_pos = [], [], []
buf_url, buf_ttl = [], []

def flush(path_tmp):
    if not buf_url:
        return
    out = pd.DataFrame({
        'url':   buf_url,
        'title': buf_ttl,
        'finbert_neg':   np.concatenate(buf_neg).astype('float32'),
        'finbert_neu':   np.concatenate(buf_neu).astype('float32'),
        'finbert_pos':   np.concatenate(buf_pos).astype('float32'),
    })
    out['finbert_score'] = out['finbert_pos'] - out['finbert_neg']
    if os.path.exists(OUTPUT_PATH):
        prev = pd.read_parquet(OUTPUT_PATH)
        out = pd.concat([prev, out], ignore_index=True)
    out.to_parquet(OUTPUT_PATH, compression='snappy', index=False)
    buf_neg.clear(); buf_neu.clear(); buf_pos.clear()
    buf_url.clear(); buf_ttl.clear()

titles = df['title'].tolist()
urls   = df['url'].tolist()

for i in tqdm(range(0, len(titles), BATCH), desc='FinBERT v2'):
    bt = titles[i:i + BATCH]
    bu = urls[i:i + BATCH]
    probs = infer_batch(bt)     # (b, 3) → neg,neu,pos
    buf_neg.append(probs[:, 0])
    buf_neu.append(probs[:, 1])
    buf_pos.append(probs[:, 2])
    buf_url.extend(bu)
    buf_ttl.extend(bt)
    if len(buf_url) >= CKPT_STEP:
        flush(OUTPUT_PATH)
        print(f'  ✓ checkpoint {i + BATCH:,}/{len(titles):,}')

flush(OUTPUT_PATH)
print('[done] wrote', OUTPUT_PATH)

# ----- Cell 6 : sanity check ----------------------------------
res = pd.read_parquet(OUTPUT_PATH)
print('rows         :', f'{len(res):,}')
print('score min/max:', res['finbert_score'].min(), res['finbert_score'].max())
print('score mean   :', res['finbert_score'].mean())
print('most NEGATIVE:')
for _, r in res.nsmallest(5, 'finbert_score').iterrows():
    print(f'  {r.finbert_score:+.3f}  {r.title[:90]}')
print('most POSITIVE:')
for _, r in res.nlargest(5, 'finbert_score').iterrows():
    print(f'  {r.finbert_score:+.3f}  {r.title[:90]}')
