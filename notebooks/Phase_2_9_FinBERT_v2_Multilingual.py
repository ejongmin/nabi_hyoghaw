# ============================================================
# Phase 2.9 · FinBERT v2 Multilingual (zh / ko / ja / de / es)
# ------------------------------------------------------------
# 입력  : data/processed/finbert_input_candidates.parquet  (7,119,988 rows)
# 대상  : Phase 2.8에서 영어로 분류되지 않은 ~3.73M rows
# 모델  :
#   zh → yiyanghkust/finbert-tone-chinese
#   ko → snunlp/KR-FinBert-SC
#   ja → christian-phu/bert-finetuned-japanese-sentiment
#   de/es/fr/etc → cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual
# 출력  : data/processed/finbert_v2_multilingual.parquet
# 예상  : T4 ≈ 2~3h (비영어 ~3.73M rows, batch=64~128)
# ============================================================

# ----- Cell 1 : Drive 연결 + GPU 확인 -------------------------
"""
from google.colab import drive
drive.mount('/content/drive')
import os, torch
PROJECT_DIR = '/content/drive/Othercomputers/내 노트북/nabi_hyoghaw'
os.chdir(PROJECT_DIR)
print('cwd :', os.getcwd())
print('GPU :', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
"""

# ----- Cell 2 : 의존성 ----------------------------------------
"""
!pip install -q transformers==4.41.2 torch pandas pyarrow tqdm fasttext-wheel sentencepiece fugashi unidic-lite
!wget -q -O lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
"""

# ----- Cell 3 : 로드 + 언어 분류 ------------------------------
import os
import numpy as np
import pandas as pd
import torch
import fasttext
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

INPUT_PATH  = 'data/processed/finbert_input_candidates.parquet'
OUTPUT_PATH = 'data/processed/finbert_v2_multilingual.parquet'
LID_PATH    = 'lid.176.bin'

BATCH_ZH = 128      # 중국어 모델은 빠름
BATCH_KO = 64
BATCH_JA = 64
BATCH_ML = 128      # multilingual 모델
MAX_LEN  = 64
LANG_MIN_P = 0.50   # 언어 판별 확률 하한 (다국어는 좀 느슨하게)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# 1) 전체 후보 로드
df = pd.read_parquet(INPUT_PATH, columns=['url', 'slug_text', 'source_domain'])
df = df.rename(columns={'slug_text': 'title'})
print(f'[load] {len(df):,} rows')

# 2) fastText 언어 판별 (전체)
lid = fasttext.load_model(LID_PATH)

CHUNK = 200_000
lang_labels = []
lang_probs  = []

for i in tqdm(range(0, len(df), CHUNK), desc='lid.176 detect'):
    sl = slice(i, i + CHUNK)
    texts = [(t or '').replace('\n', ' ').strip() for t in df['title'].iloc[sl].tolist()]
    labels, probs = lid.predict(texts, k=1)
    lang_labels.extend([lab[0].replace('__label__', '') for lab in labels])
    lang_probs.extend([pr[0] for pr in probs])

df['lang_ft'] = lang_labels
df['lang_prob'] = lang_probs

# 3) 영어 제외 (Phase 2.8에서 이미 처리됨) + 대상 언어 필터
df_non_en = df[df['lang_ft'] != 'en'].copy()
print(f'[non-English] {len(df_non_en):,} rows')

# 언어별 분포
lang_dist = df_non_en.groupby('lang_ft').size().sort_values(ascending=False)
print('\n[language distribution - top 15]')
for lang, cnt in lang_dist.head(15).items():
    print(f'  {lang}: {cnt:,}')

# 대상 언어 추출 (확률 >= LANG_MIN_P)
target_langs = {
    'zh': [],  # Chinese
    'ko': [],  # Korean
    'ja': [],  # Japanese
    'de': [],  # German
    'es': [],  # Spanish
    'fr': [],  # French (multilingual 모델로 처리)
}

for lang_code in target_langs:
    mask = (df_non_en['lang_ft'] == lang_code) & (df_non_en['lang_prob'] >= LANG_MIN_P)
    target_langs[lang_code] = df_non_en.loc[mask].index.tolist()
    print(f'  {lang_code} (prob>={LANG_MIN_P}): {len(target_langs[lang_code]):,}')

# ----- Cell 4 : 공용 추론 함수 --------------------------------

def find_label_indices(id2label):
    mapping = {i: str(v).lower().strip() for i, v in id2label.items()}
    pos_kw = {'positive', 'pos', 'bullish', 'good', 'label_2', '正面', '긍정', 'ポジティブ'}
    neg_kw = {'negative', 'neg', 'bearish', 'bad',  'label_0', '负面', '부정', 'ネガティブ'}
    neu_kw = {'neutral',  'neu', 'flat', 'label_1', '中性', '중립', 'ニュートラル'}
    pi = ni = ui = None
    for i, v in mapping.items():
        if any(k in v for k in pos_kw) and pi is None: pi = i
        elif any(k in v for k in neg_kw) and ni is None: ni = i
        elif any(k in v for k in neu_kw) and ui is None: ui = i
    return pi, ni, ui, mapping

@torch.inference_mode()
def run_finbert_batch(model, tokenizer, titles, pi, ni, ui, batch_size, max_len):
    """배치 추론 → (n, 3) ndarray [neg, neu, pos]"""
    all_neg, all_neu, all_pos = [], [], []
    for i in tqdm(range(0, len(titles), batch_size), desc='inference'):
        bt = titles[i:i+batch_size]
        enc = tokenizer(bt, padding=True, truncation=True,
                        max_length=max_len, return_tensors='pt').to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_neg.append(probs[:, ni])
        all_neu.append(probs[:, ui] if ui is not None else 1.0 - probs[:, pi] - probs[:, ni])
        all_pos.append(probs[:, pi])
    return (np.concatenate(all_neg).astype('float32'),
            np.concatenate(all_neu).astype('float32'),
            np.concatenate(all_pos).astype('float32'))

def process_language(lang_code, model_name, indices, batch_size):
    """한 언어 처리 → DataFrame 반환"""
    if not indices:
        print(f'  [{lang_code}] no rows, skipping')
        return None
    print(f'\n{"="*60}')
    print(f'[{lang_code}] {model_name} — {len(indices):,} rows')
    print(f'{"="*60}')

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    pi, ni, ui, mapping = find_label_indices(model.config.id2label)
    print(f'  id2label: {mapping}')
    print(f'  pos={pi}, neg={ni}, neu={ui}')

    sub = df.loc[indices].copy()
    titles = sub['title'].astype(str).tolist()

    neg, neu, pos = run_finbert_batch(model, tok, titles, pi, ni, ui, batch_size, MAX_LEN)

    sub['finbert_neg']   = neg
    sub['finbert_neu']   = neu
    sub['finbert_pos']   = pos
    sub['finbert_score'] = (pos - neg).astype('float32')
    sub['finbert_model'] = model_name

    print(f'  score: mean={sub["finbert_score"].mean():+.4f}, '
          f'std={sub["finbert_score"].std():.4f}')
    strong = (sub['finbert_score'].abs() > 0.3).sum()
    print(f'  strong signal (|score|>0.3): {strong:,} ({100*strong/len(sub):.1f}%)')

    del model; torch.cuda.empty_cache()
    return sub[['url', 'title', 'lang_ft',
                'finbert_neg', 'finbert_neu', 'finbert_pos',
                'finbert_score', 'finbert_model']]

# ----- Cell 5 : 언어별 추론 실행 ------------------------------

MODELS = {
    'zh': ('yiyanghkust/finbert-tone-chinese', BATCH_ZH),
    'ko': ('snunlp/KR-FinBert-SC', BATCH_KO),
    'ja': ('christian-phu/bert-finetuned-japanese-sentiment', BATCH_JA),
    'de': ('cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual', BATCH_ML),
    'es': ('cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual', BATCH_ML),
    'fr': ('cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual', BATCH_ML),
}

results = []
for lang_code, (model_name, bs) in MODELS.items():
    out = process_language(lang_code, model_name, target_langs.get(lang_code, []), bs)
    if out is not None:
        results.append(out)

# ----- Cell 6 : 저장 ------------------------------------------
if results:
    final = pd.concat(results, ignore_index=True)
    final.to_parquet(OUTPUT_PATH, compression='snappy', index=False)
    print(f'\n[done] {OUTPUT_PATH}: {len(final):,} rows')

    # 언어별 요약
    print('\n=== 언어별 요약 ===')
    for lang in final['lang_ft'].unique():
        sub = final[final['lang_ft'] == lang]
        print(f'  {lang}: {len(sub):,}건, '
              f'mean={sub["finbert_score"].mean():+.4f}, '
              f'std={sub["finbert_score"].std():.4f}')
else:
    print('[warning] no results to save')

# ----- Cell 7 : sanity check ----------------------------------
if os.path.exists(OUTPUT_PATH):
    res = pd.read_parquet(OUTPUT_PATH)
    print(f'\nrows: {len(res):,}')
    print('most NEGATIVE:')
    for _, r in res.nsmallest(5, 'finbert_score').iterrows():
        print(f'  [{r.lang_ft}] {r.finbert_score:+.3f}  {r.title[:80]}')
    print('most POSITIVE:')
    for _, r in res.nlargest(5, 'finbert_score').iterrows():
        print(f'  [{r.lang_ft}] {r.finbert_score:+.3f}  {r.title[:80]}')

    # 영어 v2와 합산 현황
    if os.path.exists('data/processed/finbert_v2.parquet'):
        en = pd.read_parquet('data/processed/finbert_v2.parquet')
        print(f'\n=== 전체 FinBERT v2 합산 ===')
        print(f'  English:      {len(en):,}')
        print(f'  Multilingual: {len(res):,}')
        print(f'  Total:        {len(en)+len(res):,}')
