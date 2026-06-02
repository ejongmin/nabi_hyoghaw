import pandas as pd
import os
import re

# 1. 경로 설정
BASE_DIR = "/mnt/c/Users/john9/nabi_hyoghaw"
DRIVE_DIR = "/mnt/g/내 드라이브/핀테크+nlp/데이터셋/이종민"
RAW_NEWS_PATH = os.path.join(DRIVE_DIR, "gdelt_2y_risk_stream.csv")
UNIVERSE_PATH = os.path.join(BASE_DIR, "data/universe/univers_final.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/processed/risk_events.parquet")

def build_alias_map(universe_df):
    """유니버스 정보를 바탕으로 기업별 별칭 사전을 구축합니다."""
    # 설계 문서 2.1절: 악칭, 약칭, 중문, 영문 표기를 모두 포함하는 사전 확장
    alias_dict = {
        "373220.KS": ["LG ENERGY", "LGES", "LG CHEM", "LGC", "LG ENERGY SOLUTION"],
        "300750.SZ": ["CATL", "CONTEMPORARY AMPEREX", "NINGDE TIMES"],
        "TSLA": ["TESLA", "TSLA", "ELON MUSK"], # 경영진 이름 포함 가능 (옵션)
        "F": ["FORD"],
        "GM": ["GENERAL MOTORS", "GM"],
        "005930.KS": ["SAMSUNG SDI", "SDI"],
        "BYD": ["BYD", "BYD CO"],
        "ALB": ["ALBEMARLE", "ALB"]
        # 나머지 38개 기업도 univers_final.csv 기반으로 자동 추가
    }
    
    # 기본적으로 canonical_name 추가
    for _, row in universe_df.iterrows():
        cid = row['company_id']
        name = str(row['canonical_name']).upper()
        if cid not in alias_dict:
            alias_dict[cid] = [name]
        elif name not in alias_dict[cid]:
            alias_dict[cid].append(name)
            
    return alias_dict

def normalize_and_filter():
    if not os.path.exists(RAW_NEWS_PATH):
        print("❌ 원본 뉴스 파일을 찾을 수 없습니다.")
        return

    print("🚀 7-8단계: 엔티티 정규화 및 리스크 이벤트 생성 시작...")
    
    # 데이터 로드 (파싱 에러 행은 건너뜀)
    news_df = pd.read_csv(RAW_NEWS_PATH, on_bad_lines='skip', low_memory=False)
    universe_df = pd.read_csv(UNIVERSE_PATH)
    alias_map = build_alias_map(universe_df)
    
    processed_events = []
    
    for _, row in news_df.iterrows():
        entities_text = str(row['entities']).upper()
        tone_str = str(row['tone_info'])
        
        # 1. 뉴스에서 언급된 우리 기업(티커) 찾기
        mentioned_tickers = []
        for ticker, aliases in alias_map.items():
            for alias in aliases:
                # 단어 경계(\b)를 사용하여 정확한 이름 매칭 (예: CATL이 포함된 다른 단어 방지)
                if re.search(r'\b' + re.escape(alias) + r'\b', entities_text):
                    mentioned_tickers.append(ticker)
                    break # 한 기업에 대해 여러 별칭이 있어도 하나만 찾으면 됨
        
        # 2. 우리 기업이 하나라도 언급된 경우에만 리스크 점수 계산
        if mentioned_tickers:
            try:
                # GDELT Tone 형식: Tone,Positive,Negative,Polarity,etc... (콤마로 구분)
                tone_score = float(tone_str.split(',')[0])
                
                # 설계 문서: Tone을 0~1 사이의 Severity(강도)로 스케일링
                # -10 이하면 매우 심각(1.0), 0 이상이면 리스크 낮음(0.1)
                severity = max(0.1, min(1.0, abs(tone_score) / 10.0)) if tone_score < 0 else 0.1
                
                processed_events.append({
                    "event_time": row['event_time'],
                    "entity_ids": mentioned_tickers, # 리스트 형태로 저장 (Co-mention 대응)
                    "severity": severity,
                    "avg_tone": tone_score,
                    "url": row['url'],
                    "themes": row['themes']
                })
            except:
                continue

    # 3. 결과 저장 (Parquet 포맷)
    if processed_events:
        result_df = pd.DataFrame(processed_events)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        result_df.to_parquet(OUTPUT_PATH, index=False)
        print(f"✅ 정규화 완료: {len(result_df)}건의 리스크 이벤트 생성 -> {OUTPUT_PATH}")
        print(f"📊 매핑 성공률(DQ Gate): {len(result_df)/len(news_df)*100:.1f}%")
    else:
        print("❌ 매핑된 기업이 없습니다. 별칭 사전을 점검하세요.")

if __name__ == "__main__":
    normalize_and_filter()
