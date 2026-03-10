import pandas as pd
from google.cloud import bigquery
import os

# 1. 설정
PROJECT_ID = "nabi-effect-project"
UNIVERSE_PATH = "data/universe/univers_final.csv"
OUTPUT_PATH = "data/raw/gdelt/articles_8y.csv"

def collect_8y_gdelt_bq():
    client = bigquery.Client(project=PROJECT_ID)
    df_uni = pd.read_csv(UNIVERSE_PATH)
    # 기업 이름의 첫 단어(핵심 키워드)만 추출하여 검색 (검색 성능 향상)
    entities = [e.split()[0].upper() for e in df_uni['canonical_name'].dropna().unique() if len(e) > 2]
    
    # 쿼리 생성
    entity_filters = [f"Actor1Name LIKE '%{e}%'" for e in entities[:50]] # 쿼리 길이 제한 방지
    query = f"""
        SELECT 
            GLOBALEVENTID, 
            PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) as event_date,
            Actor1Name, Actor2Name, EventCode, 
            GoldsteinScale, NumArticles, AvgTone, SOURCEURL
        FROM `gdelt-bq.full.events`
        WHERE SQLDATE >= 20180101
        AND ({' OR '.join(entity_filters)})
        LIMIT 15000
    """
    
    print("🚀 BigQuery 8년치 수집 시작...")
    df = client.query(query).to_dataframe()
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ 8년치 수집 완료: {len(df)}건 저장됨 -> {OUTPUT_PATH}")

if __name__ == "__main__":
    collect_8y_gdelt_bq()
