# 나비효과 MVP v4.1 사용 가이드 (로컬 + Colab)

이 문서는 팀원이 **프로젝트 전체 과정**을 이해하고, 로컬/Colab 어디서든 **같은 커맨드**로 실행할 수 있도록 만든 가이드입니다.

---

## 1) 전체 파이프라인 흐름(입력→출력)
1. Universe(기업 마스터) 정리/정규화  
2. Seed edges(관계 테이블) 생성/검증  
3. 뉴스 수집(GDELT) + 가격 수집(yfinance)  
4. Risk events 생성(키워드 기반 + FinBERT 옵션)  
5. KG 생성(nodes/edges)  
6. Exposure 계산(Shortest-path + RWR baseline)  
7. Event study(CAR)  
8. Backtest(Exclude 전략)  
9. DQ Report + Reports index

---

## 2) 설치
```bash
pip install -r requirements.txt
# 옵션(감성/FinBERT)
pip install -r requirements_nlp.txt
# 옵션(GAT)
pip install -r requirements_gnn.txt
```

---

## 3) 가장 추천 실행 순서(실데이터)
### (0) 프로젝트 폴더에서 실행하는지 확인
- 항상 `.../nabi_effect_mvp_template_v4_1/` 폴더에서 실행하세요.

### (1) 환경/경로 점검(doctor)
```bash
python -m src.cli doctor --config configs/base.yaml
```

### (2) 유니버스 정규화(대표 티커 고정/중복 점검)
```bash
python -m src.cli normalize-universe --config configs/base.yaml
```

### (3) seed_edges 자동 생성(엑셀/문서 → 관계 테이블)
엑셀:
```bash
python -m src.cli seed-from-xlsx --config configs/base.yaml --xlsx /path/to/file.xlsx --sheet "Sheet1"
python -m src.cli validate-seed --config configs/base.yaml
```

폴더(엑셀/문서 여러 개):
```bash
python -m src.cli seed-from-folder --config configs/base.yaml --folder /path/to/folder
python -m src.cli validate-seed --config configs/base.yaml
```

DOCX(표가 있는 경우):
```bash
python -m src.cli seed-from-docx --config configs/base.yaml --docx /path/to/file.docx
```

### (4) 전체 파이프라인
```bash
python -m src.cli pipeline --config configs/base.yaml
```

---

## 4) Colab(Drive) 실행
1) Drive에 폴더 업로드  
2) notebooks/COLAB_RUN.ipynb 실행  
3) configs/base_colab_drive.yaml 의 project.root_dir을 본인 Drive 경로로 수정  
4) 실행

---

## 5) 산출물(어디에 뭐가 생기나)
- 유니버스 정규화: `data/processed/universe_normalized.csv`, `reports/universe_validation_report.md`
- 뉴스: `data/raw/gdelt/articles.csv`
- 가격: `data/processed/prices_daily.parquet` (+ 누락 티커: `prices_daily.missing_tickers.txt`)
- 리스크 이벤트: `data/processed/risk_events.parquet`
- KG: `data/processed/kg_nodes.parquet`, `kg_edges.parquet`
- 노출도: `data/processed/exposure_baseline.parquet`
- CAR: `data/processed/car_panel.parquet`
- 백테스트: `reports/backtest_equity.csv`, `reports/backtest_summary.md`
- DQ 리포트: `reports/data_quality_report.md`
- 리포트 인덱스: `reports/index.md`

---

## 6) 실패/오류가 났을 때(실전 체크)
- GDELT 수집 실패: 네트워크/쿼리/기간/레이트리밋 이슈 가능 → query 단순화, max_records 줄이기
- yfinance 누락: 티커 표기법 문제(특히 중국/홍콩) → missing_tickers.txt 확인 후 universe 수정
- CAR 값이 비어있음: 가격 lookback 부족/거래일 매핑 문제 → configs/base.yaml의 prices.lookback_days 확인(기본 300)

