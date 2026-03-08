# v4 변경사항 요약

## 주요 추가 기능
1) 유니버스 정규화 자동화
- 대표 티커(primary) 선택 로직(우선순위 YAML)
- tickers 정리/중복 점검 리포트 생성
- provider_ticker(yfinance) 컬럼 생성

2) seed_edges 자동 생성기
- Excel(.xlsx) → seed_edges.csv 변환 (fuzzy 매핑 + ticker alias)
- DOCX 표(table) → seed_edges.csv 변환(베스트에포트)
- 변환 리포트(seed_import_report.md) 자동 생성
- seed validation report 자동 생성

3) 안정성 개선
- pandas.to_markdown 의존성(tabulate) 명시 + fallback 함수(df_to_markdown)

## 새 CLI 명령어
- normalize-universe
- seed-from-xlsx
- seed-from-docx
- validate-seed
