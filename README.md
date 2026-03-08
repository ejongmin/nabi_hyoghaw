# nabi_hyoghaw
nabi_effect/
  configs/            # base.yaml, schema.yaml, risk_keywords.yaml, (선택) base_colab_drive.yaml
  src/
    cli.py            # init → ingest → QC → KG → exposure → CAR → backtest → report
    ingest/           # gdelt, prices, (선택) edgar
    quality/          # data_quality_report
    kg/               # build_kg (seed 기반)
    graph/            # exposure baseline (shortest path, RWR, centrality)
    finance/          # event study(CAR), backtest
    reporting/        # reports/index.md 생성
  data/
    universe/         # *_combined.csv (정본), tickers.txt 등
    seed/seed_edges.csv
    raw/gdelt/articles.csv
    processed/*.parquet
  reports/            # 자동 생성되는 결과물
  docs/               # decisions.md, bible 등
