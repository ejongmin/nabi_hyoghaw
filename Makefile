.PHONY: setup init make_tickers ingest_gdelt ingest_prices build_risk dq build_kg exposure car backtest report pipeline smoke

setup:
	python -m pip install -r requirements.txt

init:
	python -m src.cli init --config configs/base.yaml

make_tickers:
	python -m src.cli make-tickers --config configs/base.yaml

ingest_gdelt:
	python -m src.cli ingest-gdelt --config configs/base.yaml

ingest_prices:
	python -m src.cli ingest-prices --config configs/base.yaml

build_risk:
	python -m src.cli build-risk-events --config configs/base.yaml

dq:
	python -m src.cli data-quality --config configs/base.yaml

build_kg:
	python -m src.cli build-kg --config configs/base.yaml

exposure:
	python -m src.cli compute-exposure --config configs/base.yaml

car:
	python -m src.cli event-study --config configs/base.yaml

backtest:
	python -m src.cli backtest --config configs/base.yaml

report:
	python -m src.cli report-all --config configs/base.yaml

pipeline:
	python -m src.cli pipeline --config configs/base.yaml

smoke:
	python -m src.cli smoke-test --config configs/base.yaml
