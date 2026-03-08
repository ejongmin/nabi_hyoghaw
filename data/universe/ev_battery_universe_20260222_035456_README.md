# Universe export bundle (20260222_035456)

## Quick pick (recommended)
- CSV: ev_battery_universe_20260222_035456_combined.csv
- Excel: ev_battery_universe_20260222_035456.xlsx
- YAML config: ev_battery_universe_20260222_035456.yaml
- Tickers list: ev_battery_universe_20260222_035456_tickers.txt

## Files included
- CSV (combined/china/overseas): analysis + manual editing
- TSV: CLI-friendly tab-separated
- XLSX: team editing + summary sheet
- JSON: API/web friendly array of records
- NDJSON: streaming ingestion, one record per line
- YAML: config-friendly grouped by region
- TXT: one ticker per line for price download
- MD: copy/paste table for docs
- SQL: create table + inserts for DB
- PKL: fastest for Python-only loading

## Schema
company_id, canonical_name, country, region_group, value_chain_stage, listed, exchanges, tickers, notes

## Notes
- Multiple listings exist; pick one canonical ticker/venue for your price provider.
- Some firms can change listing status; verify before "universe lock".
