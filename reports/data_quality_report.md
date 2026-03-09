# Data Quality Report - News / Risk Events

## Collection
- articles_raw: 9411
- articles_after_dedup: 6512
- events_final: 6318
- dedup_ratio: 30.80% removed

## Risk Type Distribution
- other(4686), geopolitical(764), logistics(568), natural(144), insurance_shipping(104), energy(59)

## Entity Linking
- entity_link_success_rate: 0.00%

## Evidence Text
- evidence_text_coverage: 99.98%

## Severity
- mean: 1.43
- max: 2.50
- min: 1.30

## DQ Gate Status
- risk_type consistency: PASS
- entity_link_rate ≥ 0.90: FAIL
- evidence_text_coverage ≥ 0.95: PASS
