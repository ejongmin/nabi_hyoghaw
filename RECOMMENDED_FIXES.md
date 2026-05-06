# Recommended Fixes for Stage 3 Data Gaps

## Fix 1: PyArrow Pre-filter Bug (CRITICAL)

### Problem
Type B (risk_only) articles with null `v2_organizations` are removed by PyArrow filter before they can be identified and preserved, violating the intended design.

### Current Code (BROKEN)
```python
# File: src/ingest/gdelt_bq.py, lines 556-580
for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=use_cols):
    n_processed += batch.num_rows

    # ── Stage 0: PyArrow 레벨 pre-filter ──
    # v2_organizations가 null이거나 빈 문자열인 행 제거 (pandas 변환 전)
    v2org_col = batch.column("v2_organizations")
    not_null = pc.is_valid(v2org_col)
    not_empty = pc.not_equal(pc.utf8_length(pc.if_else(not_null, v2org_col, pa.scalar(""))), 0)
    mask = pc.and_(not_null, not_empty)
    batch = batch.filter(mask)  # ← TYPE B ARTICLES WITH NULL v2_org ARE REMOVED HERE

    if batch.num_rows == 0:
        continue

    n_pyarrow_kept += batch.num_rows
    df = batch.to_pandas()

    # ── Aho-Corasick 단일 스캔 매칭 ──
    df["entity_ids"] = _vectorized_company_match(df["v2_organizations"], ac_automaton, keyword_to_cid)

    # Filter: entity 없는 행 제거
    if filter_no_entity:
        has_entity = df["entity_ids"].apply(len) > 0
        is_risk_only = df.get("collection_type", pd.Series(dtype=str)) == "risk_only"
        df = df[has_entity | is_risk_only].copy()  # ← TOO LATE! Type B already removed by PyArrow
```

### Solution Option A: Preserve Type B Before PyArrow Filter
```python
# REVISED: Check collection_type BEFORE PyArrow filter removes rows
for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=use_cols):
    n_processed += batch.num_rows

    # NEW STEP: Read collection_type from original batch (before filtering)
    if "collection_type" in use_cols:
        collection_type_col = batch.column("collection_type")
        is_type_b = pc.equal(collection_type_col, pa.scalar("risk_only"))
    else:
        is_type_b = pc.cast(pc.is_null(batch.column("v2_organizations")), pa.bool_())
        is_type_b = pc.invert(is_type_b)  # All False (no collection_type means not Type B)

    # ── Stage 0: REVISED PyArrow pre-filter ──
    # v2_organizations가 null이거나 빈 문자열인 행 제거
    # BUT: Keep Type B articles regardless of v2_organizations status
    v2org_col = batch.column("v2_organizations")
    not_null = pc.is_valid(v2org_col)
    not_empty = pc.not_equal(pc.utf8_length(pc.if_else(not_null, v2org_col, pa.scalar(""))), 0)
    has_org = pc.and_(not_null, not_empty)

    # NEW: Preserve Type B articles
    mask = pc.or_(has_org, is_type_b)
    batch = batch.filter(mask)

    if batch.num_rows == 0:
        continue

    n_pyarrow_kept += batch.num_rows
    n_type_b_kept = pc.sum(pc.cast(is_type_b, pa.int64())).as_py()  # NEW: log Type B preservation

    df = batch.to_pandas()

    # ── Aho-Corasick 단일 스캔 매칭 ──
    # For Type B articles with null v2_organizations, entity_ids will be empty list
    df["entity_ids"] = _vectorized_company_match(df["v2_organizations"], ac_automaton, keyword_to_cid)

    # Filter: entity 없는 행 제거
    if filter_no_entity:
        has_entity = df["entity_ids"].apply(len) > 0
        is_risk_only = df.get("collection_type", pd.Series(dtype=str)) == "risk_only"
        df = df[has_entity | is_risk_only].copy()  # NOW this preserves Type B correctly

    log.info(f"  Batch: {n_pyarrow_kept:,} kept, {n_type_b_kept:,} Type B preserved")
```

### Solution Option B: Separate Type B Processing
```python
# ALTERNATIVE: Process Type B and Type A (standard) articles separately

def _process_single_parquet_revised(
    parquet_path: Path,
    patterns: List[Tuple[re.Pattern, str]],
    risk_patterns: List[Tuple[str, re.Pattern]],
    ac_automaton,
    keyword_to_cid: List[str],
    risk_ac,
    risk_keyword_to_rtype: List[str],
    filter_no_entity: bool = True,
) -> pd.DataFrame:
    """
    Process Type A and Type B articles with separate pipelines.
    """
    import gc
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    log.info(f"Processing {parquet_path.name} (revised with separate Type A/B pipeline)...")
    pf = pq.ParquetFile(parquet_path)
    n_total = pf.metadata.num_rows
    log.info(f"  Total rows: {n_total:,}")

    available = set(pf.schema.names)
    use_cols = [c for c in NEEDED_COLS if c in available]

    chunks = []
    n_processed = 0
    n_type_a = 0
    n_type_b = 0

    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=use_cols):
        n_processed += batch.num_rows

        # SEPARATE: Type A (has organizations) vs Type B (risk_only)
        v2org_col = batch.column("v2_organizations")
        has_org = pc.and_(pc.is_valid(v2org_col),
                         pc.not_equal(pc.utf8_length(v2org_col), 0))

        collection_type_col = batch.column("collection_type") if "collection_type" in available else None
        is_type_b = (pc.equal(collection_type_col, pa.scalar("risk_only"))
                    if collection_type_col else pc.cast(has_org, pa.bool_()))

        # Type A: has v2_organizations (normal entity linking)
        type_a_mask = pc.and_(has_org, pc.invert(is_type_b))
        batch_type_a = batch.filter(type_a_mask)

        # Type B: risk_only regardless of v2_organizations
        type_b_mask = is_type_b
        batch_type_b = batch.filter(type_b_mask)

        # Process Type A with full pipeline
        if batch_type_a.num_rows > 0:
            n_type_a += batch_type_a.num_rows
            df_a = batch_type_a.to_pandas()
            df_a["entity_ids"] = _vectorized_company_match(df_a["v2_organizations"], ac_automaton, keyword_to_cid)
            # ... rest of processing ...
            chunks.append(df_a)
            del df_a

        # Process Type B separately (no entity filtering)
        if batch_type_b.num_rows > 0:
            n_type_b += batch_type_b.num_rows
            df_b = batch_type_b.to_pandas()
            # For Type B, entity_ids is empty list (no v2_organizations to match)
            df_b["entity_ids"] = [[] for _ in range(len(df_b))]
            # ... rest of processing (tone, risk_types, etc.) ...
            chunks.append(df_b)
            del df_b

        gc.collect()

    log.info(f"  Type A (entity-matched): {n_type_a:,}")
    log.info(f"  Type B (risk_only, no entity required): {n_type_b:,}")
    log.info(f"  Total kept: {n_type_a + n_type_b:,} / {n_total:,}")

    # ... rest of merging and output ...
```

### Testing the Fix

Add unit test to verify Type B preservation:

```python
def test_type_b_preservation():
    """Test that Type B articles with null v2_organizations are preserved."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tempfile

    # Create test data
    test_data = {
        'event_id': ['a', 'b', 'c'],
        'gkg_id': ['1', '2', '3'],
        'article_url': ['url_a', 'url_b', 'url_c'],
        'v2_organizations': ['CATL,1', None, 'BYD,2'],  # b has null v2_organizations
        'v2_themes': ['theme', 'theme', 'theme'],
        'v2_tone': ['1.0', '0.0', '2.0'],
        'v2_locations': ['CH#FIPS', '', 'CH#FIPS'],
        'collection_type': ['standard', 'risk_only', 'standard'],  # b is Type B
        'gkg_date': ['20260101', '20260101', '20260101'],
        'source_domain': ['news.com', 'news.com', 'news.com'],
    }

    batch = pa.RecordBatch.from_pydict({k: pa.array(v) for k, v in test_data.items()})

    # Run processing (should preserve event 'b' despite null v2_organizations)
    result = _process_single_parquet_batch(batch, ...)

    # Assertions
    assert len(result) == 3, f"Expected 3 events, got {len(result)}"

    # Event b should be preserved (Type B)
    event_b = result[result['event_id'] == 'b'].iloc[0]
    assert event_b['entity_ids'] == [], f"Event b should have empty entity_ids"
    assert event_b['collection_type'] == 'risk_only'

    log.info("✅ Type B preservation test PASSED")
```

---

## Fix 2: Add Missing Chinese Aliases (HIGH)

### Current State
8 Chinese companies lack Chinese-language aliases.

### Changes to `configs/company_aliases.yaml`

Add these entries:

```yaml
"002460.SZ":   # Ganfeng Lithium
  - Ganfeng
  - Ganfeng Lithium
  - Ganfeng Lithium Co
  - Jiangxi Ganfeng
  - Jiangxi Ganfeng Lithium
  - 赣锋锂业
  - 赣锋
  - 江西赣锋

"002466.SZ":   # Tianqi Lithium
  - Tianqi
  - Tianqi Lithium
  - Tianqi Lithium Corp
  - Sichuan Tianqi
  - Tianqi Lithium Industries
  - 天齐锂业
  - 天齐
  - 四川天齐

"600006.SH":   # Dongfeng Motor
  - Dongfeng
  - Dongfeng Motor
  - Dongfeng Motor Corp
  - Dongfeng Motor Group
  - Dongfeng Auto
  - 东风汽车
  - 东风
  - 东风汽车集团

"600362.SH":   # Jiangxi Copper
  - Jiangxi Copper
  - Jiangxi Copper Company
  - Jiangxi Copper Co
  - Jiangxi Copper Corp
  - Jiangxi Copper Group
  - 江西铜业
  - 江铜
  - 江西铜业集团

"601238.SH":   # GAC Group
  - GAC
  - GAC Group
  - GAC Motor
  - Guangzhou Automobile
  - GAC Aion
  - Guangzhou Auto
  - 广汽集团
  - 广汽
  - 广州汽车

"603799.SH":   # Huayou Cobalt
  - Huayou
  - Huayou Cobalt
  - Zhejiang Huayou
  - Huayou Cobalt Co
  - Zhejiang Huayou Cobalt
  - 华友钴业
  - 华友
  - 浙江华友

"603993.SH":   # CMOC
  - CMOC
  - CMOC Group
  - China Moly
  - China Molybdenum
  - CMOC Group Limited
  - 中钼
  - 中国钼业
  - 中国钼矿
```

### Note on 002340.SZ (GEM)
GEM already has minimal English aliases intentionally (bare "GEM" is common English word).
Adding Chinese: 浦城新能, 浦城 (but verify these are accurate—GEM is a materials company)

---

## Fix 3: Add Variant Aliases for Low-Coverage Companies (HIGH)

### Add to `configs/company_aliases.yaml`

```yaml
"RIVN":        # Rivian (currently 2 aliases → add variants)
  - Rivian
  - Rivian Automotive
  - Rivian Motors
  - Rivian Group
  - Rivian Inc

"ALB":         # Albemarle (currently 3 aliases → add variants)
  - Albemarle
  - Albemarle Corp
  - Albemarle Corporation
  - Albemarle Inc
  - ALB

"GLEN.L":      # Glencore (currently 3 aliases → add variants)
  - Glencore
  - Glencore PLC
  - Glencore International
  - Glencore Ltd
  - Glencore Group

"LCID":        # Lucid (currently 3 aliases → add variants)
  - Lucid Motors
  - Lucid Group
  - Lucid
  - Lucid Inc
  - Lucid Technologies
```

### Validation Test
After adding aliases, run this test:

```python
def test_alias_coverage_improvement():
    """Verify that newly added aliases match common news patterns."""
    test_patterns = [
        ("Rivian Motors announced", "RIVN"),
        ("Rivian Inc sued", "RIVN"),
        ("Albemarle Inc results", "ALB"),
        ("Glencore Ltd trading", "GLEN.L"),
        ("Lucid Technologies unveils", "LCID"),
    ]

    patterns, alias_to_cid = _build_alias_patterns(universe, extra_aliases)

    for text, expected_cid in test_patterns:
        hits = set()
        for pat, cid in patterns:
            if pat.search(text):
                hits.add(cid)
        assert expected_cid in hits, f"Failed to match '{text}' to {expected_cid}"

    log.info(f"✅ Alias coverage test PASSED ({len(test_patterns)} patterns)")
```

---

## Fix 4: Reconcile SQM Data Inconsistency (MEDIUM)

### Option A: Remove SQM from `company_aliases.yaml` (PREFERRED)

Simply delete this section from the YAML:

```yaml
"SQM":         # ← DELETE THIS ENTIRE BLOCK
  - SQM
  - Sociedad Quimica y Minera
  - Sociedad Quimica
```

**Rationale:** SQM is not in the 45-company universe. No stock price data, CAR analysis, or other downstream features exist for it. Matching will only cause data corruption.

### Option B: Add SQM to Universe (If needed)

If SQM should be included, add row to `data/universe/univers_final.csv`:

```csv
SQM,Sociedad Quimica y Minera,Chile,Americas,Upstream/Refining,Lithium (upstream),True,SQM,NYSE,SQM,Lithium supplier,핵심 공급망 전파 경로 노드,
```

Then ensure price data is available for SQM (ticker: SQM on NYSE).

### Add Validation Check

Add this to the pipeline to prevent future mismatches:

```python
def validate_alias_universe_consistency():
    """Verify that all aliases companies are in the universe."""
    import yaml
    import pandas as pd

    with open('configs/company_aliases.yaml', 'r') as f:
        aliases = yaml.safe_load(f)

    universe = pd.read_csv('data/universe/univers_final.csv')
    universe_ids = set(universe['company_id'].tolist())
    aliases_ids = set(aliases.keys())

    # Check for extra companies in aliases
    extra = aliases_ids - universe_ids
    if extra:
        raise ValueError(
            f"Aliases defined for companies not in universe: {extra}\n"
            f"Either remove from aliases or add to universe."
        )

    # Check for missing companies in aliases
    missing = universe_ids - aliases_ids
    if missing:
        raise ValueError(
            f"No aliases defined for universe companies: {missing}\n"
            f"Add aliases or remove from universe."
        )

    log.info(f"✅ Alias-universe consistency check PASSED ({len(universe_ids)} companies)")

# Run in main pipeline
validate_alias_universe_consistency()
```

---

## Implementation Priority & Timeline

| Fix | Severity | Effort | Timeline | Depends On |
|-----|----------|--------|----------|-----------|
| 1. PyArrow Bug | CRITICAL | Medium | 1-2 days | None |
| 2. Chinese Aliases | HIGH | Low | 1 day | None (but test after #1) |
| 3. Variant Aliases | HIGH | Low | 1 day | None (but test after #1) |
| 4. SQM Reconciliation | MEDIUM | Very Low | 0.5 days | None |

**Suggested Implementation Order:**
1. Fix PyArrow bug (blocking issue)
2. Add validation check (catches future mismatches)
3. Add Chinese aliases
4. Add variant aliases
5. Clean up SQM (delete from YAML)

**Total effort: ~3-4 days**

---

## Verification Checklist

After implementing all fixes:

- [ ] Type B articles with null v2_organizations are preserved in output
- [ ] Type B preservation count is logged and > 0
- [ ] Chinese news sample test achieves > 90% match rate
- [ ] Variant alias test passes for all 6 low-coverage companies
- [ ] SQM is either removed from aliases or added to universe
- [ ] Alias-universe consistency validation passes
- [ ] Full pipeline re-run shows increased entity coverage
- [ ] No regression in existing entity matching quality
- [ ] DQ report updated with new metrics

---

## Related Code Locations

| Component | File | Lines | Notes |
|-----------|------|-------|-------|
| PyArrow Filter | gdelt_bq.py | 559-565 | Stage 0 - needs fix |
| Type B Preservation | gdelt_bq.py | 579-580 | Depends on fix #1 |
| Alias Building | gdelt_bq.py | 41-99 | Uses company_aliases.yaml |
| Aho-Corasick Build | gdelt_bq.py | 266-296 | Uses alias patterns |
| Company Matching | gdelt_bq.py | 299-326 | Uses AC automaton |
| Alias Config | company_aliases.yaml | 1-359 | Add/remove aliases here |
| Universe | univers_final.csv | 1-47 | 45 companies + header |
| Main Entry | gdelt_bq.py | 732-786 | ingest_gdelt_bq() function |

