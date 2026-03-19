"""
BigQuery GKG Parquet → Pipeline Bridge
=======================================
BigQuery에서 수집한 연도별 GKG parquet 파일을 읽어
파이프라인의 risk_events 포맷으로 변환하는 브리지 모듈.

핵심 역할:
  1) 연도별 parquet 로드 (gdelt_gkg_20XX.parquet)
  2) V2Organizations → company_ids 추출 (alias regex)
  3) V2Tone → tone + severity 계산
  4) V2Themes → risk_types 분류
  5) V2Locations → country_ids 추출
  6) risk_events DataFrame 포맷으로 변환

두 가지 출력:
  - articles DataFrame (기존 build_risk_events 호환, 선택적)
  - risk_events DataFrame (직접 사용, 권장)

사용법:
  from src.ingest.gdelt_bq import ingest_gdelt_bq
  risk_events = ingest_gdelt_bq(cfg)
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml

log = logging.getLogger("gdelt_bq_bridge")


# ──────────────────────── Company Alias Matching ────────────────────────

def _build_alias_patterns(
    universe: pd.DataFrame,
    extra_aliases: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[Tuple[re.Pattern, str]], Dict[str, str]]:
    """
    Universe + extra_aliases → compiled regex patterns for company matching.
    Returns:
        patterns: [(compiled_pattern, company_id), ...]
        alias_to_cid: {alias_lower: company_id}
    """
    if extra_aliases is None:
        extra_aliases = _default_extra_aliases()

    company_aliases: Dict[str, List[str]] = {}
    alias_to_cid: Dict[str, str] = {}

    universe_ids = set()
    for _, row in universe.iterrows():
        cid = str(row["company_id"]).strip()
        cname = str(row.get("canonical_name", "")).strip()
        universe_ids.add(cid)
        aliases = set()
        if cname and cname != "nan":
            aliases.add(cname)
        for a in extra_aliases.get(cid, []):
            aliases.add(a)
        aliases = {a for a in aliases if len(a) >= 3}
        if aliases:
            company_aliases[cid] = sorted(aliases)
            for a in aliases:
                alias_to_cid[a.lower()] = cid

    for cid, alias_list in extra_aliases.items():
        if cid not in universe_ids:
            valid = [a for a in alias_list if len(a) >= 3]
            if valid:
                company_aliases[cid] = valid
                for a in valid:
                    alias_to_cid[a.lower()] = cid

    patterns = []
    for cid, aliases in company_aliases.items():
        for alias in aliases:
            try:
                pat = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                patterns.append((pat, cid))
            except re.error:
                pass

    log.info(f"Alias patterns: {len(company_aliases)} companies, {len(patterns)} patterns")
    return patterns, alias_to_cid


def _default_extra_aliases() -> Dict[str, List[str]]:
    """collect_gdelt_bigquery.py와 동일한 166개 alias 패턴."""
    return {
        # ── Cell/Pack ──
        "300750.SZ": ["CATL", "Contemporary Amperex", "Contemporary Amperex Technology",
                       "Contemporary Amperex Technology Co"],
        "373220.KS": ["LG Energy", "LGES", "LG Energy Solution", "LG Energy Solutions"],
        "1211.HK": ["BYD", "BYD Company", "BYD Auto", "BYD Co"],
        "051910.KS": ["LG Chem", "LG Chemical", "LG Chem Ltd"],
        "300014.SZ": ["EVE Energy", "EVE Energy Co"],
        "300207.SZ": ["Sunwoda", "Sunwoda Electronic", "Sunwoda Electronic Co"],
        "3931.HK": ["CALB", "CALB Group", "CALB Co",
                     "China Aviation Lithium Battery", "China Aviation Lithium"],
        "002074.SZ": ["Gotion", "Gotion High-Tech", "Guoxuan", "Guoxuan High-Tech"],
        # ── Materials ──
        "247540.KQ": ["EcoPro", "EcoPro BM", "EcoPro Co"],
        "066970.KS": ["L&F", "L and F", "LnF", "LF Co"],
        "003670.KS": ["POSCO Future", "POSCO Future M", "POSCO Chemical", "POSCO Holdings"],
        "300919.SZ": ["CNGR", "CNGR Advanced Material", "CNGR Advanced"],
        "603659.SH": ["Putailai", "Putailai New Energy"],
        "300037.SZ": ["Capchem", "Shenzhen Capchem", "Capchem Technology"],
        "600110.SH": ["Nuode", "Nuode Investment", "Nuode New Material"],
        "688388.SH": ["Jiayuan", "Jiayuan International", "Jiayuan Technology"],
        "002340.SZ": ["GEM Co", "GEM Co Ltd"],
        "920185.BJ": ["BTR", "BTR New Material", "BTR New Energy"],
        "3407.T": ["Asahi Kasei", "Asahi Kasei Corp", "Asahi Kasei Corporation"],
        # ── Upstream/Refining ──
        "002460.SZ": ["Ganfeng", "Ganfeng Lithium", "Ganfeng Lithium Co",
                       "Jiangxi Ganfeng", "Jiangxi Ganfeng Lithium"],
        "002466.SZ": ["Tianqi", "Tianqi Lithium", "Tianqi Lithium Corp",
                       "Sichuan Tianqi", "Tianqi Lithium Industries"],
        "603799.SH": ["Huayou", "Huayou Cobalt", "Zhejiang Huayou",
                       "Huayou Cobalt Co", "Zhejiang Huayou Cobalt"],
        "603993.SH": ["CMOC", "CMOC Group", "China Moly", "China Molybdenum",
                       "CMOC Group Limited"],
        "600362.SH": ["Jiangxi Copper", "Jiangxi Copper Company",
                       "Jiangxi Copper Co", "Jiangxi Copper Corp"],
        "ALB": ["Albemarle", "Albemarle Corp", "Albemarle Corporation"],
        "SQM": ["SQM", "Sociedad Quimica y Minera", "Sociedad Quimica"],
        "GLEN.L": ["Glencore", "Glencore PLC", "Glencore International"],
        # ── OEM ──
        "BMW.DE": ["BMW", "Bayerische Motoren", "BMW Group", "BMW AG"],
        "VOW3.DE": ["Volkswagen", "VW", "Volkswagen AG", "Volkswagen Group", "VW Group"],
        "MBG.DE": ["Mercedes-Benz", "Mercedes Benz", "Daimler", "Mercedes-Benz Group"],
        "TSLA": ["Tesla", "Tesla Motors", "Tesla Inc"],
        "F": ["Ford Motor", "Ford Motor Company"],
        "GM": ["General Motors", "General Motors Company"],
        "LCID": ["Lucid Motors", "Lucid Group", "Lucid"],
        "RIVN": ["Rivian", "Rivian Automotive"],
        "7267.T": ["Honda Motor", "Honda", "Honda Motor Co"],
        "7203.T": ["Toyota Motor", "Toyota", "Toyota Motor Corp"],
        "0175.HK": ["Geely", "Geely Automobile", "Geely Auto", "Zhejiang Geely"],
        "600733.SH": ["BAIC", "BAIC BluePark", "BAIC Motor", "BAIC Group",
                       "Beijing Automotive"],
        "000625.SZ": ["Changan", "Changan Auto", "Changan Automobile",
                       "Chongqing Changan"],
        "600006.SH": ["Dongfeng", "Dongfeng Motor", "Dongfeng Motor Corp",
                       "Dongfeng Motor Group", "Dongfeng Auto"],
        "601238.SH": ["GAC", "GAC Group", "GAC Motor", "Guangzhou Automobile",
                       "GAC Aion", "Guangzhou Auto"],
        "601633.SH": ["Great Wall Motor", "Great Wall Motors", "GWM",
                       "Great Wall Motor Company"],
        "600418.SH": ["JAC Motors", "JAC", "Anhui Jianghuai", "JAC Automobile"],
        "LICY": ["Li-Cycle", "Li Cycle", "Li-Cycle Holdings"],
        # ── Chemical ──
        "BAS.DE": ["BASF", "BASF SE", "BASF AG", "BASF Corporation"],
    }


# ──────────────────────── Risk Classification ────────────────────────

def _load_risk_patterns(keywords_yaml: Path) -> List[Tuple[str, re.Pattern]]:
    """risk_keywords.yaml → [(risk_type, compiled_pattern), ...]"""
    try:
        data = yaml.safe_load(keywords_yaml.read_text(encoding="utf-8"))
    except Exception:
        log.warning("risk_keywords.yaml 로드 실패, fallback 사용")
        fallback = [
            ("geopolitics", re.compile(r"sanction|embargo|tariff|trade war", re.IGNORECASE)),
            ("logistics", re.compile(r"port disruption|shipping delay|blockade", re.IGNORECASE)),
            ("natural", re.compile(r"earthquake|tsunami|flood|wildfire", re.IGNORECASE)),
        ]
        return fallback

    patterns = []
    for risk_type in data:
        section = data[risk_type]
        if not isinstance(section, dict):
            continue
        for kw in section.get("keywords", []):
            if isinstance(kw, dict) and "pattern" in kw:
                try:
                    pat = re.compile(kw["pattern"], re.IGNORECASE)
                    patterns.append((risk_type, pat))
                except re.error:
                    pass
            elif isinstance(kw, str):
                try:
                    pat = re.compile(re.escape(kw), re.IGNORECASE)
                    patterns.append((risk_type, pat))
                except re.error:
                    pass

    log.info(f"Risk patterns loaded: {len(patterns)} from {len(data)} types")
    return patterns


# ──────────────────────── Post-Processing Functions ────────────────────────

def _extract_company_ids(v2orgs: str, patterns: List[Tuple[re.Pattern, str]]) -> List[str]:
    """V2Organizations → list of matched company_ids."""
    if pd.isna(v2orgs) or not v2orgs:
        return []
    # V2Organizations format: "OrgName,offset;OrgName,offset;..."
    org_names = set()
    for entry in str(v2orgs).split(";"):
        if "," in entry:
            name = entry.rsplit(",", 1)[0].strip()
            if name:
                org_names.add(name.upper())
    if not org_names:
        return []
    text = " | ".join(org_names)
    hits = set()
    for pat, cid in patterns:
        if pat.search(text):
            hits.add(cid)
    return sorted(hits)


def _parse_tone(v2tone) -> float:
    """V2Tone → float tone value (first field)."""
    if pd.isna(v2tone) or not v2tone:
        return 0.0
    try:
        return float(str(v2tone).split(",")[0])
    except (ValueError, IndexError):
        return 0.0


def _compute_severity_tone_only(tone: float) -> float:
    """Tone 기반 severity (1.0~5.0). 부정적 톤일수록 severity 높음."""
    norm_tone = np.clip((10 - tone) / 20.0, 0, 1)
    return float(np.clip(1.0 + norm_tone * 4.0, 1.0, 5.0))


def _classify_risk(v2themes: str, risk_patterns: List[Tuple[str, re.Pattern]]) -> List[str]:
    """V2Themes → list of risk_type labels."""
    if pd.isna(v2themes) or not v2themes:
        return ["other"]
    text = str(v2themes)
    hits = set()
    for rtype, pat in risk_patterns:
        if pat.search(text):
            hits.add(rtype)
    return sorted(hits) if hits else ["other"]


FIPS_MAP = {
    "CI": "Chile", "AS": "Australia", "ID": "Indonesia",
    "CG": "Congo", "CF": "Congo",
    "CH": "China", "KS": "South_Korea", "GM": "Germany",
    "JA": "Japan", "US": "USA", "UK": "UK", "CA": "Canada",
}


def _extract_countries(v2loc) -> str:
    """V2Locations → semicolon-separated country names."""
    if pd.isna(v2loc) or not v2loc:
        return ""
    hits = set()
    for entry in str(v2loc).split(";"):
        if "," in entry:
            entry = entry.rsplit(",", 1)[0]
        parts = entry.split("#")
        if len(parts) >= 3:
            fips = parts[2].strip()
            if fips in FIPS_MAP:
                hits.add(FIPS_MAP[fips])
    return ";".join(sorted(hits))


def _event_id(gkg_id: str, url: str) -> str:
    """Stable event_id from gkg_id + url."""
    raw = (str(gkg_id) + "|" + str(url)).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


# ──────────────────────── Main Bridge ────────────────────────

def _list_bq_parquets(parquet_dir: Path) -> List[Path]:
    """연도별 GKG parquet 파일 목록 (gdelt_gkg_articles.parquet 제외)."""
    files = sorted(parquet_dir.glob("gdelt_gkg_*.parquet"))
    # 빈 파일이나 메타 파일 제외
    files = [f for f in files if f.stat().st_size > 1024 and "articles" not in f.name]
    if not files:
        raise FileNotFoundError(f"No GKG parquet files in {parquet_dir}")
    return files


NEEDED_COLS = [
    "gkg_id", "gkg_date", "source_domain", "article_url",
    "v2_organizations", "v2_themes", "v2_tone", "v2_locations",
    "collection_type",
]
BATCH_ROWS = 100_000  # PyArrow batch size


def _build_company_patterns(patterns: List[Tuple[re.Pattern, str]]) -> Tuple[re.Pattern, Dict[str, re.Pattern]]:
    """
    2단계 최적화용 패턴 빌드:
    1) mega_pattern: 모든 alias OR (pre-filter, 1회 호출)
    2) cid_patterns: company별 OR 패턴 (매칭 행에서만 세부 분류)
    """
    from collections import defaultdict
    cid_to_aliases = defaultdict(list)
    all_pats = []
    for pat, cid in patterns:
        kw = re.sub(r'\\b|\\B', '', pat.pattern).strip()
        cid_to_aliases[cid].append(kw)
        all_pats.append(kw)

    mega = re.compile("|".join(all_pats), re.IGNORECASE)
    cid_patterns = {cid: re.compile("|".join(aliases), re.IGNORECASE)
                    for cid, aliases in cid_to_aliases.items()}
    return mega, cid_patterns


def _vectorized_company_match(
    v2orgs_series: pd.Series,
    mega_pattern: re.Pattern,
    cid_patterns: Dict[str, re.Pattern],
) -> pd.Series:
    """
    2단계 vectorized:
    1) mega OR 1회 str.contains → 후보행 필터 (비매칭 60%+ 제거)
    2) 후보행에서만 46개 company별 str.contains
    """
    text = v2orgs_series.fillna("").astype(str).str.upper()
    n = len(text)

    # 1단계: 1회 pre-filter
    any_match = text.str.contains(mega_pattern, na=False, regex=True).values
    match_indices = np.where(any_match)[0]

    entity_ids = [[] for _ in range(n)]
    if len(match_indices) == 0:
        return pd.Series(entity_ids, index=v2orgs_series.index)

    log.info(f"    Pre-filter: {len(match_indices):,}/{n:,} matched ({100*len(match_indices)/n:.1f}%)")

    # 2단계: 매칭 행만 세부 분류
    matched_text = text.iloc[match_indices]
    for cid, pat in cid_patterns.items():
        hits = matched_text.str.contains(pat, na=False, regex=True).values
        for local_idx, global_idx in enumerate(match_indices):
            if hits[local_idx]:
                entity_ids[global_idx].append(cid)

    for i in match_indices:
        entity_ids[i] = sorted(entity_ids[i])

    return pd.Series(entity_ids, index=v2orgs_series.index)


def _vectorized_tone_severity(v2tone_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Vectorized tone parsing + severity computation."""
    tone = v2tone_series.fillna("0").astype(str).str.split(",").str[0]
    tone = pd.to_numeric(tone, errors="coerce").fillna(0.0)
    norm_tone = np.clip((10 - tone.values) / 20.0, 0, 1)
    severity = np.clip(1.0 + norm_tone * 4.0, 1.0, 5.0).round(3)
    return tone, pd.Series(severity, index=v2tone_series.index)


def _vectorized_risk_classify(
    v2themes_series: pd.Series,
    risk_patterns: List[Tuple[str, re.Pattern]],
) -> pd.Series:
    """Vectorized risk classification."""
    text = v2themes_series.fillna("").astype(str)

    # 각 risk_type별로 패턴 그룹핑
    from collections import defaultdict
    type_to_patterns = defaultdict(list)
    for rtype, pat in risk_patterns:
        type_to_patterns[rtype].append(pat.pattern)

    n = len(text)
    rtypes = list(type_to_patterns.keys())
    match_arrays = {}
    for rtype, pats in type_to_patterns.items():
        combined = "|".join(pats)
        try:
            match_arrays[rtype] = text.str.contains(combined, case=False, na=False, regex=True).values
        except re.error:
            result = np.zeros(n, dtype=bool)
            for p in pats:
                try:
                    result |= text.str.contains(p, case=False, na=False, regex=True).values
                except re.error:
                    pass
            match_arrays[rtype] = result

    risk_types = []
    for i in range(n):
        hits = sorted([rt for rt in rtypes if match_arrays[rt][i]])
        risk_types.append(",".join(hits) if hits else "other")

    return pd.Series(risk_types, index=v2themes_series.index)


def _vectorized_event_ids(gkg_ids: pd.Series, urls: pd.Series) -> pd.Series:
    """Vectorized event_id generation."""
    combined = gkg_ids.fillna("").astype(str) + "|" + urls.fillna("").astype(str)
    return combined.apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())


def _process_single_parquet(
    parquet_path: Path,
    patterns: List[Tuple[re.Pattern, str]],
    risk_patterns: List[Tuple[str, re.Pattern]],
    mega_pattern: re.Pattern,
    cid_patterns: Dict[str, re.Pattern],
    filter_no_entity: bool = True,
) -> pd.DataFrame:
    """
    단일 연도 parquet → risk_events.

    3단계 속도 최적화:
      0) PyArrow 레벨: v2_organizations null/빈값 행 제거 (전체의 ~40% 제거)
      1) mega OR 1회 pre-filter (나머지의 ~90% 제거)
      2) 살아남은 ~5%에만 46개 company별 매칭

    490만행 → ~25만행 수준으로 줄어듦 → 1~2분/파일
    """
    import gc
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    log.info(f"Processing {parquet_path.name}...")
    pf = pq.ParquetFile(parquet_path)
    n_total = pf.metadata.num_rows
    log.info(f"  Total rows: {n_total:,}")

    available = set(pf.schema.names)
    use_cols = [c for c in NEEDED_COLS if c in available]

    chunks = []
    n_processed = 0
    n_pyarrow_kept = 0

    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=use_cols):
        n_processed += batch.num_rows

        # ── Stage 0: PyArrow 레벨 pre-filter ──
        # v2_organizations가 null이거나 빈 문자열인 행 제거 (pandas 변환 전)
        v2org_col = batch.column("v2_organizations")
        not_null = pc.is_valid(v2org_col)
        not_empty = pc.not_equal(pc.utf8_length(pc.if_else(not_null, v2org_col, pa.scalar(""))), 0)
        mask = pc.and_(not_null, not_empty)
        batch = batch.filter(mask)

        if batch.num_rows == 0:
            continue

        n_pyarrow_kept += batch.num_rows
        df = batch.to_pandas()

        # ── Stage 1+2: mega pre-filter → 세부 매칭 ──
        df["entity_ids"] = _vectorized_company_match(df["v2_organizations"], mega_pattern, cid_patterns)

        # Filter: entity 없는 행 제거
        if filter_no_entity:
            has_entity = df["entity_ids"].apply(len) > 0
            is_risk_only = df.get("collection_type", pd.Series(dtype=str)) == "risk_only"
            df = df[has_entity | is_risk_only].copy()

        if df.empty:
            continue

        # ── 후처리 (매칭된 소수 행에만 적용) ──
        df["tone"], df["severity"] = _vectorized_tone_severity(df["v2_tone"])
        df["risk_types"] = _vectorized_risk_classify(df["v2_themes"], risk_patterns)
        df["country_ids"] = df["v2_locations"].apply(_extract_countries) if "v2_locations" in df.columns else ""
        df["event_time"] = pd.to_datetime(
            df["gkg_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce"
        )
        df["event_id"] = _vectorized_event_ids(df["gkg_id"], df["article_url"])
        df["entity_scores"] = df["entity_ids"].apply(
            lambda ids: [{"company_id": c, "score": 100, "alias": "bq_regex"} for c in ids]
        )
        df["title"] = ""
        df["finbert"] = None
        df["url"] = df["article_url"]

        out_cols = [
            "event_id", "event_time", "url", "title", "risk_types",
            "severity", "entity_ids", "entity_scores", "finbert",
            "country_ids", "tone",
        ]
        if "source_domain" in df.columns:
            out_cols.append("source_domain")
        if "collection_type" in df.columns:
            out_cols.append("collection_type")

        chunks.append(df[[c for c in out_cols if c in df.columns]].copy())
        del df

        if n_processed % 300_000 == 0:
            log.info(f"  ... {n_processed:,}/{n_total:,} scanned, "
                     f"PyArrow kept: {n_pyarrow_kept:,}, chunks: {sum(len(c) for c in chunks):,}")

    gc.collect()
    log.info(f"  PyArrow filter: {n_total:,} → {n_pyarrow_kept:,} "
             f"({100*n_pyarrow_kept/max(n_total,1):.1f}% had v2_organizations)")

    if not chunks:
        log.warning(f"  No matching rows in {parquet_path.name}")
        return pd.DataFrame()

    result = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    log.info(f"  → {len(result):,} risk_events from {parquet_path.name}")
    return result


def process_bq_to_risk_events(
    parquet_dir: Path,
    universe: pd.DataFrame,
    keywords_yaml: Path,
    extra_aliases: Optional[Dict[str, List[str]]] = None,
    filter_no_entity: bool = True,
) -> pd.DataFrame:
    """
    연도별 배치 처리 + PyArrow 청크.
    피크 메모리 ~500MB (배치 10만행 + 결과 누적).

    출력 컬럼 (risk_events 호환):
        event_id, event_time, url, title, risk_types, severity,
        entity_ids, entity_scores, finbert,
        country_ids, collection_type, source_domain
    """
    import gc

    files = _list_bq_parquets(parquet_dir)
    log.info(f"Found {len(files)} parquet files for batch processing")

    # 패턴은 한 번만 빌드 (재사용)
    patterns, _ = _build_alias_patterns(universe, extra_aliases)
    risk_patterns = _load_risk_patterns(keywords_yaml)
    mega_pattern, cid_patterns = _build_company_patterns(patterns)
    log.info(f"Mega pre-filter: {len(cid_patterns)} companies, 1 regex pass")

    # 연도별 중간 저장 디렉토리
    checkpoint_dir = parquet_dir / "risk_events_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cp_paths: List[Path] = []
    for f in files:
        # 체크포인트 있으면 스킵
        cp_path = checkpoint_dir / f"risk_{f.stem}.parquet"
        if cp_path.exists():
            log.info(f"[SKIP] {f.name} → checkpoint exists: {cp_path.name}")
            cp_paths.append(cp_path)
            continue

        chunk = _process_single_parquet(f, patterns, risk_patterns, mega_pattern, cid_patterns, filter_no_entity)
        if not chunk.empty:
            chunk.to_parquet(cp_path, index=False)
            log.info(f"  Checkpoint saved: {cp_path.name} ({len(chunk):,} rows)")
            cp_paths.append(cp_path)
        del chunk
        gc.collect()

    if not cp_paths:
        raise RuntimeError("No risk_events produced from any parquet file")

    # 메모리 효율적 합치기: 1개씩 읽어서 dedup 후 append
    log.info(f"Merging {len(cp_paths)} checkpoint files (streaming)...")
    seen_urls: set = set()
    total_rows = 0
    total_dedup = 0
    first = True

    for cp in cp_paths:
        df = pd.read_parquet(cp)
        n_before = len(df)
        # URL dedup (이전 파일과 중복 제거)
        mask = ~df["url"].isin(seen_urls)
        df = df[mask].reset_index(drop=True)
        # 내부 dedup
        df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
        seen_urls.update(df["url"].tolist())
        total_dedup += n_before - len(df)
        total_rows += len(df)

        # append 모드로 저장 (첫 파일은 새로 생성)
        tmp_out = checkpoint_dir / "_merged_risk_events.parquet"
        if first:
            df.to_parquet(tmp_out, index=False)
            first = False
        else:
            import pyarrow.parquet as pq
            existing = pq.read_table(tmp_out)
            new_table = pa.concat_tables([existing, pa.Table.from_pandas(df)])
            pq.write_table(new_table, tmp_out)
            del existing, new_table

        log.info(f"  {cp.name}: {len(df):,} rows added (dedup: {n_before - len(df):,})")
        del df
        gc.collect()

    if total_dedup > 0:
        log.info(f"Deduped total: {total_dedup:,} duplicate URLs removed")

    # 최종 결과: 파일 경로만 반환 (메모리에 전체 로드하지 않음)
    log.info(f"risk_events ready: {total_rows:,} rows (dedup removed: {total_dedup:,})")
    log.info(f"  Merged file: {tmp_out}")

    # 최종 출력 경로로 이동
    return tmp_out


def ingest_gdelt_bq(
    cfg: Dict[str, Any],
    universe: pd.DataFrame,
    out_path: Path,
    force: bool = False,
) -> pd.DataFrame:
    """
    메인 진입점: BigQuery parquet → risk_events parquet.
    연도별 배치 처리로 메모리 안전.

    Args:
        cfg: base.yaml 전체 config dict
        universe: universe DataFrame (company_id, canonical_name, ...)
        out_path: 출력 risk_events parquet 경로
        force: True면 캐시 무시하고 재처리

    Returns:
        risk_events DataFrame
    """
    if out_path.exists() and not force:
        log.info(f"Cached risk_events found: {out_path}")
        return pd.read_parquet(out_path)

    # Parquet 디렉토리 결정
    paths_cfg = cfg.get("paths", {})
    parquet_dir_rel = paths_cfg.get("gdelt_bq_parquet_dir", "data/raw/gdelt")
    root = Path(cfg.get("project", {}).get("root_dir", "."))
    parquet_dir = (root / parquet_dir_rel).resolve()

    # Keywords
    risk_cfg = cfg.get("risk", {})
    keywords_rel = risk_cfg.get("keywords_yaml", "configs/risk_keywords.yaml")
    keywords_yaml = (root / keywords_rel).resolve()

    filter_no_entity = bool(risk_cfg.get("filter_no_entity_events", False))

    # 연도별 배치 처리 → 합친 파일 경로 반환
    import shutil
    merged_path = process_bq_to_risk_events(
        parquet_dir=parquet_dir,
        universe=universe,
        keywords_yaml=keywords_yaml,
        filter_no_entity=filter_no_entity,
    )

    # 합친 파일을 최종 출력 경로로 복사
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(merged_path), str(out_path))
    log.info(f"Saved risk_events: {out_path}")

    # 경량 확인용으로 일부만 읽기
    sample = pd.read_parquet(out_path, columns=["url"])
    log.info(f"  Total rows: {len(sample):,}")

    return out_path
