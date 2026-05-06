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

    # 일반 영어 단어와 겹치는 짧은 alias 차단 (오매칭 방지)
    _COMMON_WORD_BLOCKLIST = {
        "gem", "ace", "can", "man", "ion", "air", "arc", "art",
        "bay", "bit", "cap", "cup", "era", "eve", "gap", "hub",
        "joy", "key", "lab", "map", "net", "ore", "pan", "ray",
    }

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
        aliases = {a for a in aliases
                   if len(a) >= 3 and a.lower() not in _COMMON_WORD_BLOCKLIST}
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
    """configs/company_aliases.yaml에서 alias 로드. 파일 없으면 빈 dict 반환."""
    alias_paths = [
        Path("configs/company_aliases.yaml"),
        Path(__file__).resolve().parents[2] / "configs" / "company_aliases.yaml",
    ]
    for p in alias_paths:
        if p.exists():
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    # YAML에서 key가 숫자로 파싱될 수 있으므로 str 변환
                    result = {str(k): [str(a) for a in v] for k, v in data.items()
                              if isinstance(v, list)}
                    log.info(f"Loaded {len(result)} company aliases from {p}")
                    return result
            except Exception as e:
                log.warning(f"company_aliases.yaml 로드 실패: {e}")
    log.warning("company_aliases.yaml not found, using empty aliases")
    return {}


# ──────────────────────── Risk Classification ────────────────────────

def _load_risk_patterns(keywords_yaml: Path) -> List[Tuple[str, re.Pattern]]:
    """risk_keywords.yaml → [(risk_type, compiled_pattern), ...]"""
    try:
        data = yaml.safe_load(keywords_yaml.read_text(encoding="utf-8"))
    except Exception:
        log.warning("risk_keywords.yaml 로드 실패, fallback 사용")
        fallback = [
            ("geopolitics", re.compile(r"sanction|embargo|tariff|trade war|invasion|escalat", re.IGNORECASE)),
            ("logistics", re.compile(r"port disruption|shipping delay|blockade|supply disruption|factory shutdown", re.IGNORECASE)),
            ("natural", re.compile(r"earthquake|tsunami|flood|wildfire|typhoon|hurricane", re.IGNORECASE)),
            ("regulatory", re.compile(r"regulat|recall|antitrust|subsidy|battery fire|penalty|investigat", re.IGNORECASE)),
            ("market", re.compile(r"bankrupt|layoff|demand.*declin|overcapacity|price.*crash|default", re.IGNORECASE)),
            ("labor", re.compile(r"strike.*worker|labor shortage|mine.*accident|protest.*factory", re.IGNORECASE)),
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
BATCH_ROWS = 50_000  # PyArrow batch size (reduced from 100K for memory safety)


def _build_aho_automaton(patterns: List[Tuple[re.Pattern, str]]):
    """
    Aho-Corasick automaton 빌드.
    모든 alias keyword를 한 번에 등록 → O(n) 단일 스캔으로 전체 company 매칭.
    기존 46개 × str.contains() 반복 → 1회 스캔으로 대체 (23분 → 수 초).

    주의: re.escape()가 공백/하이픈 등을 \\ 로 이스케이프하므로,
    pattern.pattern에서 모든 backslash-escape를 원래 문자로 복원해야 AC 매칭됨.
    """
    import ahocorasick_rs

    all_keywords = []
    keyword_to_cid = []

    for pat, cid in patterns:
        # re.escape()가 만든 backslash escapes 전체 제거
        # e.g. "\\bLG\\ Chem\\b" → "LG Chem"
        kw = pat.pattern
        # Step 1: \b, \B word boundaries 제거
        kw = kw.replace(r'\b', '').replace(r'\B', '')
        # Step 2: 남은 backslash escapes 복원 (\ → 원래 문자)
        # re.escape는 특수문자 앞에 \를 붙임: \  \- \& \. 등
        kw = re.sub(r'\\(.)', r'\1', kw)
        kw = kw.strip().upper()
        if kw:
            all_keywords.append(kw)
            keyword_to_cid.append(cid)

    log.info(f"Company AC automaton: {len(all_keywords)} keywords")
    ac = ahocorasick_rs.AhoCorasick(all_keywords, matchkind=ahocorasick_rs.MATCHKIND_STANDARD)
    return ac, keyword_to_cid


def _vectorized_company_match(
    v2orgs_series: pd.Series,
    ac_automaton,
    keyword_to_cid: List[str],
) -> pd.Series:
    """
    Aho-Corasick 단일 스캔으로 company 매칭.
    기존 46개 × str.contains() → 1회 스캔으로 대체.
    10만행 기준 23분 → 수 초.
    """
    texts = v2orgs_series.fillna("").astype(str).str.upper().tolist()
    n = len(texts)
    entity_ids = [[] for _ in range(n)]
    matched_count = 0

    for i, txt in enumerate(texts):
        if not txt:
            continue
        matches = ac_automaton.find_matches_as_indexes(txt)
        if matches:
            matched_count += 1
            cids = set()
            for pattern_idx, _start, _end in matches:
                cids.add(keyword_to_cid[pattern_idx])
            entity_ids[i] = sorted(cids)

    log.info(f"    Aho-Corasick: {matched_count:,}/{n:,} matched ({100*matched_count/n:.1f}%)")
    return pd.Series(entity_ids, index=v2orgs_series.index)


def _vectorized_tone_severity(v2tone_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Vectorized tone parsing + severity computation."""
    tone = v2tone_series.fillna("0").astype(str).str.split(",").str[0]
    tone = pd.to_numeric(tone, errors="coerce").fillna(0.0)
    norm_tone = np.clip((10 - tone.values) / 20.0, 0, 1)
    severity = np.clip(1.0 + norm_tone * 4.0, 1.0, 5.0).round(3)
    return tone, pd.Series(severity, index=v2tone_series.index)


def _expand_regex_to_literals(pattern_str: str) -> List[str]:
    """
    Regex 패턴을 Aho-Corasick용 리터럴 키워드 목록으로 전개.

    처리 순서:
      1) \\b, \\B 제거
      2) | (alternation) 기준으로 분리
      3) 각 fragment에서 regex quantifier/metachar 정리:
         - [s]? → 두 변형 (있는/없는) 전개
         - [sz] → 두 변형 전개
         - [es]? → 두 변형 전개
         - .* / .? → 공백 1칸으로 치환 (AC가 공백 포함 매칭)
         - 남은 [], () 등은 제거
      4) 2글자 미만 키워드 필터
    """
    # Step 1: strip word boundaries
    s = pattern_str.replace(r'\b', '').replace(r'\B', '').strip()

    # Step 2: split on | (top-level alternation)
    fragments = s.split('|')

    literals = []
    for frag in fragments:
        frag = frag.strip()
        if not frag:
            continue
        # Step 3: expand common quantifier patterns
        expanded = _expand_fragment(frag)
        literals.extend(expanded)

    # Step 4: filter too-short, deduplicate, uppercase
    seen = set()
    result = []
    for lit in literals:
        lit = lit.strip().upper()
        if len(lit) >= 2 and lit not in seen:
            seen.add(lit)
            result.append(lit)
    return result


def _expand_fragment(frag: str) -> List[str]:
    """
    단일 regex fragment → 리터럴 변형 리스트.
    예: "sanction[s]?" → ["sanction", "sanctions"]
        "demand.*declin" → ["demand declin"]
        "wind.?down" → ["wind down", "winddown"]
    """
    results = [""]

    i = 0
    while i < len(frag):
        # [s]? 패턴
        m = re.match(r'\[([a-zA-Z]+)\]\??', frag[i:])
        if m:
            chars = m.group(1)
            optional = '?' in m.group(0)
            new_results = []
            for r in results:
                for c in chars:
                    new_results.append(r + c)
                if optional:
                    new_results.append(r)  # 없는 변형
            results = new_results
            i += m.end()
            continue

        # .* or .+ → 공백으로 치환 (AC 리터럴 매칭용)
        if frag[i] == '.' and i + 1 < len(frag) and frag[i + 1] in ('*', '+'):
            results = [r + ' ' for r in results]
            i += 2
            continue

        # .? → 공백 또는 빈문자열 두 변형
        if frag[i] == '.' and i + 1 < len(frag) and frag[i + 1] == '?':
            new_results = []
            for r in results:
                new_results.append(r + ' ')
                new_results.append(r)
            results = new_results
            i += 2
            continue

        # 단독 . → 공백
        if frag[i] == '.' and (i + 1 >= len(frag) or frag[i + 1] not in ('*', '+', '?')):
            results = [r + ' ' for r in results]
            i += 1
            continue

        # 남은 metachar 무시
        if frag[i] in ('(', ')', '^', '$'):
            i += 1
            continue

        # 일반 문자
        results = [r + frag[i] for r in results]
        i += 1

    # 정리: 연속 공백 → 단일 공백, 앞뒤 공백 제거
    cleaned = []
    for r in results:
        r = re.sub(r'\s+', ' ', r).strip()
        if r:
            cleaned.append(r)
    return cleaned


def _build_risk_automaton(risk_patterns: List[Tuple[str, re.Pattern]]):
    """
    Risk keyword용 Aho-Corasick automaton 빌드.

    regex 패턴을 리터럴 키워드로 전개하여 AC에 등록.
    예: ("market", re.compile("bankrupt|insolvency|chapter 11"))
      → AC keywords: ["BANKRUPT", "INSOLVENCY", "CHAPTER 11"]
         각각 rtype "market"에 매핑
    """
    import ahocorasick_rs
    keywords = []
    keyword_to_rtype = []
    for rtype, pat in risk_patterns:
        literals = _expand_regex_to_literals(pat.pattern)
        for lit in literals:
            keywords.append(lit)
            keyword_to_rtype.append(rtype)

    log.info(f"Risk AC automaton: {len(keywords)} literal keywords "
             f"from {len(risk_patterns)} regex patterns")
    ac = ahocorasick_rs.AhoCorasick(keywords, matchkind=ahocorasick_rs.MATCHKIND_STANDARD)
    return ac, keyword_to_rtype


def _vectorized_risk_classify(
    v2themes_series: pd.Series,
    risk_ac,
    risk_keyword_to_rtype: List[str],
) -> pd.Series:
    """Aho-Corasick 기반 risk classification. 8개 str.contains → 1회 스캔."""
    texts = v2themes_series.fillna("").astype(str).str.upper().tolist()
    n = len(texts)
    risk_types = ["other"] * n

    for i, txt in enumerate(texts):
        if not txt:
            continue
        matches = risk_ac.find_matches_as_indexes(txt)
        if matches:
            rtypes = sorted(set(risk_keyword_to_rtype[idx] for idx, _s, _e in matches))
            risk_types[i] = ",".join(rtypes)

    return pd.Series(risk_types, index=v2themes_series.index)


def _vectorized_countries(v2loc_series: pd.Series) -> pd.Series:
    """Vectorized country extraction. row-by-row apply → batch처리."""
    texts = v2loc_series.fillna("").astype(str).tolist()
    results = [""] * len(texts)
    for i, txt in enumerate(texts):
        if not txt:
            continue
        hits = set()
        for entry in txt.split(";"):
            if "," in entry:
                entry = entry.rsplit(",", 1)[0]
            parts = entry.split("#")
            if len(parts) >= 3:
                fips = parts[2].strip()
                if fips in FIPS_MAP:
                    hits.add(FIPS_MAP[fips])
        if hits:
            results[i] = ";".join(sorted(hits))
    return pd.Series(results, index=v2loc_series.index)


def _vectorized_event_ids(gkg_ids: pd.Series, urls: pd.Series) -> pd.Series:
    """Batch MD5 generation. apply(lambda) → list comprehension."""
    gids = gkg_ids.fillna("").astype(str).tolist()
    us = urls.fillna("").astype(str).tolist()
    hashes = [hashlib.md5((g + "|" + u).encode("utf-8")).hexdigest() for g, u in zip(gids, us)]
    return pd.Series(hashes, index=gkg_ids.index)


def _process_single_parquet(
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
        # 단, collection_type == "risk_only"인 Type B 기사는 보존
        v2org_col = batch.column("v2_organizations")
        not_null = pc.is_valid(v2org_col)
        not_empty = pc.not_equal(pc.utf8_length(pc.if_else(not_null, v2org_col, pa.scalar(""))), 0)
        has_v2org = pc.and_(not_null, not_empty)

        # Type B (risk_only) 보존: collection_type 컬럼이 있으면 risk_only 행 유지
        if "collection_type" in available:
            ct_col = batch.column("collection_type")
            is_risk_only = pc.equal(pc.if_else(pc.is_valid(ct_col), ct_col, pa.scalar("")), pa.scalar("risk_only"))
            mask = pc.or_(has_v2org, is_risk_only)
        else:
            mask = has_v2org
        batch = batch.filter(mask)

        if batch.num_rows == 0:
            continue

        n_pyarrow_kept += batch.num_rows
        df = batch.to_pandas()

        # ── Aho-Corasick 단일 스캔 매칭 ──
        df["entity_ids"] = _vectorized_company_match(df["v2_organizations"], ac_automaton, keyword_to_cid)

        # ── Type B fallback: v2_organizations 매칭 실패 시 article_url로 재시도 ──
        no_match_mask = df["entity_ids"].apply(len) == 0
        if no_match_mask.any() and "article_url" in df.columns:
            url_matches = _vectorized_company_match(df.loc[no_match_mask, "article_url"], ac_automaton, keyword_to_cid)
            df.loc[no_match_mask, "entity_ids"] = url_matches

        # Filter: entity 없는 행 제거
        if filter_no_entity:
            has_entity = df["entity_ids"].apply(len) > 0
            is_risk_only = df.get("collection_type", pd.Series(dtype=str)) == "risk_only"
            df = df[has_entity | is_risk_only].copy()

        if df.empty:
            continue

        # ── 후처리 (Aho-Corasick + batch) ──
        df["tone"], df["severity"] = _vectorized_tone_severity(df["v2_tone"])
        df["risk_types"] = _vectorized_risk_classify(df["v2_themes"], risk_ac, risk_keyword_to_rtype)
        df["country_ids"] = _vectorized_countries(df["v2_locations"]) if "v2_locations" in df.columns else ""
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
    ac_automaton, keyword_to_cid = _build_aho_automaton(patterns)
    risk_ac, risk_keyword_to_rtype = _build_risk_automaton(risk_patterns)
    log.info(f"Aho-Corasick: {len(set(keyword_to_cid))} companies, {len(keyword_to_cid)} keywords")
    log.info(f"Risk Aho-Corasick: {len(set(risk_keyword_to_rtype))} types, {len(risk_keyword_to_rtype)} keywords")

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

        chunk = _process_single_parquet(f, patterns, risk_patterns, ac_automaton, keyword_to_cid, risk_ac, risk_keyword_to_rtype, filter_no_entity)
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
