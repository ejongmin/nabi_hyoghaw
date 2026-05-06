"""
Aho-Corasick v2 재실행 — 메모리 효율 최적화 버전
=================================================
OOM 방지를 위해:
  1) 파일 1개씩 순차 처리
  2) 청크 결과를 메모리 누적 대신 **매 50배치마다 디스크 flush**
  3) 처리 완료된 파일은 checkpoint skip
  4) 각 파일 사이 gc.collect()

Usage:
  python scripts/rerun_aho_corasick_v2.py
"""
import gc
import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import yaml

# ── 프로젝트 root로 이동 ──
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("aho_v2")

# ── Config ──
BATCH_ROWS = 10_000  # OOM 방지를 위해 극소 배치
FLUSH_EVERY = 3      # 매 3 배치마다 디스크 flush (10K × 3 = 30K rows 단위)
PARQUET_DIR = ROOT / "data" / "raw" / "gdelt"
CHECKPOINT_DIR = PARQUET_DIR / "risk_events_checkpoint"
KEYWORDS_YAML = ROOT / "configs" / "risk_keywords.yaml"
ALIASES_YAML = ROOT / "configs" / "company_aliases.yaml"
UNIVERSE_CSV = ROOT / "data" / "universe" / "univers_final.csv"

FILES_TO_PROCESS = [
    # 2022는 이미 완료 (risk_v2_gdelt_gkg_2022.parquet 존재)
    PARQUET_DIR / "gdelt_gkg_2023.parquet",
    PARQUET_DIR / "gdelt_gkg_2024.parquet",
    PARQUET_DIR / "gdelt_gkg_2025.parquet",
]

NEEDED_COLS = [
    "gkg_id", "gkg_date", "source_domain", "article_url",
    "v2_organizations", "v2_themes", "v2_tone", "v2_locations",
    "collection_type",
]

OUT_COLS = [
    "event_id", "event_time", "url", "title", "risk_types",
    "severity", "entity_ids", "entity_scores", "finbert",
    "country_ids", "tone", "source_domain", "collection_type",
]

FIPS_MAP = {
    "CI": "Chile", "AS": "Australia", "ID": "Indonesia",
    "CG": "Congo", "CF": "Congo",
    "CH": "China", "KS": "South_Korea", "GM": "Germany",
    "JA": "Japan", "US": "USA", "UK": "UK", "CA": "Canada",
}

_COMMON_WORD_BLOCKLIST = {
    "gem", "ace", "can", "man", "ion", "air", "arc", "art",
    "bay", "bit", "cap", "cup", "era", "eve", "gap", "hub",
    "joy", "key", "lab", "map", "net", "ore", "pan", "ray",
}


# ══════════════════════════════════════════════════════════════
# 1. Pattern builders (gdelt_bq.py에서 가져옴)
# ══════════════════════════════════════════════════════════════

def _load_aliases() -> Dict[str, List[str]]:
    data = yaml.safe_load(ALIASES_YAML.read_text(encoding="utf-8"))
    return {str(k): [str(a) for a in v] for k, v in data.items() if isinstance(v, list)}


def _build_alias_patterns(universe: pd.DataFrame, extra_aliases: Dict[str, List[str]]):
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
        aliases = {a for a in aliases if len(a) >= 3 and a.lower() not in _COMMON_WORD_BLOCKLIST}
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
    return patterns, alias_to_cid


def _build_aho_automaton(patterns):
    import ahocorasick_rs
    all_keywords = []
    keyword_to_cid = []
    for pat, cid in patterns:
        kw = pat.pattern
        kw = kw.replace(r'\b', '').replace(r'\B', '')
        kw = re.sub(r'\\(.)', r'\1', kw)
        kw = kw.strip().upper()
        if kw:
            all_keywords.append(kw)
            keyword_to_cid.append(cid)
    log.info(f"Company AC: {len(all_keywords)} keywords")
    ac = ahocorasick_rs.AhoCorasick(all_keywords, matchkind=ahocorasick_rs.MATCHKIND_STANDARD)
    return ac, keyword_to_cid


def _expand_regex_to_literals(pattern_str: str) -> List[str]:
    s = pattern_str.replace(r'\b', '').replace(r'\B', '').strip()
    fragments = s.split('|')
    literals = []
    for frag in fragments:
        frag = frag.strip()
        if not frag:
            continue
        expanded = _expand_fragment(frag)
        literals.extend(expanded)
    seen = set()
    result = []
    for lit in literals:
        lit = re.sub(r'\s+', ' ', lit).strip().upper()
        if len(lit) >= 2 and lit not in seen:
            seen.add(lit)
            result.append(lit)
    return result


def _expand_fragment(frag: str) -> List[str]:
    results = [""]
    i = 0
    while i < len(frag):
        m = re.match(r'\[([a-zA-Z]+)\]\??', frag[i:])
        if m:
            chars = m.group(1)
            optional = '?' in m.group(0)
            new_results = []
            for r in results:
                for c in chars:
                    new_results.append(r + c)
                if optional:
                    new_results.append(r)
            results = new_results
            i += m.end()
            continue
        if frag[i] == '.' and i + 1 < len(frag) and frag[i + 1] in ('*', '+'):
            results = [r + ' ' for r in results]
            i += 2
            continue
        if frag[i] == '.' and i + 1 < len(frag) and frag[i + 1] == '?':
            new_results = []
            for r in results:
                new_results.append(r + ' ')
                new_results.append(r)
            results = new_results
            i += 2
            continue
        if frag[i] == '.' and (i + 1 >= len(frag) or frag[i + 1] not in ('*', '+', '?')):
            results = [r + ' ' for r in results]
            i += 1
            continue
        if frag[i] in ('(', ')', '^', '$'):
            i += 1
            continue
        results = [r + frag[i] for r in results]
        i += 1
    return [re.sub(r'\s+', ' ', r).strip() for r in results if r.strip()]


def _load_risk_patterns(keywords_yaml: Path):
    try:
        data = yaml.safe_load(keywords_yaml.read_text(encoding="utf-8"))
    except Exception:
        return [
            ("geopolitics", re.compile(r"sanction|embargo|tariff|trade war", re.IGNORECASE)),
            ("logistics", re.compile(r"port disruption|shipping delay|blockade", re.IGNORECASE)),
            ("natural", re.compile(r"earthquake|tsunami|flood|wildfire", re.IGNORECASE)),
        ]
    patterns = []
    for risk_type in data:
        section = data[risk_type]
        if not isinstance(section, dict):
            continue
        for kw in section.get("keywords", []):
            if isinstance(kw, dict) and "pattern" in kw:
                try:
                    patterns.append((risk_type, re.compile(kw["pattern"], re.IGNORECASE)))
                except re.error:
                    pass
            elif isinstance(kw, str):
                try:
                    patterns.append((risk_type, re.compile(re.escape(kw), re.IGNORECASE)))
                except re.error:
                    pass
    return patterns


def _build_risk_automaton(risk_patterns):
    import ahocorasick_rs
    keywords = []
    keyword_to_rtype = []
    for rtype, pat in risk_patterns:
        for lit in _expand_regex_to_literals(pat.pattern):
            keywords.append(lit)
            keyword_to_rtype.append(rtype)
    log.info(f"Risk AC: {len(keywords)} keywords from {len(risk_patterns)} patterns")
    ac = ahocorasick_rs.AhoCorasick(keywords, matchkind=ahocorasick_rs.MATCHKIND_STANDARD)
    return ac, keyword_to_rtype


# ══════════════════════════════════════════════════════════════
# 2. Vectorized helpers
# ══════════════════════════════════════════════════════════════

def _vec_company_match(series: pd.Series, ac, kw2cid) -> pd.Series:
    texts = series.fillna("").astype(str).str.upper().tolist()
    entity_ids = [[] for _ in range(len(texts))]
    for i, txt in enumerate(texts):
        if not txt:
            continue
        matches = ac.find_matches_as_indexes(txt)
        if matches:
            entity_ids[i] = sorted(set(kw2cid[idx] for idx, _s, _e in matches))
    return pd.Series(entity_ids, index=series.index)


def _vec_tone_severity(v2tone: pd.Series):
    tone = v2tone.fillna("0").astype(str).str.split(",").str[0]
    tone = pd.to_numeric(tone, errors="coerce").fillna(0.0)
    norm = np.clip((10 - tone.values) / 20.0, 0, 1)
    sev = np.clip(1.0 + norm * 4.0, 1.0, 5.0).round(3)
    return tone, pd.Series(sev, index=v2tone.index)


def _vec_risk_classify(v2themes: pd.Series, risk_ac, rtype_map) -> pd.Series:
    texts = v2themes.fillna("").astype(str).str.upper().tolist()
    results = ["other"] * len(texts)
    for i, txt in enumerate(texts):
        if not txt:
            continue
        matches = risk_ac.find_matches_as_indexes(txt)
        if matches:
            results[i] = ",".join(sorted(set(rtype_map[idx] for idx, _s, _e in matches)))
    return pd.Series(results, index=v2themes.index)


def _vec_countries(v2loc: pd.Series) -> pd.Series:
    texts = v2loc.fillna("").astype(str).tolist()
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
    return pd.Series(results, index=v2loc.index)


def _vec_event_ids(gkg_ids: pd.Series, urls: pd.Series) -> pd.Series:
    gids = gkg_ids.fillna("").astype(str).tolist()
    us = urls.fillna("").astype(str).tolist()
    return pd.Series(
        [hashlib.md5((g + "|" + u).encode()).hexdigest() for g, u in zip(gids, us)],
        index=gkg_ids.index,
    )


# ══════════════════════════════════════════════════════════════
# 3. Core: 단일 파일 처리 (디스크 flush 방식)
# ══════════════════════════════════════════════════════════════

def process_file(
    parquet_path: Path,
    ac_automaton,
    keyword_to_cid: List[str],
    risk_ac,
    risk_kw_to_rtype: List[str],
    output_path: Path,
    filter_no_entity: bool = True,
):
    """
    단일 parquet 처리 → output_path에 결과 저장.
    핵심: chunks를 메모리에 누적하지 않고 매 FLUSH_EVERY 배치마다 디스크 flush.
    """
    pf = pq.ParquetFile(parquet_path)
    n_total = pf.metadata.num_rows
    log.info(f"=== {parquet_path.name}: {n_total:,} rows ===")

    available = set(pf.schema.names)
    use_cols = [c for c in NEEDED_COLS if c in available]

    n_processed = 0
    n_kept = 0
    batch_count = 0
    pending_chunks: List[pd.DataFrame] = []
    writer = None

    try:
        for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=use_cols):
            n_processed += batch.num_rows

            # ── PyArrow pre-filter: null/빈 v2org 제거, Type B 보존 ──
            v2org_col = batch.column("v2_organizations")
            not_null = pc.is_valid(v2org_col)
            not_empty = pc.not_equal(
                pc.utf8_length(pc.if_else(not_null, v2org_col, pa.scalar(""))), 0
            )
            has_v2org = pc.and_(not_null, not_empty)

            if "collection_type" in available:
                ct_col = batch.column("collection_type")
                is_risk_only = pc.equal(
                    pc.if_else(pc.is_valid(ct_col), ct_col, pa.scalar("")),
                    pa.scalar("risk_only"),
                )
                mask = pc.or_(has_v2org, is_risk_only)
            else:
                mask = has_v2org
            batch = batch.filter(mask)

            if batch.num_rows == 0:
                continue

            n_kept += batch.num_rows
            df = batch.to_pandas()
            del batch

            # ── Aho-Corasick company matching ──
            df["entity_ids"] = _vec_company_match(df["v2_organizations"], ac_automaton, keyword_to_cid)

            # ── Type B fallback: URL matching ──
            no_match = df["entity_ids"].apply(len) == 0
            if no_match.any() and "article_url" in df.columns:
                url_matches = _vec_company_match(df.loc[no_match, "article_url"], ac_automaton, keyword_to_cid)
                df.loc[no_match, "entity_ids"] = url_matches

            # ── Filter: entity 없는 행 제거 (Type B 보존) ──
            if filter_no_entity:
                has_ent = df["entity_ids"].apply(len) > 0
                is_risk = df.get("collection_type", pd.Series(dtype=str)) == "risk_only"
                df = df[has_ent | is_risk].copy()

            if df.empty:
                continue

            # ── 후처리 ──
            df["tone"], df["severity"] = _vec_tone_severity(df["v2_tone"])
            df["risk_types"] = _vec_risk_classify(df["v2_themes"], risk_ac, risk_kw_to_rtype)
            df["country_ids"] = _vec_countries(df["v2_locations"]) if "v2_locations" in df.columns else ""
            df["event_time"] = pd.to_datetime(
                df["gkg_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce"
            )
            df["event_id"] = _vec_event_ids(df["gkg_id"], df["article_url"])
            df["entity_scores"] = df["entity_ids"].apply(
                lambda ids: [{"company_id": c, "score": 100, "alias": "bq_regex"} for c in ids]
            )
            df["title"] = ""
            df["finbert"] = None
            df["url"] = df["article_url"]

            out = df[[c for c in OUT_COLS if c in df.columns]].copy()
            pending_chunks.append(out)
            del df, out
            batch_count += 1

            # ── 주기적 디스크 flush (메모리 절약 핵심) ──
            if batch_count % FLUSH_EVERY == 0:
                _flush_chunks(pending_chunks, output_path, writer is None)
                if writer is None:
                    writer = True  # flag: 파일 이미 생성됨
                pending_chunks = []
                gc.collect()
                log.info(f"  [{parquet_path.stem}] {n_processed:,}/{n_total:,} "
                         f"kept={n_kept:,} (flushed to disk)")

        # ── 마지막 잔여 청크 flush ──
        if pending_chunks:
            _flush_chunks(pending_chunks, output_path, writer is None)
            del pending_chunks
            gc.collect()

    except Exception as e:
        log.error(f"Error processing {parquet_path.name}: {e}")
        raise

    log.info(f"  PyArrow: {n_total:,} → {n_kept:,} ({100*n_kept/max(n_total,1):.1f}%)")

    # ── 결과 행 수 확인 ──
    if output_path.exists():
        final = pq.read_metadata(output_path)
        log.info(f"  ✓ Saved: {output_path.name} ({final.num_rows:,} rows)")
        return final.num_rows
    else:
        log.warning(f"  No rows produced for {parquet_path.name}")
        return 0


def _flush_chunks(chunks: List[pd.DataFrame], output_path: Path, is_first: bool):
    """pending chunks → disk (append 모드)."""
    if not chunks:
        return
    combined = pd.concat(chunks, ignore_index=True)
    table = pa.Table.from_pandas(combined, preserve_index=False)
    del combined

    if is_first or not output_path.exists():
        pq.write_table(table, output_path)
    else:
        # append: 기존 파일 읽어서 concat 후 재기록
        # 메모리 부담이 되지만 parquet append가 PyArrow에 없어서 불가피
        # 대안: 별도 파트 파일로 저장 후 마지막에 merge
        part_path = output_path.with_suffix(f".part{output_path.stat().st_size}")
        pq.write_table(table, part_path)
    del table
    gc.collect()


# ══════════════════════════════════════════════════════════════
# 4. Main
# ══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Aho-Corasick v2 재실행 (OOM-safe)")
    log.info("=" * 60)

    # ── 패턴 빌드 (1회) ──
    universe = pd.read_csv(UNIVERSE_CSV)
    extra_aliases = _load_aliases()
    patterns, _ = _build_alias_patterns(universe, extra_aliases)
    risk_patterns = _load_risk_patterns(KEYWORDS_YAML)
    ac, kw2cid = _build_aho_automaton(patterns)
    risk_ac, risk_kw2rtype = _build_risk_automaton(risk_patterns)
    del universe, patterns, risk_patterns  # 패턴 객체만 유지

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for f in FILES_TO_PROCESS:
        if not f.exists():
            log.warning(f"File not found: {f}")
            continue

        cp_path = CHECKPOINT_DIR / f"risk_v2_{f.stem}.parquet"
        if cp_path.exists():
            existing_rows = pq.read_metadata(cp_path).num_rows
            log.info(f"[SKIP] {f.name} → checkpoint exists: {cp_path.name} ({existing_rows:,} rows)")
            total_rows += existing_rows
            continue

        # part 파일 방식: 각 flush는 별도 파일로 저장
        # 최종에 merge → 메모리 효율적
        parts_dir = CHECKPOINT_DIR / f"_parts_{f.stem}"
        parts_dir.mkdir(exist_ok=True)

        # 기존 part 파일 정리
        for old in parts_dir.glob("*.parquet"):
            old.unlink()

        n = _process_file_partitioned(f, ac, kw2cid, risk_ac, risk_kw2rtype, parts_dir)

        # part 파일 merge
        if n > 0:
            _merge_parts(parts_dir, cp_path)
            final_rows = pq.read_metadata(cp_path).num_rows
            log.info(f"  ✓ Merged: {cp_path.name} ({final_rows:,} rows)")
            total_rows += final_rows
        else:
            log.warning(f"  No rows from {f.name}")

        gc.collect()

    # 2022 checkpoint도 카운트
    ck_2022 = CHECKPOINT_DIR / "risk_v2_gdelt_gkg_2022.parquet"
    if ck_2022.exists():
        total_rows += pq.read_metadata(ck_2022).num_rows

    log.info(f"\n{'='*60}")
    log.info(f"All done! Total v2 rows: {total_rows:,}")
    log.info(f"Checkpoints in: {CHECKPOINT_DIR}")


def _process_file_partitioned(
    parquet_path: Path,
    ac_automaton,
    keyword_to_cid,
    risk_ac,
    risk_kw_to_rtype,
    parts_dir: Path,
    filter_no_entity: bool = True,
) -> int:
    """파일 처리 → parts_dir에 part 파일로 분할 저장."""
    pf = pq.ParquetFile(parquet_path)
    n_total = pf.metadata.num_rows
    log.info(f"=== {parquet_path.name}: {n_total:,} rows ===")

    available = set(pf.schema.names)
    use_cols = [c for c in NEEDED_COLS if c in available]

    n_processed = 0
    n_kept = 0
    batch_count = 0
    part_idx = 0
    pending: List[pd.DataFrame] = []
    total_output_rows = 0

    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=use_cols):
        n_processed += batch.num_rows

        # PyArrow pre-filter
        v2org_col = batch.column("v2_organizations")
        not_null = pc.is_valid(v2org_col)
        not_empty = pc.not_equal(
            pc.utf8_length(pc.if_else(not_null, v2org_col, pa.scalar(""))), 0
        )
        has_v2org = pc.and_(not_null, not_empty)

        if "collection_type" in available:
            ct_col = batch.column("collection_type")
            is_risk_only = pc.equal(
                pc.if_else(pc.is_valid(ct_col), ct_col, pa.scalar("")),
                pa.scalar("risk_only"),
            )
            mask = pc.or_(has_v2org, is_risk_only)
        else:
            mask = has_v2org
        batch = batch.filter(mask)

        if batch.num_rows == 0:
            continue

        n_kept += batch.num_rows
        df = batch.to_pandas()
        del batch

        # Aho-Corasick company matching
        df["entity_ids"] = _vec_company_match(df["v2_organizations"], ac_automaton, keyword_to_cid)

        # Type B fallback: URL matching
        no_match = df["entity_ids"].apply(len) == 0
        if no_match.any() and "article_url" in df.columns:
            url_matches = _vec_company_match(df.loc[no_match, "article_url"], ac_automaton, keyword_to_cid)
            df.loc[no_match, "entity_ids"] = url_matches

        # Filter
        if filter_no_entity:
            has_ent = df["entity_ids"].apply(len) > 0
            is_risk = df.get("collection_type", pd.Series(dtype=str)) == "risk_only"
            df = df[has_ent | is_risk].copy()

        if df.empty:
            del df
            continue

        # 후처리 — 원본 컬럼 즉시 삭제로 메모리 절약
        df["tone"], df["severity"] = _vec_tone_severity(df["v2_tone"])
        df.drop(columns=["v2_tone"], errors="ignore", inplace=True)
        df["risk_types"] = _vec_risk_classify(df["v2_themes"], risk_ac, risk_kw_to_rtype)
        df.drop(columns=["v2_themes"], errors="ignore", inplace=True)
        df["country_ids"] = _vec_countries(df["v2_locations"]) if "v2_locations" in df.columns else ""
        df.drop(columns=["v2_locations"], errors="ignore", inplace=True)
        df["event_time"] = pd.to_datetime(
            df["gkg_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce"
        )
        df["event_id"] = _vec_event_ids(df["gkg_id"], df["article_url"])
        df.drop(columns=["gkg_id", "gkg_date", "v2_organizations"], errors="ignore", inplace=True)
        df["entity_scores"] = df["entity_ids"].apply(
            lambda ids: [{"company_id": c, "score": 100, "alias": "bq_regex"} for c in ids]
        )
        df["title"] = ""
        df["finbert"] = None
        df["url"] = df["article_url"]
        df.drop(columns=["article_url"], errors="ignore", inplace=True)

        out = df[[c for c in OUT_COLS if c in df.columns]].copy()
        pending.append(out)
        total_output_rows += len(out)
        del df, out
        gc.collect()
        batch_count += 1

        # 주기적 flush → part 파일
        if batch_count % FLUSH_EVERY == 0:
            part_path = parts_dir / f"part_{part_idx:04d}.parquet"
            combined = pd.concat(pending, ignore_index=True)
            combined.to_parquet(part_path, index=False)
            log.info(f"  [{parquet_path.stem}] {n_processed:,}/{n_total:,} "
                     f"kept={n_kept:,} output={total_output_rows:,} "
                     f"→ {part_path.name} ({len(combined):,} rows)")
            del combined
            pending = []
            part_idx += 1
            gc.collect()

    # 잔여 flush
    if pending:
        part_path = parts_dir / f"part_{part_idx:04d}.parquet"
        combined = pd.concat(pending, ignore_index=True)
        combined.to_parquet(part_path, index=False)
        log.info(f"  [{parquet_path.stem}] final flush → {part_path.name} ({len(combined):,} rows)")
        del combined, pending
        gc.collect()

    log.info(f"  PyArrow: {n_total:,} → {n_kept:,} ({100*n_kept/max(n_total,1):.1f}%)")
    log.info(f"  Output rows: {total_output_rows:,}")
    return total_output_rows


def _merge_parts(parts_dir: Path, output_path: Path):
    """part 파일들 → 하나의 parquet으로 merge (스트리밍)."""
    parts = sorted(parts_dir.glob("part_*.parquet"))
    if not parts:
        return

    # ParquetWriter로 스트리밍 merge (메모리 효율적)
    writer = None
    for p in parts:
        table = pq.read_table(p)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        del table
        gc.collect()

    if writer:
        writer.close()

    # part 파일 정리
    for p in parts:
        p.unlink()
    try:
        parts_dir.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
