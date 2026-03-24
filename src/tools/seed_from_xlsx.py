from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yaml

from src.nlp.entity_linking import EntityLinker
from src.tools.universe_utils import build_ticker_alias_map

@dataclass
class SeedImportResult:
    edges: pd.DataFrame
    rows_in: int
    rows_out: int
    mapped_src_rate: float
    mapped_dst_rate: float
    dropped_rows: int
    unmapped_examples: pd.DataFrame

def _norm_col(c: str) -> str:
    return str(c).strip().lower()

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        if _norm_col(cand) in cols:
            return cols[_norm_col(cand)]
    return None

def _normalize_rel(rel: str, rel_norm: Dict[str, List[str]], default: str) -> str:
    if rel is None:
        return default
    r = str(rel).strip().lower()
    if r == "" or r == "nan":
        return default
    # exact key
    for k in rel_norm.keys():
        if r == k.lower():
            return k
    # alias match
    for k, aliases in rel_norm.items():
        for a in aliases:
            if a.lower() in r:
                return k
    return default

def _map_entity(val: Any, alias_map: Dict[str,str], linker: EntityLinker, min_score: int) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    # direct ticker alias
    if s in alias_map:
        return alias_map[s]
    # try normalized ticker forms
    s2 = s.replace(" ", "")
    if s2 in alias_map:
        return alias_map[s2]
    # fallback: fuzzy via linker
    hits = linker.link_title(s)
    if not hits:
        return None
    cid, score, _alias = hits[0]
    return cid if score >= min_score else None

def seed_edges_from_excel(xlsx_path: Path,
                          universe: pd.DataFrame,
                          mapping_yaml: Path,
                          normalize_rules: Dict[str,str],
                          allowed_relations: List[str],
                          sheet: str | int | None = None,
                          min_fuzzy_score: int = 88) -> SeedImportResult:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"xlsx not found: {xlsx_path}")

    cfg = yaml.safe_load(mapping_yaml.read_text(encoding="utf-8"))
    cols_cfg = cfg.get("columns", {})
    defaults = cfg.get("defaults", {})
    rel_norm = cfg.get("relation_normalization", {})

    df = pd.read_excel(xlsx_path, sheet_name=sheet if sheet is not None else 0)
    rows_in = len(df)

    # infer columns
    src_col = _find_col(df, cols_cfg.get("src", []))
    dst_col = _find_col(df, cols_cfg.get("dst", []))
    rel_col = _find_col(df, cols_cfg.get("rel_type", []))
    conf_col = _find_col(df, cols_cfg.get("confidence", []))
    str_col  = _find_col(df, cols_cfg.get("strength", []))
    ev_col   = _find_col(df, cols_cfg.get("evidence", []))
    srcinfo_col = _find_col(df, cols_cfg.get("source", []))
    vf_col = _find_col(df, cols_cfg.get("valid_from", []))
    vt_col = _find_col(df, cols_cfg.get("valid_to", []))

    if src_col is None or dst_col is None:
        raise ValueError(f"필수 컬럼(src/dst)을 찾지 못했습니다. mapping_yaml={mapping_yaml} 를 수정하세요. 발견된 컬럼={list(df.columns)}")

    # build mapping helpers
    alias_map = build_ticker_alias_map(universe, normalize_rules=normalize_rules)
    linker = EntityLinker.from_universe(universe, min_score=min_fuzzy_score, max_entities=1)

    mapped_src = []
    mapped_dst = []
    rels = []
    confs = []
    strengths = []
    evidences = []
    sources = []
    valid_froms = []
    valid_tos = []

    for _, row in df.iterrows():
        src_raw = row.get(src_col)
        dst_raw = row.get(dst_col)
        src_id = _map_entity(src_raw, alias_map, linker, min_score=min_fuzzy_score)
        dst_id = _map_entity(dst_raw, alias_map, linker, min_score=min_fuzzy_score)
        mapped_src.append(src_id)
        mapped_dst.append(dst_id)

        rel_raw = row.get(rel_col) if rel_col else defaults.get("rel_type", "SUPPLIES")
        rel = _normalize_rel(rel_raw, rel_norm, defaults.get("rel_type","SUPPLIES"))
        if rel not in allowed_relations:
            # fallback to default rather than dropping (avoid empty output)
            rel = defaults.get("rel_type","SUPPLIES")
        rels.append(rel)

        conf = row.get(conf_col) if conf_col else defaults.get("confidence_plink", 0.6)
        strength = row.get(str_col) if str_col else defaults.get("strength", 1.0)
        try:
            conf = float(conf)
        except Exception:
            conf = float(defaults.get("confidence_plink", 0.6))
        try:
            strength = float(strength)
        except Exception:
            strength = float(defaults.get("strength", 1.0))
        confs.append(conf)
        strengths.append(strength)

        ev = row.get(ev_col) if ev_col else ""
        evidences.append("" if ev is None else str(ev))

        srcinfo = row.get(srcinfo_col) if srcinfo_col else defaults.get("source","xlsx_import")
        sources.append("" if srcinfo is None else str(srcinfo))

        vf = row.get(vf_col) if vf_col else None
        vt = row.get(vt_col) if vt_col else None
        valid_froms.append(vf)
        valid_tos.append(vt)

    out = pd.DataFrame({
        "src_company_id": mapped_src,
        "rel_type": rels,
        "dst_company_id": mapped_dst,
        "confidence_plink": confs,
        "strength": strengths,
        "evidence": evidences,
        "source": sources,
        "valid_from": valid_froms,
        "valid_to": valid_tos,
    })

    # drop unmapped rows
    before = len(out)
    out2 = out.dropna(subset=["src_company_id","dst_company_id"]).copy()
    dropped = before - len(out2)

    src_rate = 0.0 if before==0 else float(pd.Series(mapped_src).notna().mean())
    dst_rate = 0.0 if before==0 else float(pd.Series(mapped_dst).notna().mean())

    # examples
    unmapped = out[out["src_company_id"].isna() | out["dst_company_id"].isna()].head(10).copy()

    return SeedImportResult(
        edges=out2.reset_index(drop=True),
        rows_in=rows_in,
        rows_out=len(out2),
        mapped_src_rate=src_rate,
        mapped_dst_rate=dst_rate,
        dropped_rows=dropped,
        unmapped_examples=unmapped,
    )
