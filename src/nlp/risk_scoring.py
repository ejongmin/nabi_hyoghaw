from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import pandas as pd
import numpy as np
import yaml

from src.nlp.entity_linking import EntityLinker
from src.nlp.finbert import score_texts

@dataclass
class RiskKeywordModel:
    keyword_map: Dict[str, List[str]]

    @staticmethod
    def load(path: Path) -> "RiskKeywordModel":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        km = {k: v.get("keywords", []) for k,v in data.items()}
        return RiskKeywordModel(keyword_map=km)

    def classify(self, text: str) -> List[str]:
        t = (text or "").lower()
        labels: List[str] = []
        for rtype, kws in self.keyword_map.items():
            for kw in kws:
                if kw.lower() in t:
                    labels.append(rtype)
                    break
        return labels or ["other"]

def _event_id(url: str, seendate: str) -> str:
    raw = (str(url) + "|" + str(seendate)).encode("utf-8")
    return hashlib.md5(raw).hexdigest()

def build_risk_events(articles: pd.DataFrame,
                      universe: pd.DataFrame,
                      keywords_yaml: Path,
                      cfg_risk: Dict[str, Any],
                      cfg_link: Dict[str, Any]) -> pd.DataFrame:
    """Articles -> risk_events with (risk_types, severity, entity_ids)."""
    if articles is None or len(articles) == 0:
        return pd.DataFrame(columns=["event_id","event_time","url","title","risk_types","severity","entity_ids","entity_scores","finbert"])

    # Standardize columns we use
    df = articles.copy()
    if "seendate" not in df.columns:
        # try alternative names
        for alt in ["published", "datetime", "date"]:
            if alt in df.columns:
                df["seendate"] = df[alt]
                break
    if "title" not in df.columns:
        # minimal fallback
        df["title"] = ""
    if "url" not in df.columns:
        df["url"] = ""

    df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce")
    df = df.dropna(subset=["seendate"])
    df["title"] = df["title"].fillna("").astype(str)
    df["url"] = df["url"].fillna("").astype(str)

    km = RiskKeywordModel.load(keywords_yaml)
    labels = [km.classify(t) for t in df["title"].tolist()]

    use_finbert = bool(cfg_risk.get("use_finbert", False))
    finbert_model = str(cfg_risk.get("finbert_model", "ProsusAI/finbert"))

    fin_out = None
    fin_score = np.zeros(len(df), dtype=float)
    if use_finbert:
        # may raise if deps missing
        fin_res = score_texts(df["title"].tolist(), model_name=finbert_model)
        fin_out = [r.__dict__ for r in fin_res]
        fin_score = np.array([r.negative - r.positive for r in fin_res], dtype=float)

    # keyword counts
    kw_counts = []
    for title, lbs in zip(df["title"].tolist(), labels):
        c = 0
        tl = title.lower()
        for lb in lbs:
            if lb == "other":
                continue
            for kw in km.keyword_map.get(lb, []):
                if kw.lower() in tl:
                    c += 1
        kw_counts.append(c)

    sev_cfg = cfg_risk.get("severity", {})
    base = float(sev_cfg.get("base", 1.0))
    kw_w = float(sev_cfg.get("keyword_weight", 0.4))
    fin_w = float(sev_cfg.get("finbert_weight", 0.2))
    cap  = float(sev_cfg.get("max_cap", 5.0))

    severity = base + kw_w*np.array(kw_counts, dtype=float) + fin_w*np.clip(fin_score, 0, 1.5)
    severity = np.clip(severity, 0, cap)

    # entity linking
    linker = EntityLinker.from_universe(
        universe=universe,
        min_score=int(cfg_link.get("min_fuzzy_score", 90)),
        max_entities=int(cfg_link.get("max_entities_per_article", 3)),
        use_company_id=bool(cfg_link.get("use_company_id_as_alias", True)),
        use_tickers=bool(cfg_link.get("use_tickers_as_aliases", True)),
    )
    linked = [linker.link_title(t) for t in df["title"].tolist()]
    entity_ids = [[cid for cid,score,alias in items] for items in linked]
    entity_scores = [[{"company_id":cid,"score":score,"alias":alias} for cid,score,alias in items] for items in linked]

    out = pd.DataFrame({
        "event_id": [ _event_id(u, str(d)) for u,d in zip(df["url"].tolist(), df["seendate"].astype(str).tolist()) ],
        "event_time": df["seendate"],
        "url": df["url"],
        "title": df["title"],
        "risk_types": [",".join(l) for l in labels],
        "severity": severity,
        "entity_ids": entity_ids,
        "entity_scores": entity_scores,
        "finbert": fin_out,
    })

    if bool(cfg_risk.get("filter_no_entity_events", True)):
        out = out[out["entity_ids"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()

    return out.reset_index(drop=True)
