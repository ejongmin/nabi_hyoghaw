from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
from rapidfuzz import fuzz

def _clean(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9가-힣\s\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_aliases_for_company(row: pd.Series, use_company_id: bool = True, use_tickers: bool = True) -> List[str]:
    aliases = []
    cid = str(row.get("company_id", "")).strip()
    cname = str(row.get("canonical_name", "")).strip()
    tickers = str(row.get("tickers", "")).strip()

    def add(x):
        x = _clean(x)
        if x and len(x) >= 2:
            aliases.append(x)

    if cname and cname != "nan":
        add(cname)
        # remove common suffixes
        cname2 = re.sub(r"\b(co\.?|ltd\.?|inc\.?|corp\.?|company)\b", "", cname, flags=re.IGNORECASE)
        add(cname2)

    if use_company_id and cid and cid != "nan":
        add(cid)

    if use_tickers and tickers and tickers != "nan":
        for t in tickers.split(";"):
            add(t.strip())

    # unique preserve order
    seen = set()
    out = []
    for a in aliases:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out

@dataclass
class EntityLinker:
    alias_to_company: Dict[str, str]
    alias_list: List[str]
    min_score: int = 90
    max_entities: int = 3

    @staticmethod
    def from_universe(universe: pd.DataFrame, min_score: int = 90, max_entities: int = 3,
                      use_company_id: bool = True, use_tickers: bool = True) -> "EntityLinker":
        m: Dict[str,str] = {}
        for _, row in universe.iterrows():
            cid = str(row.get("company_id","")).strip()
            if not cid or cid == "nan":
                continue
            for a in build_aliases_for_company(row, use_company_id=use_company_id, use_tickers=use_tickers):
                # if collision, keep first (avoid flapping)
                m.setdefault(a, cid)
        alias_list = list(m.keys())
        return EntityLinker(alias_to_company=m, alias_list=alias_list, min_score=min_score, max_entities=max_entities)

    def link_title(self, title: str) -> List[Tuple[str,int,str]]:
        """Return list of (company_id, score, matched_alias) sorted by score."""
        t = _clean(title)
        if not t:
            return []
        hits = []

        # fast exact substring with word boundary where possible
        for alias in self.alias_list:
            # avoid extremely short aliases
            if len(alias) < 3:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", t):
                hits.append((self.alias_to_company[alias], 100, alias))

        # if no exact hits, do fuzzy match
        if not hits:
            for alias in self.alias_list:
                if len(alias) < 4:
                    continue
                score = fuzz.partial_ratio(t, alias)
                if score >= self.min_score:
                    hits.append((self.alias_to_company[alias], int(score), alias))

        # dedupe by company_id keep best score
        best: Dict[str, Tuple[int,str]] = {}
        for cid, score, alias in hits:
            if cid not in best or score > best[cid][0]:
                best[cid] = (score, alias)
        out = [(cid, sc, al) for cid,(sc,al) in best.items()]
        out.sort(key=lambda x: x[1], reverse=True)
        return out[: self.max_entities]
