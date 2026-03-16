"""나비효과 파이프라인 — GDELT v2 Raw File 과거 리플레이

2023~2025년의 GDELT v2 Raw Export 파일을 순차적으로 다운로드하면서
실시간 파이프라인을 시뮬레이션합니다. 마치 2023년에 실시간 수집기를
운영하는 것처럼 동작합니다.

사용법:
    python live_replay.py                              # 기본 (daily, 5초 간격)
    python live_replay.py --start 2023-06-01           # 시작일 지정
    python live_replay.py --interval weekly --delay 3  # 주간, 3초 간격
    python live_replay.py --max-per-file 5             # 파일당 최대 5개
    python live_replay.py --no-cache                   # 캐시 사용 안함
"""
import sys
import os
import io
import time
import argparse
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import yaml
import requests
from rapidfuzz import fuzz

# ── 경로 설정 ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.entity_link import _build_lookup, link_entities
from src.classify import classify_risk

# ── stdout UTF-8 (Windows 호환) ──
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── GDELT v2 설정 ──
GDELT_BASE = "http://data.gdeltproject.org/gdeltv2"
EVENTS_COLUMNS = {
    0: "GLOBALEVENTID", 1: "SQLDATE", 6: "Actor1Name",
    16: "Actor2Name", 26: "EventCode", 30: "GoldsteinScale",
    32: "NumArticles", 34: "AvgTone", 60: "SOURCEURL",
}

# ── 파이프라인 상수 ──
MIN_TERM_LEN = 3
SUPPLY_RELS = {"SUPPLIES", "BUYS_FROM", "PARTNERS_WITH", "OWNS", "COMPETES_WITH"}
CAR_WINDOWS = [1, 3, 5]
EST_WINDOW = (-120, -20)

ROUTE_NODES = {
    "ROUTE_SUEZ": {
        "keywords": ["suez canal", "suez"],
        "regions": ["China", "Germany", "United Kingdom"],
        "strength": 0.7,
    },
    "ROUTE_HORMUZ": {
        "keywords": ["strait of hormuz", "hormuz"],
        "regions": ["China", "Japan", "South Korea", "Germany"],
        "strength": 0.6,
    },
    "ROUTE_MALACCA": {
        "keywords": ["malacca strait", "malacca"],
        "regions": ["China", "Japan", "South Korea"],
        "strength": 0.6,
    },
}

EXTRA_RISK_KEYWORDS = {
    "geopolitical": ["tariff", "sanction", "conflict", "invasion", "missile",
                     "nuclear", "war"],
    "logistics": ["shipping", "freight", "port", "blockade", "canal", "strait"],
    "natural": ["earthquake", "flood", "typhoon", "hurricane", "tsunami",
                "wildfire", "drought"],
    "energy": ["oil", "crude", "opec", "refinery", "pipeline", "fuel"],
}

# ── ANSI 컬러 ──
try:
    os.system("")  # Windows ANSI 활성화
except Exception:
    pass

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"


def p(text=""):
    print(text, flush=True)


def _sev_color(s):
    return RED if s >= 4 else (YELLOW if s >= 3 else WHITE)


def _rate_color(r):
    return GREEN if r >= 55 else (YELLOW if r >= 45 else RED)


def _exp_bar(v, w=15):
    f = int(v * w)
    bar = "█" * f + "░" * (w - f)
    return f"{RED}{bar}{RESET}" if v >= 0.8 else (
        f"{YELLOW}{bar}{RESET}" if v >= 0.4 else f"{DIM}{bar}{RESET}")


# ═══════════════════════════════════════════════════════════════════
# 1. STATIC DATA (1회 로드)
# ═══════════════════════════════════════════════════════════════════
def load_static():
    """유니버스, KG, 주가, 키워드 등 정적 데이터 로드."""
    p(f"\n{BOLD}{'=' * 70}{RESET}")
    p(f"{BOLD}  LOADING STATIC DATA...{RESET}")
    p(f"{'=' * 70}")

    # Universe
    uni = pd.read_csv(
        PROJECT_ROOT / "data/universe/univers_final.csv", encoding="utf-8-sig")
    id2name = dict(zip(uni["company_id"], uni["canonical_name"]))

    # Entity lookup (짧은 검색어 제거)
    lookup = _build_lookup(uni)
    for e in lookup:
        e["terms"] = {t for t in e["terms"] if len(t) >= MIN_TERM_LEN}
    lookup = [e for e in lookup if e["terms"]]

    # Risk keywords
    with open(PROJECT_ROOT / "configs/risk_keywords.yaml", "r",
              encoding="utf-8") as f:
        kw_map = yaml.safe_load(f)

    # --- KG 구축 ---
    G = nx.Graph()
    for _, r in uni.iterrows():
        G.add_node(r["company_id"], name=r["canonical_name"],
                   country=r.get("country", ""), ntype="company")

    edges = pd.read_parquet(PROJECT_ROOT / "data/processed/kg_edges.parquet")
    for _, r in edges[edges["rel_type"].isin(SUPPLY_RELS)].iterrows():
        s, d = r["src_company_id"], r["dst_company_id"]
        if s in G and d in G:
            w = float(r.get("strength", 0.5)) * float(
                r.get("confidence_plink", 1.0))
            G.add_edge(s, d, weight=w, rel_type=r["rel_type"])

    # Route nodes
    c2c = defaultdict(list)
    for _, r in uni.iterrows():
        c2c[r.get("country", "")].append(r["company_id"])
    for rid, info in ROUTE_NODES.items():
        G.add_node(rid, name=rid, ntype="route")
        for reg in info["regions"]:
            for cid in c2c.get(reg, []):
                G.add_edge(rid, cid, weight=info["strength"],
                           rel_type="SHIPS_VIA")

    initial_edges = G.number_of_edges()

    # --- 주가 수익률 캐시 ---
    pr = pd.read_parquet(PROJECT_ROOT / "data/processed/prices_daily.parquet")
    pr["date"] = pd.to_datetime(pr["date"])
    pr = pr.sort_values(["company_id", "date"])
    pr["ret"] = pr.groupby("company_id")["adj_close"].pct_change()
    ret_cache = {}
    for cid in pr["company_id"].unique():
        cd = pr[pr["company_id"] == cid][["date", "ret"]].dropna()
        ret_cache[cid] = cd.set_index("date")["ret"]

    company_ids = [n for n, d in G.nodes(data=True)
                   if d.get("ntype") == "company"]
    route_kw = {}
    for rid, info in ROUTE_NODES.items():
        for kw in info["keywords"]:
            route_kw[kw] = rid

    p(f"  Universe: {len(uni)} companies")
    p(f"  KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    p(f"  Prices: {len(pr):,} rows ({len(ret_cache)} companies)")
    p(f"  Risk categories: {list(kw_map.keys())}")

    return {
        "lookup": lookup, "kw_map": kw_map, "G": G, "id2name": id2name,
        "ret_cache": ret_cache, "company_ids": company_ids,
        "route_kw": route_kw, "initial_edges": initial_edges,
    }


# ═══════════════════════════════════════════════════════════════════
# 2. GDELT DOWNLOAD & PARSE
# ═══════════════════════════════════════════════════════════════════
def gen_timestamps(start, end, interval):
    """리플레이할 GDELT 타임스탬프 목록 생성."""
    ts_list = []
    cur = start
    if interval == "daily":
        while cur <= end:
            ts_list.append(cur.strftime("%Y%m%d") + "120000")
            cur += timedelta(days=1)
    elif interval == "weekly":
        while cur <= end:
            ts_list.append(cur.strftime("%Y%m%d") + "120000")
            cur += timedelta(weeks=1)
    elif interval == "hourly":
        while cur <= end:
            ts_list.append(cur.strftime("%Y%m%d%H") + "0000")
            cur += timedelta(hours=1)
    return ts_list


def fetch_gdelt(ts, cache_dir=None):
    """GDELT v2 export 파일 1개를 다운로드하고 DataFrame으로 파싱.

    cache_dir이 지정되면 로컬 캐시를 먼저 확인하고,
    없으면 다운로드 후 캐시에 저장합니다.
    """
    # 캐시 확인
    if cache_dir:
        cached = cache_dir / f"{ts}.export.CSV.zip"
        if cached.exists():
            try:
                return _parse_zip(cached.read_bytes()), None
            except Exception:
                pass  # 손상된 캐시 → 재다운로드

    url = f"{GDELT_BASE}/{ts}.export.CSV.zip"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return None, "404"
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return None, str(e)[:80]

    # 캐시 저장
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{ts}.export.CSV.zip").write_bytes(resp.content)

    try:
        return _parse_zip(resp.content), None
    except Exception as e:
        return None, str(e)[:80]


def _parse_zip(content):
    """ZIP 바이트를 DataFrame으로 파싱."""
    raw = content if isinstance(content, bytes) else bytes(content)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV")][0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(
                f, sep="\t", header=None, dtype=str, on_bad_lines="skip")

    keep = {i: n for i, n in EVENTS_COLUMNS.items() if i < len(df.columns)}
    df = df[list(keep.keys())].copy()
    df.columns = list(keep.values())
    df["GoldsteinScale"] = pd.to_numeric(
        df["GoldsteinScale"], errors="coerce").fillna(0)
    df["AvgTone"] = pd.to_numeric(
        df["AvgTone"], errors="coerce").fillna(0)
    df["NumArticles"] = pd.to_numeric(
        df["NumArticles"], errors="coerce").fillna(1).astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════
# 3. EVENT PROCESSING (파일 1개 → 매칭 이벤트)
# ═══════════════════════════════════════════════════════════════════
def process_file(df, ctx):
    """한 GDELT 파일에서 매칭 이벤트를 추출, 처리.

    Returns:
        (matched_events: list[dict], category_counts: dict)
    """
    lookup = ctx["lookup"]
    kw_map = ctx["kw_map"]
    G = ctx["G"]
    ret_cache = ctx["ret_cache"]
    company_ids = ctx["company_ids"]
    route_kw = ctx["route_kw"]

    # 빠른 사전 필터: Actor1/2에 유니버스 검색어가 포함된 행만 남기기
    all_terms = set()
    for entry in lookup:
        all_terms.update(entry["terms"])
    # 소문자 비교용
    terms_lower = {t.lower() for t in all_terms}

    def _has_term(val):
        if pd.isna(val):
            return False
        v = str(val).lower()
        return any(t in v for t in terms_lower)

    # risk keyword도 포함해서 사전필터
    all_risk_kw = set()
    for cfg in kw_map.values():
        all_risk_kw.update(kw.lower() for kw in cfg["keywords"])
    for kws in EXTRA_RISK_KEYWORDS.values():
        all_risk_kw.update(kws)

    def _has_risk_kw(row):
        parts = []
        for col in ["SOURCEURL", "Actor1Name", "Actor2Name"]:
            v = row.get(col)
            if pd.notna(v):
                parts.append(str(v).lower())
        text = " ".join(parts)
        return any(kw in text for kw in all_risk_kw)

    # 사전 필터: 기업 매칭 가능성이 있는 행
    mask = df["Actor1Name"].apply(_has_term) | df["Actor2Name"].apply(_has_term)
    df_filtered = df[mask].copy()

    results = []
    cat_counts = {"company": 0, "both": 0}

    for _, row in df_filtered.iterrows():
        # --- Entity linking ---
        ids, mentions, details = [], [], []
        for col in ["Actor1Name", "Actor2Name"]:
            a = row.get(col)
            if pd.notna(a) and str(a).strip():
                a_str = str(a).strip()
                ei, em = link_entities(a_str, lookup, 90, 3)
                for eid, mn in zip(ei, em):
                    if eid not in ids:
                        ids.append(eid)
                        mentions.append(mn)
                        sc = fuzz.partial_ratio(mn, a_str.lower())
                        details.append((a_str, eid, sc))

        if not ids:
            continue

        # --- Risk classification ---
        parts = [str(row.get(c, "")) for c in
                 ["SOURCEURL", "Actor1Name", "Actor2Name"] if pd.notna(row.get(c))]
        text = " ".join(parts)
        text_lower = text.lower()

        risk_t = classify_risk(text, kw_map)
        if risk_t == ["other"]:
            ext = []
            for rt, kws in EXTRA_RISK_KEYWORDS.items():
                if any(kw in text_lower for kw in kws):
                    ext.append(rt)
            risk_t = ext or ["other"]

        has_risk = risk_t != ["other"]
        cat = "both" if has_risk else "company"
        cat_counts[cat] += 1

        # --- Severity ---
        gs = float(row["GoldsteinScale"])
        at = float(row["AvgTone"])
        sev = float(np.clip(
            1 + np.clip((10 - gs) / 20, 0, 1) * 2.5
              + np.clip((10 - at) / 20, 0, 1) * 1.5, 1, 5))

        # --- Matched keywords ---
        mkw = []
        for rt, cfg in kw_map.items():
            for kw in cfg["keywords"]:
                if kw.lower() in text_lower:
                    mkw.append(kw)
        for rt, kws in EXTRA_RISK_KEYWORDS.items():
            for kw in kws:
                if kw in text_lower and kw not in mkw:
                    mkw.append(kw)

        # --- Source nodes ---
        src_nodes = list(ids)
        for kw, rid in route_kw.items():
            if kw in text_lower and rid not in src_nodes:
                src_nodes.append(rid)

        # --- Edge updates (2+ companies co-occur) ---
        edge_upd = []
        if len(ids) >= 2:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    s, d = ids[i], ids[j]
                    if s in G and d in G:
                        if G.has_edge(s, d):
                            ow = G[s][d].get("weight", 0.5)
                            nw = min(ow + 0.05, 1.0)
                            G[s][d]["weight"] = nw
                            edge_upd.append((s, d, ow, nw, "STRENGTHENED"))
                        else:
                            G.add_edge(s, d, weight=0.3,
                                       rel_type="PARTNERS_WITH")
                            edge_upd.append((s, d, 0, 0.3, "NEW"))

        # --- Exposure (top 5) ---
        exps = []
        for cid in company_ids:
            if cid in src_nodes:
                exps.append((cid, 1.0, [cid]))
            else:
                md, bp = float("inf"), None
                for sn in src_nodes:
                    try:
                        path = nx.shortest_path(G, sn, cid)
                        if len(path) - 1 < md:
                            md = len(path) - 1
                            bp = path
                    except nx.NetworkXNoPath:
                        pass
                if md < float("inf"):
                    exps.append((cid, 1 / (1 + md), bp))
        exps.sort(key=lambda x: -x[1])

        # --- CAR (상위 노출 기업) ---
        ev_date = pd.to_datetime(
            row["SQLDATE"], format="%Y%m%d", errors="coerce")
        if pd.isna(ev_date):
            continue

        car_res = []
        for cid, exp_val, path in exps[:5]:
            if cid not in ret_cache or ret_cache[cid].empty:
                continue
            cr = ret_cache[cid]
            fut = cr.index[cr.index >= ev_date].sort_values()
            if len(fut) < 2:
                continue
            est_s = ev_date + timedelta(days=EST_WINDOW[0])
            est_e = ev_date + timedelta(days=EST_WINDOW[1])
            er = cr[(cr.index >= est_s) & (cr.index <= est_e)]
            exp_r = er.mean() if len(er) >= 20 else cr.mean()

            cars = {}
            for w in CAR_WINDOWS:
                wd = fut[:w + 1]
                wr = cr.reindex(wd).dropna().values
                cars[w] = float(np.sum(wr - exp_r)) if len(wr) > 0 else None

            is_neg = gs < 0 or has_risk
            c3 = cars.get(3)
            succ = None
            if c3 is not None:
                succ = (c3 < 0) if is_neg else (c3 >= 0)

            car_res.append({
                "cid": cid, "exp": exp_val, "path": path,
                "next_td": fut[0], "cars": cars,
                "pred_neg": is_neg, "success": succ,
            })

        if not car_res:
            continue

        results.append({
            "date": ev_date,
            "a1": str(row.get("Actor1Name", "")),
            "a2": str(row.get("Actor2Name", "")),
            "gs": gs, "at": at,
            "url": str(row.get("SOURCEURL", "")),
            "risk_t": risk_t, "sev": round(sev, 3), "mkw": mkw,
            "details": details, "cat": cat,
            "src_nodes": src_nodes, "edge_upd": edge_upd,
            "top_exp": exps[:5], "car": car_res,
        })

    # 심각도 순 정렬
    results.sort(key=lambda x: -x["sev"])
    return results, cat_counts


# ═══════════════════════════════════════════════════════════════════
# 4. DISPLAY
# ═══════════════════════════════════════════════════════════════════
def show_event(ev, idx, total, id2name, sc):
    """개별 이벤트 표시 및 스코어보드 업데이트."""
    cat_c = YELLOW if ev["cat"] == "both" else WHITE
    p(f"\n  {BOLD}-- [{idx+1}/{total}] {cat_c}{ev['cat'].upper()}{RESET}"
      f" {'─' * 40}{RESET}")

    a1 = ev["a1"] if ev["a1"] != "nan" else "-"
    a2 = ev["a2"] if ev["a2"] != "nan" else "-"
    url = ev["url"]
    hl = url.split("/")[-1][:55].replace(".html", "").replace("-", " ") \
        if "/" in url else url[:55]
    if not hl:
        hl = "(no headline)"
    rt = ", ".join(ev["risk_t"]) if ev["risk_t"] != ["other"] else "-"
    kw = ", ".join(ev["mkw"][:4]) if ev["mkw"] else "-"

    p(f"  {BOLD}[*] NEWS DETECTED{RESET}")
    p(f"      Actor1: {WHITE}{a1}{RESET}  |  Actor2: {WHITE}{a2}{RESET}")
    p(f"      Risk: {YELLOW}{rt}{RESET}  |  "
      f"Severity: {_sev_color(ev['sev'])}{ev['sev']:.2f}{RESET}")
    p(f"      Goldstein: {ev['gs']:+.1f}  |  Tone: {ev['at']:+.1f}"
      f"  |  Keywords: {kw}")
    if hl != "(no headline)":
        p(f"      {DIM}{hl}{RESET}")

    # Entity link
    p(f"  {BOLD}[>>] ENTITY LINK{RESET}")
    for aname, eid, score in ev["details"]:
        nm = id2name.get(eid, "?")
        p(f'      "{aname}" -> {GREEN}{eid}{RESET} ({nm})  [score: {score:.0f}]')

    # Edge updates
    if ev["edge_upd"]:
        p(f"  {BOLD}[>>] EDGE UPDATE{RESET}")
        for s, d, ow, nw, label in ev["edge_upd"]:
            sn = id2name.get(s, s)
            dn = id2name.get(d, d)
            lc = GREEN if label == "NEW" else YELLOW
            p(f"      {sn} <-> {dn}: {ow:.2f} -> {nw:.2f} ({lc}{label}{RESET})")

    # Propagation
    p(f"  {BOLD}[>>] RISK PROPAGATION{RESET}")
    routes = [s for s in ev["src_nodes"] if s.startswith("ROUTE_")]
    if routes:
        p(f"      Route: {CYAN}{', '.join(routes)}{RESET}")
    for cid, ev_val, path in ev["top_exp"][:5]:
        nm = id2name.get(cid, cid)
        ps = " -> ".join(path) if path and len(path) > 1 else "direct"
        p(f"      {nm:>25}: {_exp_bar(ev_val)} {ev_val:.3f}"
          f"  ({DIM}{ps}{RESET})")

    # Prediction vs Reality
    p(f"  {BOLD}[>>] PREDICTION vs REALITY{RESET}")
    for cr in ev["car"]:
        nm = id2name.get(cr["cid"], cr["cid"])
        c3 = cr["cars"].get(3)
        pd_dir = "DOWN" if cr["pred_neg"] else "UP"
        pd_c = RED if cr["pred_neg"] else GREEN
        c3s = f"{c3:+.4f}" if c3 is not None else " N/A "
        c3c = RED if (c3 is not None and c3 < 0) else GREEN

        if cr["success"] is True:
            rs = f"{GREEN}[OK] SUCCESS{RESET}"
            sc["ok"] += 1
        elif cr["success"] is False:
            rs = f"{RED}[FAIL] MISS{RESET}"
            sc["fail"] += 1
        else:
            rs = f"{DIM}[--] NO DATA{RESET}"

        ntd = (cr["next_td"].strftime("%Y-%m-%d")
               if hasattr(cr["next_td"], "strftime") else str(cr["next_td"]))
        p(f"      {nm:>25} (exp={cr['exp']:.3f}):")
        p(f"        Predict: {pd_c}{pd_dir}{RESET}  |  "
          f"CAR[0,3]: {c3c}{c3s}{RESET}  |  {ntd}  |  {rs}")


# ═══════════════════════════════════════════════════════════════════
# 5. MAIN REPLAY LOOP
# ═══════════════════════════════════════════════════════════════════
def replay(args):
    ctx = load_static()
    id2name = ctx["id2name"]
    initial_edges = ctx["initial_edges"]

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    timestamps = gen_timestamps(start, end, args.interval)

    cache_dir = None
    if not args.no_cache:
        cache_dir = PROJECT_ROOT / "data/raw/gdelt/historical"

    sc = {"ok": 0, "fail": 0, "files_ok": 0, "files_match": 0,
          "total_match": 0, "edge_new": 0, "edge_str": 0}

    p(f"\n{BOLD}{'=' * 70}{RESET}")
    p(f"{BOLD}  GDELT RAW FILE REPLAY{RESET}")
    p(f"{BOLD}  Period: {args.start} ~ {args.end}  |  "
      f"Interval: {args.interval}  |  {len(timestamps)} files{RESET}")
    p(f"{BOLD}  Delay: {args.delay}s  |  Max/file: {args.max_per_file}  |  "
      f"Ctrl+C to stop{RESET}")
    p(f"{'=' * 70}\n")

    seen_geids = set()
    files_processed = 0

    try:
        for i, ts in enumerate(timestamps):
            # 시뮬레이션 날짜 표시
            dt_str = (f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} "
                      f"{ts[8:10]}:{ts[10:12]} UTC")

            p(f"{BOLD}{BLUE}{'=' * 70}{RESET}")
            p(f"{BOLD}  [{dt_str}]  GDELT FETCH #{i+1}/{len(timestamps)}"
              f"{RESET}")
            p(f"{BLUE}{'=' * 70}{RESET}")

            # --- 다운로드 ---
            df, err = fetch_gdelt(ts, cache_dir)
            if err:
                p(f"  {DIM}Skip: {err}{RESET}")
                time.sleep(1)
                continue

            sc["files_ok"] += 1
            files_processed = i + 1

            # --- GLOBALEVENTID 중복 제거 ---
            if "GLOBALEVENTID" in df.columns:
                before = len(df)
                df = df[~df["GLOBALEVENTID"].isin(seen_geids)]
                seen_geids.update(df["GLOBALEVENTID"].dropna().unique())
                deduped = before - len(df)
            else:
                deduped = 0

            p(f"  Downloaded: {len(df) + deduped:,} events"
              + (f"  (dedup: -{deduped})" if deduped > 0 else ""))

            if df.empty:
                p(f"  {DIM}No new events{RESET}")
                time.sleep(1)
                continue

            # --- 이벤트 처리 ---
            matched, cats = process_file(df, ctx)

            if not matched:
                p(f"  {DIM}No matching events{RESET}")
                time.sleep(1)
                continue

            sc["files_match"] += 1
            sc["total_match"] += len(matched)

            # Edge update 통계
            for ev in matched:
                for _, _, _, _, label in ev.get("edge_upd", []):
                    if label == "NEW":
                        sc["edge_new"] += 1
                    else:
                        sc["edge_str"] += 1

            p(f"  Matched: {len(matched)} events "
              f"(company={cats['company']}, both={cats['both']})")

            # --- 상위 이벤트 표시 ---
            show_n = min(len(matched), args.max_per_file)
            for j in range(show_n):
                show_event(matched[j], j, show_n, id2name, sc)

            if len(matched) > show_n:
                p(f"\n  {DIM}... +{len(matched) - show_n} more events "
                  f"(top {show_n} by severity){RESET}")

            # --- 누적 스코어보드 ---
            tot = sc["ok"] + sc["fail"]
            rate = sc["ok"] / tot * 100 if tot > 0 else 0
            cur_edges = ctx["G"].number_of_edges()
            p(f"\n  {DIM}{'─' * 66}{RESET}")
            p(f"  {BOLD}Score: {GREEN}{sc['ok']}{RESET}/{tot} "
              f"({_rate_color(rate)}{rate:.1f}%{RESET})  |  "
              f"{GREEN}OK {sc['ok']}{RESET}  {RED}FAIL {sc['fail']}{RESET}  |  "
              f"File {i+1}/{len(timestamps)}")
            p(f"  {DIM}KG: {cur_edges} edges "
              f"({initial_edges} static + {sc['edge_new']} new + "
              f"{sc['edge_str']} strengthened){RESET}")
            p()

            # --- 대기 ---
            if i < len(timestamps) - 1:
                time.sleep(args.delay)

    except KeyboardInterrupt:
        p(f"\n\n  {YELLOW}Interrupted by user.{RESET}")

    # ── 최종 결과 ──
    tot = sc["ok"] + sc["fail"]
    rate = sc["ok"] / tot * 100 if tot > 0 else 0
    p(f"\n{BOLD}{'=' * 70}{RESET}")
    p(f"{BOLD}  REPLAY COMPLETE{RESET}")
    p(f"{'=' * 70}")
    p(f"  Period:           {args.start} ~ {args.end} ({args.interval})")
    p(f"  Files downloaded: {sc['files_ok']}/{len(timestamps)}")
    p(f"  Files w/ matches: {sc['files_match']}")
    p(f"  Total matched:    {sc['total_match']} events")
    p(f"  Predictions:      {tot} checked")
    p(f"  Success:          {GREEN}{sc['ok']}{RESET} ({rate:.1f}%)")
    p(f"  Failure:          {RED}{sc['fail']}{RESET} ({100-rate:.1f}%)")
    p(f"  KG updates:       +{sc['edge_new']} new edges, "
      f"{sc['edge_str']} strengthened")
    p(f"{'=' * 70}\n")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Butterfly Effect - GDELT v2 Raw File Historical Replay")
    parser.add_argument(
        "--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--interval", choices=["daily", "weekly", "hourly"],
        default="daily", help="Sampling interval (default: daily)")
    parser.add_argument(
        "--delay", type=float, default=10.0,
        help="Delay between files in seconds (default: 10)")
    parser.add_argument(
        "--max-per-file", type=int, default=3,
        help="Max events to display per file (default: 3)")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable local file cache")
    args = parser.parse_args()

    replay(args)


if __name__ == "__main__":
    main()
