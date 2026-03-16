"""나비효과(Butterfly Effect) 파이프라인 End-to-End 검증 스크립트

EV-배터리 공급망 리스크 MVP 파이프라인을 end-to-end로 검증합니다:
  GDELT 뉴스 탐지 → 엔티티 링킹 → 공급망 KG 전파 → 주가 반응(CAR) 검증

사용법:
    python validate_pipeline.py
"""
import sys
import hashlib
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import yaml
from rapidfuzz import fuzz

# ── 경로 설정 ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.entity_link import _build_lookup, link_entities
from src.classify import classify_risk

# ── 로깅 설정 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("validate")

# ── 상수 ──
DATE_START = pd.Timestamp("2023-01-01")
DATE_END = pd.Timestamp("2025-12-31")
MIN_TERM_LEN = 3
SUPPLY_RELS = {"SUPPLIES", "BUYS_FROM", "PARTNERS_WITH", "OWNS", "COMPETES_WITH"}

# Route nodes: 지정학/물류 리스크의 지리적 전파 경로
ROUTE_NODES = {
    "ROUTE_SUEZ": {
        "keywords": ["suez canal", "suez"],
        "connected_regions": ["China", "Germany", "United Kingdom"],
        "edge_strength": 0.7,
    },
    "ROUTE_HORMUZ": {
        "keywords": ["strait of hormuz", "hormuz"],
        "connected_regions": ["China", "Japan", "South Korea", "Germany"],
        "edge_strength": 0.6,
    },
    "ROUTE_MALACCA": {
        "keywords": ["malacca strait", "malacca"],
        "connected_regions": ["China", "Japan", "South Korea"],
        "edge_strength": 0.6,
    },
}

# 추가 리스크 키워드 (단일 단어) — 매칭률 향상용
EXTRA_RISK_KEYWORDS = {
    "geopolitical": ["tariff", "sanction", "conflict", "invasion", "missile", "nuclear", "war"],
    "logistics": ["shipping", "freight", "port", "blockade", "canal", "strait"],
    "natural": ["earthquake", "flood", "typhoon", "hurricane", "tsunami", "wildfire", "drought"],
    "energy": ["oil", "crude", "opec", "refinery", "pipeline", "fuel"],
}

# CAR 설정
CAR_WINDOWS = [1, 3, 5, 10, 21]
ESTIMATION_WINDOW = (-120, -20)

# DQ 임계값
DQ_MIN_EVENTS = 50
DQ_MIN_ENTITY_RATE = 0.01
DQ_MIN_RISK_RATE = 0.01


# ═══════════════════════════════════════════════════════════════════════
# PHASE 0: 데이터 로딩
# ═══════════════════════════════════════════════════════════════════════
def load_data():
    """모든 데이터 파일 로딩 및 검증."""
    log.info("=" * 72)
    log.info("  PHASE 0: 데이터 로딩")
    log.info("=" * 72)

    # Universe
    uni = pd.read_csv(PROJECT_ROOT / "data/universe/univers_final.csv", encoding="utf-8-sig")
    log.info(f"  [LOAD] Universe: {len(uni)} companies")

    # GDELT events (BigQuery 8-year)
    articles = pd.read_csv(PROJECT_ROOT / "data/raw/gdelt/articles_8y.csv")
    articles["event_date"] = pd.to_datetime(articles["event_date"], errors="coerce")
    total_raw = len(articles)
    mask = (articles["event_date"] >= DATE_START) & (articles["event_date"] <= DATE_END)
    articles = articles[mask].copy().reset_index(drop=True)
    log.info(f"  [LOAD] GDELT events: {total_raw} total -> {len(articles)} in range")

    # KG edges
    edges = pd.read_parquet(PROJECT_ROOT / "data/processed/kg_edges.parquet")
    log.info(f"  [LOAD] KG edges: {len(edges)} relationships")

    # Risk keywords
    kw_path = PROJECT_ROOT / "configs/risk_keywords.yaml"
    with open(kw_path, "r", encoding="utf-8") as f:
        keyword_map = yaml.safe_load(f)
    total_kw = sum(len(v["keywords"]) for v in keyword_map.values())
    log.info(f"  [LOAD] Risk keywords: {total_kw} across {len(keyword_map)} categories")

    # Prices
    prices = pd.read_parquet(PROJECT_ROOT / "data/processed/prices_daily.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    log.info(f"  [LOAD] Prices: {len(prices):,} rows, {prices['company_id'].nunique()} companies")
    log.info(f"         Date range: {prices['date'].min().date()} ~ {prices['date'].max().date()}")

    return uni, articles, edges, keyword_map, prices


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: 뉴스 탐지 + 엔티티 링킹
# ═══════════════════════════════════════════════════════════════════════
def _classify_risk_extended(text: str, keyword_map: dict) -> list[str]:
    """기본 키워드 + 확장 키워드로 리스크 분류."""
    # 기본 classify_risk 사용
    result = classify_risk(text, keyword_map)
    if result != ["other"]:
        return result

    # 확장 키워드 매칭 (단일 단어)
    text_lower = text.lower()
    labels = []
    for risk_type, keywords in EXTRA_RISK_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                labels.append(risk_type)
                break
    return labels if labels else ["other"]


def phase1_detection(articles, uni, keyword_map):
    """뉴스 탐지: 엔티티 링킹 + 리스크 키워드 분류 + severity 계산."""
    log.info("")
    log.info("=" * 72)
    log.info("  PHASE 1: 뉴스 탐지 (리스크 이벤트 + 엔티티 링킹)")
    log.info("=" * 72)
    t0 = time.time()

    # Entity lookup 구축 (짧은 검색어 제거)
    lookup = _build_lookup(uni)
    for entry in lookup:
        entry["terms"] = {t for t in entry["terms"] if len(t) >= MIN_TERM_LEN}
    lookup = [e for e in lookup if e["terms"]]
    log.info(f"  Entity lookup: {len(lookup)} companies (min_term_len={MIN_TERM_LEN})")

    results = []
    entity_link_total = 0
    risk_match_total = 0

    for idx, row in articles.iterrows():
        if idx % 2000 == 0 and idx > 0:
            log.info(f"  ... processing {idx}/{len(articles)} events ...")

        # ── 엔티티 링킹 (Actor 이름 기반) ──
        all_ids = []
        all_mentions = []
        link_details = []

        for col in ["Actor1Name", "Actor2Name"]:
            actor = row.get(col)
            if pd.notna(actor) and str(actor).strip():
                actor_str = str(actor).strip()
                ids, mentions = link_entities(actor_str, lookup, min_fuzzy_score=90, max_entities=3)
                for eid, mention in zip(ids, mentions):
                    if eid not in all_ids:
                        all_ids.append(eid)
                        all_mentions.append(mention)
                        score = fuzz.partial_ratio(mention, actor_str.lower())
                        link_details.append((actor_str, eid, score))

        # ── 리스크 키워드 분류 ──
        text_parts = []
        for col in ["SOURCEURL", "Actor1Name", "Actor2Name"]:
            val = row.get(col)
            if pd.notna(val):
                text_parts.append(str(val))
        text = " ".join(text_parts)

        risk_types = _classify_risk_extended(text, keyword_map)
        has_risk = risk_types != ["other"]

        # ── Severity (GoldsteinScale + AvgTone) ──
        gs = float(row["GoldsteinScale"]) if pd.notna(row.get("GoldsteinScale")) else 0
        at = float(row["AvgTone"]) if pd.notna(row.get("AvgTone")) else 0
        norm_gs = np.clip((10 - gs) / 20.0, 0, 1)
        norm_at = np.clip((10 - at) / 20.0, 0, 1)
        severity = float(np.clip(1.0 + (norm_gs * 2.5) + (norm_at * 1.5), 1.0, 5.0))

        # ── 매칭된 키워드 추출 ──
        matched_kw = []
        text_lower = text.lower()
        for rtype, cfg in keyword_map.items():
            for kw in cfg["keywords"]:
                if kw.lower() in text_lower:
                    matched_kw.append(kw)
        for rtype, kws in EXTRA_RISK_KEYWORDS.items():
            for kw in kws:
                if kw in text_lower and kw not in matched_kw:
                    matched_kw.append(kw)

        has_entity = len(all_ids) > 0
        if has_entity:
            entity_link_total += 1
        if has_risk:
            risk_match_total += 1

        # ── 카테고리 결정 (OR 조건) ──
        if has_entity and has_risk:
            category = "both"
        elif has_entity:
            category = "company"
        elif has_risk:
            category = "risk"
        else:
            category = "none"

        if category != "none":
            results.append({
                "GLOBALEVENTID": row["GLOBALEVENTID"],
                "event_date": row["event_date"],
                "Actor1Name": row.get("Actor1Name", ""),
                "Actor2Name": row.get("Actor2Name", ""),
                "EventCode": row.get("EventCode", ""),
                "GoldsteinScale": gs,
                "AvgTone": at,
                "SOURCEURL": str(row.get("SOURCEURL", "")),
                "entity_ids": all_ids,
                "entity_mentions": all_mentions,
                "risk_types": risk_types if has_risk else [],
                "severity": round(severity, 3),
                "event_category": category,
                "matched_keywords": matched_kw,
                "link_details": link_details,
            })

    events_df = pd.DataFrame(results)
    elapsed = time.time() - t0
    log.info(f"\n  처리 완료: {elapsed:.1f}s")

    # ── DQ 검증 ──
    if len(events_df) < DQ_MIN_EVENTS:
        log.info(f"  [DQ_FAIL] reason=too_few_events ({len(events_df)} < {DQ_MIN_EVENTS})")
        return pd.DataFrame()

    ent_rate = entity_link_total / len(articles) if len(articles) > 0 else 0
    risk_rate = risk_match_total / len(articles) if len(articles) > 0 else 0

    if ent_rate < DQ_MIN_ENTITY_RATE:
        log.info(f"  [DQ_FAIL] reason=entity_link_rate_too_low ({ent_rate:.3f} < {DQ_MIN_ENTITY_RATE})")
    if risk_rate < DQ_MIN_RISK_RATE:
        log.info(f"  [DQ_FAIL] reason=risk_match_rate_too_low ({risk_rate:.3f} < {DQ_MIN_RISK_RATE})")

    # ── 통계 ──
    log.info(f"\n  === 탐지 결과 요약 ===")
    log.info(f"  총 이벤트: {len(events_df)} / {len(articles)} ({len(events_df)/len(articles)*100:.1f}%)")
    cat_counts = events_df["event_category"].value_counts()
    for cat, cnt in cat_counts.items():
        log.info(f"    {cat:>10}: {cnt}")
    log.info(f"  Entity link rate: {ent_rate:.1%}")
    log.info(f"  Risk match rate:  {risk_rate:.1%}")

    # ── [DETECTED] 로그 (리스크 이벤트 상위 30건) ──
    risk_events = events_df[events_df["risk_types"].apply(len) > 0].sort_values("severity", ascending=False)
    log.info(f"\n  --- [DETECTED] 리스크 이벤트 (severity 상위 30건) ---")
    for i, (_, ev) in enumerate(risk_events.head(30).iterrows()):
        rt = ",".join(ev["risk_types"])
        url = str(ev["SOURCEURL"])
        # headline = URL의 마지막 부분에서 추출
        headline = url.split("/")[-1][:70] if "/" in url else url[:70]
        kw = ",".join(ev["matched_keywords"][:3])
        log.info(
            f'  [DETECTED] {ev["event_date"].date()} | risk_type={rt} | severity={ev["severity"]:.3f} '
            f'| headline="{headline}" | matched_keywords={{{kw}}}'
        )

    # ── [ENTITY_LINK] 로그 ──
    entity_events = events_df[events_df["entity_ids"].apply(len) > 0]
    log.info(f"\n  --- [ENTITY_LINK] 엔티티 링크 샘플 (최대 30건) ---")
    link_logged = 0
    for _, ev in entity_events.head(100).iterrows():
        for actor_name, company_id, score in ev["link_details"]:
            log.info(
                f'  [ENTITY_LINK] "{actor_name}" -> company_id="{company_id}" (score={score:.2f})'
            )
            link_logged += 1
            if link_logged >= 30:
                break
        if link_logged >= 30:
            break

    # ── 기업별 매칭 분포 ──
    entity_counter = defaultdict(int)
    for ids in events_df["entity_ids"]:
        for eid in ids:
            entity_counter[eid] += 1
    if entity_counter:
        log.info(f"\n  --- 기업별 매칭 빈도 ---")
        for cid, cnt in sorted(entity_counter.items(), key=lambda x: -x[1])[:15]:
            name = uni.loc[uni["company_id"] == cid, "canonical_name"].values
            name_str = name[0] if len(name) > 0 else "?"
            log.info(f"    {cid:>12} ({name_str:>25}): {cnt} events")

    return events_df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Knowledge Graph 구축
# ═══════════════════════════════════════════════════════════════════════
def phase2_build_kg(edges, uni):
    """공급망 KG 구축 + Route Node 추가."""
    log.info("")
    log.info("=" * 72)
    log.info("  PHASE 2: 공급망 Knowledge Graph 구축")
    log.info("=" * 72)

    G = nx.Graph()

    # ── 기업 노드 ──
    for _, row in uni.iterrows():
        G.add_node(
            row["company_id"],
            name=row["canonical_name"],
            country=row.get("country", ""),
            stage=row.get("stage_bucket", ""),
            node_type="company",
        )

    # ── 공급망 엣지 (관련 관계 유형만) ──
    sc_edges = edges[edges["rel_type"].isin(SUPPLY_RELS)]
    for _, row in sc_edges.iterrows():
        src, dst = row["src_company_id"], row["dst_company_id"]
        if src in G.nodes and dst in G.nodes:
            w = float(row.get("strength", 0.5)) * float(row.get("confidence_plink", 1.0))
            G.add_edge(src, dst, weight=w, rel_type=row["rel_type"],
                       evidence=str(row.get("evidence", "")))

    log.info(f"  Base KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    log.info(f"  Edge types: {dict(sc_edges['rel_type'].value_counts())}")

    # ── Route Node 추가 ──
    country_to_companies = defaultdict(list)
    for _, row in uni.iterrows():
        country_to_companies[row.get("country", "")].append(row["company_id"])

    for route_id, info in ROUTE_NODES.items():
        G.add_node(route_id, name=route_id, node_type="route")
        connected = set()
        for region in info["connected_regions"]:
            for cid in country_to_companies.get(region, []):
                connected.add(cid)
        for cid in connected:
            G.add_edge(route_id, cid, weight=info["edge_strength"],
                       rel_type="SHIPS_VIA",
                       evidence=f"Trade route dependency ({route_id})")
        log.info(f"  Route node {route_id}: -> {len(connected)} companies {info['connected_regions']}")

    log.info(f"  Final KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    components = list(nx.connected_components(G))
    log.info(f"  Connected components: {len(components)} (largest: {max(len(c) for c in components)} nodes)")

    return G


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: 공급망 관계 변화 감지
# ═══════════════════════════════════════════════════════════════════════
def phase3_edge_updates(events_df, edges):
    """2개 이상 기업이 동시 언급된 이벤트에서 관계 변화 감지."""
    log.info("")
    log.info("=" * 72)
    log.info("  PHASE 3: 공급망 관계 변화 감지")
    log.info("=" * 72)

    # 기존 엣지 셋 구축
    edge_strengths = {}
    for _, row in edges.iterrows():
        key = (row["src_company_id"], row["dst_company_id"])
        edge_strengths[key] = float(row.get("strength", 0.5))
        edge_strengths[(row["dst_company_id"], row["src_company_id"])] = float(row.get("strength", 0.5))

    # 2개 이상 기업 언급 이벤트
    multi_entity = events_df[events_df["entity_ids"].apply(len) >= 2]
    log.info(f"  Events with 2+ companies: {len(multi_entity)}")

    rel_keywords = {
        "SUPPLIES": ["supply", "supplies", "supplier", "provide", "deliver"],
        "BUYS_FROM": ["purchase", "buy", "procurement", "acquire"],
        "PARTNERS_WITH": ["partner", "joint venture", "collaborate", "alliance", "agreement", "deal", "contract"],
    }

    updates = []
    for _, ev in multi_entity.iterrows():
        entities = ev["entity_ids"]
        full_text = " ".join([
            str(ev.get("SOURCEURL", "")), str(ev.get("Actor1Name", "")), str(ev.get("Actor2Name", ""))
        ]).lower()

        # 관계 유형 추정
        detected_rel = "PARTNERS_WITH"
        for rel_type, keywords in rel_keywords.items():
            if any(kw in full_text for kw in keywords):
                detected_rel = rel_type
                break

        gs = float(ev.get("GoldsteinScale", 0))
        delta = np.clip(gs / 10.0, -0.3, 0.3)

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                src, dst = entities[i], entities[j]
                key = (src, dst)
                rev_key = (dst, src)
                evidence = f"{ev.get('Actor1Name', '')} <-> {ev.get('Actor2Name', '')} ({ev['event_date'].date()})"

                if key in edge_strengths:
                    before = edge_strengths[key]
                    after = float(np.clip(before + delta, 0.1, 1.0))
                    updates.append({"src": src, "dst": dst, "rel_type": detected_rel,
                                    "before": before, "after": after, "evidence": evidence})
                    log.info(
                        f'  [EDGE_UPDATE] {src} -[{detected_rel}]-> {dst} '
                        f'| strength: {before:.2f} -> {after:.2f} | evidence="{evidence}"'
                    )
                elif rev_key in edge_strengths:
                    before = edge_strengths[rev_key]
                    after = float(np.clip(before + delta, 0.1, 1.0))
                    updates.append({"src": dst, "dst": src, "rel_type": detected_rel,
                                    "before": before, "after": after, "evidence": evidence})
                    log.info(
                        f'  [EDGE_UPDATE] {dst} -[{detected_rel}]-> {src} '
                        f'| strength: {before:.2f} -> {after:.2f} | evidence="{evidence}"'
                    )
                else:
                    new_str = float(np.clip(0.3 + delta, 0.1, 1.0))
                    updates.append({"src": src, "dst": dst, "rel_type": detected_rel,
                                    "before": 0.0, "after": new_str, "evidence": evidence})
                    log.info(
                        f'  [EDGE_UPDATE] {src} -[{detected_rel}]-> {dst} '
                        f'| strength: 0.00 -> {new_str:.2f} (NEW) | evidence="{evidence}"'
                    )

    log.info(f"\n  Total edge updates: {len(updates)}")
    return updates


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: 리스크 전파 (Exposure 계산)
# ═══════════════════════════════════════════════════════════════════════
def _rwr_scores(G, source_nodes, restart_prob=0.15, max_iter=50):
    """Random Walk with Restart from multiple source nodes."""
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    restart_vec = np.zeros(n)
    for s in source_nodes:
        if s in node_idx:
            restart_vec[node_idx[s]] = 1.0 / max(len(source_nodes), 1)
    scores = restart_vec.copy()

    # 인접 행렬 기반 전이 (사전 계산)
    adj = defaultdict(list)  # node -> [(neighbor, weight/degree)]
    for nd in nodes:
        neighbors = list(G.neighbors(nd))
        if not neighbors:
            continue
        total_w = sum(G[nd][nb].get("weight", 1.0) for nb in neighbors)
        for nb in neighbors:
            w = G[nd][nb].get("weight", 1.0)
            adj[nd].append((nb, w / total_w))

    for _ in range(max_iter):
        new_scores = np.zeros(n)
        for nd in nodes:
            i = node_idx[nd]
            for nb, trans_prob in adj[nd]:
                j = node_idx[nb]
                new_scores[j] += (1 - restart_prob) * scores[i] * trans_prob
        new_scores += restart_prob * restart_vec
        scores = new_scores

    return {nodes[i]: float(scores[i]) for i in range(n)}


def phase4_propagation(events_df, G, uni):
    """리스크 이벤트별 기업 노출도 계산 (Shortest Path + RWR)."""
    log.info("")
    log.info("=" * 72)
    log.info("  PHASE 4: 리스크 전파 (Exposure 계산)")
    log.info("=" * 72)
    t0 = time.time()

    company_ids = [n for n, d in G.nodes(data=True) if d.get("node_type") == "company"]

    # Route keyword -> route node 매핑
    route_keyword_map = {}
    for route_id, info in ROUTE_NODES.items():
        for kw in info["keywords"]:
            route_keyword_map[kw] = route_id

    # 소스 노드 결정 함수
    def get_source_nodes(ev):
        sources = list(ev["entity_ids"]) if isinstance(ev["entity_ids"], list) else []
        text = " ".join([
            str(ev.get("SOURCEURL", "")),
            str(ev.get("Actor1Name", "")),
            str(ev.get("Actor2Name", ""))
        ]).lower()
        for kw, route_id in route_keyword_map.items():
            if kw in text and route_id in G.nodes:
                if route_id not in sources:
                    sources.append(route_id)
        return list(set(sources))

    # 소스 노드가 있는 이벤트만 필터
    events_with_sources = events_df.copy()
    events_with_sources["source_nodes"] = events_with_sources.apply(get_source_nodes, axis=1)
    events_with_sources = events_with_sources[events_with_sources["source_nodes"].apply(len) > 0]

    # 중복 이벤트 축소 — 날짜 + 소스노드 기준 대표 이벤트 선택
    events_with_sources["_src_key"] = events_with_sources["source_nodes"].apply(
        lambda x: ",".join(sorted(x))
    )
    events_deduped = events_with_sources.sort_values("severity", ascending=False).drop_duplicates(
        subset=["event_date", "_src_key"], keep="first"
    )

    log.info(f"  Events with source nodes: {len(events_with_sources)} -> {len(events_deduped)} (deduped)")

    if len(events_deduped) == 0:
        log.info("  [DQ_FAIL] reason=no_events_with_source_nodes")
        return pd.DataFrame()

    all_exposures = []
    logged = 0

    for evt_idx, (_, ev) in enumerate(events_deduped.iterrows()):
        if evt_idx % 200 == 0 and evt_idx > 0:
            log.info(f"  ... propagating {evt_idx}/{len(events_deduped)} events ...")

        source_nodes = ev["source_nodes"]
        severity = ev["severity"]

        # ── Shortest Path Exposure ──
        sp_exposures = {}
        sp_paths = {}
        for cid in company_ids:
            if cid in source_nodes:
                sp_exposures[cid] = 1.0
                sp_paths[cid] = [cid]
            else:
                min_dist = float("inf")
                best_path = None
                for src in source_nodes:
                    try:
                        path = nx.shortest_path(G, src, cid)
                        dist = len(path) - 1
                        if dist < min_dist:
                            min_dist = dist
                            best_path = path
                    except nx.NetworkXNoPath:
                        pass
                if min_dist < float("inf"):
                    sp_exposures[cid] = 1.0 / (1 + min_dist)
                    sp_paths[cid] = best_path
                else:
                    sp_exposures[cid] = 0.0
                    sp_paths[cid] = []

        # ── RWR Exposure ──
        rwr = _rwr_scores(G, source_nodes)

        # ── [PROPAGATE] + [EXPOSURE] 로그 (상위 10개 이벤트) ──
        if logged < 10:
            route_info = [s for s in source_nodes if s.startswith("ROUTE_")]
            company_info = [s for s in source_nodes if not s.startswith("ROUTE_")]
            route_str = " -> " + ",".join(route_info) if route_info else ""
            src_str = ",".join(company_info) if company_info else "(route only)"

            top5 = sorted(
                [(cid, sp_exposures.get(cid, 0), rwr.get(cid, 0)) for cid in company_ids],
                key=lambda x: x[1], reverse=True,
            )[:5]
            target_str = ", ".join([c[0] for c in top5])

            rt = ",".join(ev["risk_types"]) if isinstance(ev["risk_types"], list) and ev["risk_types"] else "company"
            log.info(
                f'\n  [PROPAGATE] event={ev["GLOBALEVENTID"]} | severity={severity:.3f} '
                f'| risk={rt} | source={src_str}{route_str} -> targets=[{target_str}]'
            )

            for cid, sp_exp, rwr_exp in top5:
                if sp_exp > 0:
                    path = sp_paths.get(cid, [])
                    path_str = " -> ".join(path) if path else "direct"
                    log.info(
                        f'  [EXPOSURE] company_id={cid} | exposure_sp={sp_exp:.4f} '
                        f'| exposure_rwr={rwr_exp:.6f} | path={path_str}'
                    )
            logged += 1

        # ── 모든 노출 저장 ──
        for cid in company_ids:
            sp_exp = sp_exposures.get(cid, 0)
            rwr_exp = rwr.get(cid, 0)
            if sp_exp > 0 or rwr_exp > 0.001:
                all_exposures.append({
                    "GLOBALEVENTID": ev["GLOBALEVENTID"],
                    "event_date": ev["event_date"],
                    "company_id": cid,
                    "exposure_sp": sp_exp,
                    "exposure_rwr": rwr_exp,
                    "severity": severity,
                    "risk_types": ev["risk_types"],
                    "event_category": ev["event_category"],
                    "source_nodes": source_nodes,
                })

    exposure_df = pd.DataFrame(all_exposures)
    elapsed = time.time() - t0

    log.info(f"\n  === 전파 결과 요약 ({elapsed:.1f}s) ===")
    log.info(f"  Total exposure records: {len(exposure_df):,}")
    log.info(f"  Events propagated: {exposure_df['GLOBALEVENTID'].nunique()}")
    log.info(f"  Companies exposed: {exposure_df['company_id'].nunique()}")

    if not exposure_df.empty:
        avg_by_company = exposure_df.groupby("company_id")["exposure_sp"].mean().sort_values(ascending=False)
        log.info(f"\n  --- 기업별 평균 노출도 (상위 10) ---")
        for cid, exp in avg_by_company.head(10).items():
            name = uni.loc[uni["company_id"] == cid, "canonical_name"].values
            name_str = name[0] if len(name) > 0 else "?"
            log.info(f"    {cid:>12} ({name_str:>25}): exposure_sp={exp:.4f}")

    return exposure_df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: 주가 반응 검증 (CAR 분석)
# ═══════════════════════════════════════════════════════════════════════
def phase5_car(exposure_df, prices, uni):
    """(이벤트, 기업) 쌍별 Cumulative Abnormal Return 계산."""
    log.info("")
    log.info("=" * 72)
    log.info("  PHASE 5: 주가 반응 검증 (CAR 분석)")
    log.info("=" * 72)
    t0 = time.time()

    if exposure_df.empty:
        log.info("  [DQ_FAIL] reason=no_exposure_data")
        return pd.DataFrame()

    # ── 주가 수익률 사전 계산 ──
    prices_sorted = prices.sort_values(["company_id", "date"]).copy()
    prices_sorted["return"] = prices_sorted.groupby("company_id")["adj_close"].pct_change()
    price_companies = set(prices["company_id"].unique())

    # ── (event_date, company_id) 쌍 — 최고 노출도 기준 축소 ──
    pairs = exposure_df[exposure_df["exposure_sp"] > 0].copy()
    pairs = pairs.sort_values("exposure_sp", ascending=False).drop_duplicates(
        subset=["event_date", "company_id"], keep="first"
    )
    pairs = pairs[pairs["company_id"].isin(price_companies)]

    log.info(f"  (event, company) pairs: {len(pairs):,}")
    log.info(f"  Companies with price data: {len(price_companies)}")

    if len(pairs) < 10:
        log.info(f"  [DQ_FAIL] reason=too_few_pairs ({len(pairs)} < 10)")
        return pd.DataFrame()

    # ── 기업별 수익률 데이터 캐싱 ──
    returns_cache = {}
    for cid in pairs["company_id"].unique():
        comp_data = prices_sorted[prices_sorted["company_id"] == cid][["date", "return"]].dropna()
        returns_cache[cid] = comp_data.set_index("date")["return"]

    car_results = []
    logged_cars = 0

    for _, pair in pairs.iterrows():
        cid = pair["company_id"]
        ev_date = pd.Timestamp(pair["event_date"])
        exposure = pair["exposure_sp"]

        if cid not in returns_cache or returns_cache[cid].empty:
            continue

        comp_ret = returns_cache[cid]

        # 이벤트일 이후 거래일 찾기
        future_dates = comp_ret.index[comp_ret.index >= ev_date].sort_values()
        if len(future_dates) < 2:
            continue
        next_trade_day = future_dates[0]

        # Estimation window: 기대수익률 계산
        est_start = ev_date + timedelta(days=ESTIMATION_WINDOW[0])
        est_end = ev_date + timedelta(days=ESTIMATION_WINDOW[1])
        est_ret = comp_ret[(comp_ret.index >= est_start) & (comp_ret.index <= est_end)]
        expected_return = est_ret.mean() if len(est_ret) >= 20 else comp_ret.mean()

        # CAR 계산
        car_dict = {
            "company_id": cid,
            "event_date": ev_date,
            "next_trade_day": next_trade_day,
            "exposure_sp": exposure,
            "severity": pair["severity"],
            "risk_types": pair["risk_types"],
            "event_category": pair["event_category"],
        }

        for w in CAR_WINDOWS:
            window_dates = future_dates[:w + 1]
            window_returns = comp_ret.reindex(window_dates).dropna().values
            if len(window_returns) > 0:
                abnormal = window_returns - expected_return
                car_dict[f"CAR_0_{w}"] = float(np.sum(abnormal))
            else:
                car_dict[f"CAR_0_{w}"] = np.nan

        car_1 = car_dict.get("CAR_0_1", np.nan)
        car_3 = car_dict.get("CAR_0_3", np.nan)
        direction_match = (not np.isnan(car_1)) and (car_1 < 0)
        car_dict["direction_match"] = direction_match

        car_results.append(car_dict)

        # ── [CAR_CHECK] 로그 ──
        if logged_cars < 30:
            log.info(
                f'  [CAR_CHECK] company_id={cid} | event_date={ev_date.date()} '
                f'| next_trade_day={next_trade_day.date()} '
                f'| CAR[0,1]={car_dict.get("CAR_0_1", 0):.4f} '
                f'| CAR[0,3]={car_dict.get("CAR_0_3", 0):.4f} '
                f'| exposure={exposure:.4f} | direction_match={direction_match}'
            )
            logged_cars += 1

    car_df = pd.DataFrame(car_results)
    elapsed = time.time() - t0

    if car_df.empty:
        log.info("  [DQ_FAIL] reason=no_car_results")
        return pd.DataFrame()

    log.info(f"\n  === CAR 분석 결과 ({elapsed:.1f}s) ===")
    log.info(f"  분석 대상: {len(car_df):,} (event, company) pairs")

    # ── [GROUP_SUMMARY] ──
    log.info(f"\n  --- [GROUP_SUMMARY] 노출도 상위 vs 하위 ---")
    median_exp = car_df["exposure_sp"].median()
    high_exp = car_df[car_df["exposure_sp"] > median_exp]
    low_exp = car_df[car_df["exposure_sp"] <= median_exp]

    for w in CAR_WINDOWS:
        col = f"CAR_0_{w}"
        if col in car_df.columns:
            h_mean = high_exp[col].mean()
            l_mean = low_exp[col].mean()
            diff = h_mean - l_mean
            log.info(
                f'  [GROUP_SUMMARY] window=[0,{w:>2}] | high_exposure_CAR_mean={h_mean:+.4f} '
                f'| low_exposure_CAR_mean={l_mean:+.4f} | diff={diff:+.4f}'
            )

    # 방향성 일치율
    dm_rate = car_df["direction_match"].mean()
    dm_count = int(car_df["direction_match"].sum())
    log.info(f"\n  Direction match rate (CAR[0,1] < 0): {dm_rate:.1%} ({dm_count}/{len(car_df)})")

    # 리스크 유형별 요약
    log.info(f"\n  --- 리스크 유형별 CAR 평균 ---")
    risk_type_cars = defaultdict(list)
    for _, row in car_df.iterrows():
        rt_list = row["risk_types"]
        if isinstance(rt_list, list):
            for rt in rt_list:
                if rt != "other":
                    risk_type_cars[rt].append(row.get("CAR_0_3", 0))
    for rt, cars in sorted(risk_type_cars.items()):
        valid_cars = [c for c in cars if not np.isnan(c)]
        if valid_cars:
            mean_car = np.mean(valid_cars)
            log.info(f"    {rt:>20}: mean CAR[0,3]={mean_car:+.4f} (n={len(valid_cars)})")

    return car_df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6: 검증 리포트 저장
# ═══════════════════════════════════════════════════════════════════════
def save_report(events_df, exposure_df, car_df, edge_updates, G, uni):
    """reports/validation_summary.md에 검증 결과 저장."""
    log.info("")
    log.info("=" * 72)
    log.info("  PHASE 6: 검증 리포트 저장")
    log.info("=" * 72)

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "validation_summary.md"

    lines = [
        "# Pipeline Validation Summary",
        "",
        f"**검증일시:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**검증 기간:** {DATE_START.date()} ~ {DATE_END.date()}",
        "",
    ]

    # ── 1. 뉴스 탐지 ──
    lines.append("## 1. 뉴스 탐지")
    if not events_df.empty:
        lines.append(f"- 총 매칭 이벤트: {len(events_df)}")
        cat = events_df["event_category"].value_counts()
        for c, n in cat.items():
            lines.append(f"  - {c}: {n}")
        ent_rate = events_df["entity_ids"].apply(len).gt(0).mean()
        risk_rate = events_df["risk_types"].apply(len).gt(0).mean()
        lines.append(f"- 엔티티 링크율: {ent_rate:.1%}")
        lines.append(f"- 리스크 매칭율: {risk_rate:.1%}")
        lines.append(f"- 평균 severity: {events_df['severity'].mean():.3f}")
    lines.append("")

    # ── 2. Knowledge Graph ──
    lines.append("## 2. Knowledge Graph")
    if G:
        comp_count = len([n for n, d in G.nodes(data=True) if d.get("node_type") == "company"])
        route_count = len([n for n, d in G.nodes(data=True) if d.get("node_type") == "route"])
        lines.append(f"- 노드: {G.number_of_nodes()} (기업 {comp_count}, route {route_count})")
        lines.append(f"- 엣지: {G.number_of_edges()}")
        components = list(nx.connected_components(G))
        lines.append(f"- 연결 컴포넌트: {len(components)} (최대 {max(len(c) for c in components)} nodes)")
    lines.append("")

    # ── 3. 관계 변화 ──
    lines.append("## 3. 공급망 관계 변화")
    lines.append(f"- 감지된 엣지 업데이트: {len(edge_updates)}")
    for u in edge_updates[:10]:
        tag = "(NEW)" if u["before"] == 0 else ""
        lines.append(f"  - `{u['src']}` -[{u['rel_type']}]-> `{u['dst']}`: "
                      f"{u['before']:.2f} -> {u['after']:.2f} {tag}")
    lines.append("")

    # ── 4. 리스크 전파 ──
    lines.append("## 4. 리스크 전파")
    if not exposure_df.empty:
        lines.append(f"- 총 노출 레코드: {len(exposure_df):,}")
        lines.append(f"- 전파된 이벤트: {exposure_df['GLOBALEVENTID'].nunique()}")
        lines.append(f"- 노출 기업: {exposure_df['company_id'].nunique()}")
        lines.append("")
        lines.append("### 기업별 평균 노출도 (상위 10)")
        lines.append("| Company ID | Name | Mean Exposure (SP) |")
        lines.append("|---|---|---|")
        avg_exp = exposure_df.groupby("company_id")["exposure_sp"].mean().sort_values(ascending=False).head(10)
        for cid, exp in avg_exp.items():
            name = uni.loc[uni["company_id"] == cid, "canonical_name"].values
            name_str = name[0] if len(name) > 0 else "?"
            lines.append(f"| {cid} | {name_str} | {exp:.4f} |")
    lines.append("")

    # ── 5. CAR 분석 ──
    lines.append("## 5. 주가 반응 (CAR)")
    if not car_df.empty:
        lines.append(f"- 분석 대상: {len(car_df):,} (event, company) pairs")
        dm_rate = car_df["direction_match"].mean()
        lines.append(f"- Direction match rate (CAR[0,1] < 0): {dm_rate:.1%}")
        lines.append("")
        lines.append("### High vs Low Exposure CAR")
        lines.append("| Window | High Exposure | Low Exposure | Diff |")
        lines.append("|--------|--------------|-------------|------|")
        median_exp = car_df["exposure_sp"].median()
        high = car_df[car_df["exposure_sp"] > median_exp]
        low = car_df[car_df["exposure_sp"] <= median_exp]
        for w in CAR_WINDOWS:
            col = f"CAR_0_{w}"
            if col in car_df.columns:
                h = high[col].mean()
                l = low[col].mean()
                lines.append(f"| [0,{w}] | {h:+.4f} | {l:+.4f} | {h - l:+.4f} |")
    lines.append("")

    # ── 6. 검증 결론 ──
    lines.append("## 6. 검증 결론")
    verdicts = []
    if not events_df.empty and len(events_df) >= DQ_MIN_EVENTS:
        verdicts.append("[PASS] 리스크 이벤트 탐지 정상")
    else:
        verdicts.append("[FAIL] 리스크 이벤트 부족")
    if not exposure_df.empty:
        verdicts.append("[PASS] 리스크 전파 정상")
    else:
        verdicts.append("[FAIL] 리스크 전파 실패")
    if not car_df.empty:
        dm_rate = car_df["direction_match"].mean()
        if dm_rate > 0.45:
            verdicts.append(f"[PASS] 방향성 일치율 {dm_rate:.1%}")
        else:
            verdicts.append(f"[WARN] 방향성 일치율 {dm_rate:.1%} (기대: >45%)")
    else:
        verdicts.append("[FAIL] CAR 분석 실패")

    for v in verdicts:
        lines.append(f"- {v}")

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    log.info(f"  Report saved: {report_path}")
    return report_text


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    log.info("")
    log.info("=" * 72)
    log.info("  나비효과(Butterfly Effect) 파이프라인 End-to-End 검증")
    log.info(f"  검증 기간: {DATE_START.date()} ~ {DATE_END.date()}")
    log.info("=" * 72)
    total_t0 = time.time()

    # Phase 0: 데이터 로딩
    uni, articles, edges, keyword_map, prices = load_data()

    # Phase 1: 뉴스 탐지
    events_df = phase1_detection(articles, uni, keyword_map)
    if events_df.empty:
        log.info("\n  [ABORT] Phase 1 실패 — 파이프라인 중단")
        return

    # Phase 2: KG 구축
    G = phase2_build_kg(edges, uni)

    # Phase 3: 관계 변화 감지
    edge_updates = phase3_edge_updates(events_df, edges)

    # Phase 4: 리스크 전파
    exposure_df = phase4_propagation(events_df, G, uni)

    # Phase 5: CAR 분석
    car_df = phase5_car(exposure_df, prices, uni)

    # Phase 6: 리포트 저장
    save_report(events_df, exposure_df, car_df, edge_updates, G, uni)

    total_elapsed = time.time() - total_t0
    log.info("")
    log.info("=" * 72)
    log.info(f"  검증 완료 (총 {total_elapsed:.1f}s)")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
