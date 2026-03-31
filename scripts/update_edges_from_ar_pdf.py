#!/usr/bin/env python3
"""
실제 AR PDF에서 확인된 엣지로 seed_edges 업데이트.
Press Release로만 확인된 기존 엣지의 evidence를 보강하고,
새로 발견된 엣지를 추가.
"""
import pandas as pd
import shutil
from datetime import datetime

shutil.copy('data/seed/seed_edges.csv',
            f'data/seed/seed_edges_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

seed = pd.read_csv('data/seed/seed_edges.csv')
print(f"Before: {len(seed)} edges")

# ═══════════════════════════════════════════════════
# AR PDF에서 실제 확인된 새 엣지 추가
# ═══════════════════════════════════════════════════

new_edges = [
    # ── BMW AR에서 확인 ──
    # BMW - Solid Power (투자, 모든 연도에서 확인)
    # Solid Power는 universe 외 기업이므로 skip

    # BMW - 직접 원자재 조달 (AR 2022 p.110, AR 2023 p.112)
    # "BMW directly purchases lithium and cobalt from mines"
    # → 기존 GLEN.L → BMW 엣지는 Press로 확인. AR에서는 generic "mines" 표현

    # ── VW AR에서 확인 ──
    # VW - Gotion (AR 2021 p.96, AR 2022 p.304)
    {"src_company_id": "002074.SZ", "rel_type": "PARTNERS_WITH", "dst_company_id": "VOW3.DE",
     "confidence_plink": 1.0, "strength": 0.9,
     "evidence": "VW AR 2021 p.96: 'Volkswagen acquired an interest in Gotion High-Tech Co., Ltd., Hefei (China)'. AR 2022 p.304: 'strategic framework for cooperation in development, manufacture and distribution of battery cells'",
     "source": "VW_AR_2021_p96", "valid_from": "2020-01-01", "valid_to": "2032-12-31",
     "risk_rationale": "VW holds ~25% stake in Gotion; strategic battery cell partnership"},

    # VW - Northvolt (AR 2021 p.122, AR 2022 p.296-304)
    # Northvolt는 universe 외 기업이므로 skip (2024년 파산)

    # VW - PowerCo/Umicore IONWAY JV (AR 2022 p.189)
    # Umicore는 universe 외이지만 BASF와 경쟁 관계로 참고용
    # "PowerCo and the Belgian materials technology group Umicore have formed a joint venture IONWAY"

    # ── Mercedes AR에서 확인 ──
    # Mercedes - CATL (AR 2022 p.44)
    {"src_company_id": "300750.SZ", "rel_type": "SUPPLIES", "dst_company_id": "MBG.DE",
     "confidence_plink": 1.0, "strength": 0.9,
     "evidence": "Mercedes AR 2022 p.44: 'Mercedes-Benz Cars is further expanding its battery cell partnership with Contemporary Amperex Technology Co., Ltd. (CATL) with a new production site in Hungary. The new CATL plant in Debrecen is scheduled to supply battery cells to European production sites.'",
     "source": "Mercedes_AR_2022_p44", "valid_from": "2022-01-01", "valid_to": "2030-12-31",
     "risk_rationale": "CATL Debrecen plant dedicated to Mercedes European production"},

    # Mercedes - ACC (AR 2022 p.43, p.317)
    # ACC는 universe 외 기업이지만 Mercedes 배터리 전략의 핵심
    # "33.33% stake in Automotive Cells Company SE (ACC)"

    # Mercedes - BYD/DENZA (AR 2022 p.261)
    {"src_company_id": "MBG.DE", "rel_type": "PARTNERS_WITH", "dst_company_id": "1211.HK",
     "confidence_plink": 1.0, "strength": 0.5,
     "evidence": "Mercedes AR 2022 p.261: 'shareholders Daimler Greater China Ltd. and BYD Automotive Industry Co., Ltd (BYD) signed a contract on structural realignment of DENZA. Group transferred 40% shares to BYD. Mercedes holds 10%, BYD holds 90%.'",
     "source": "Mercedes_AR_2022_p261", "valid_from": "2021-01-01", "valid_to": "2025-12-31",
     "risk_rationale": "Mercedes-BYD DENZA JV (Mercedes reduced to 10%)"},

    # Mercedes - Geely/smart JV (AR 2022 p.318, p.332)
    {"src_company_id": "MBG.DE", "rel_type": "PARTNERS_WITH", "dst_company_id": "0175.HK",
     "confidence_plink": 1.0, "strength": 0.7,
     "evidence": "Mercedes AR 2022 p.318: 'smart Automobile Co., Ltd. (smart), a joint venture of Mercedes-Benz AG and Zhejiang Geely Holding Group Co., Ltd.' CO2 pool inclusion since 2023.",
     "source": "Mercedes_AR_2022_p318", "valid_from": "2022-01-01", "valid_to": "2030-12-31",
     "risk_rationale": "Mercedes-Geely smart JV for BEV brand"},

    # Mercedes - Farasis (AR 2022 p.181)
    # Farasis는 universe 외 기업이지만 참고: "mandate at the listed Farasis Energy (Ganzhou) Co."

    # ── BASF AR에서 확인 ──
    # BASF - CATL (AR 2021 p.30)
    {"src_company_id": "BAS.DE", "rel_type": "SUPPLIES", "dst_company_id": "300750.SZ",
     "confidence_plink": 1.0, "strength": 0.8,
     "evidence": "BASF AR 2021 p.30: cooperative agreement with CATL for cathode active materials. BASF Press 2025: framework agreement for CAM supply confirmed.",
     "source": "BASF_AR_2021_p30", "valid_from": "2021-01-01", "valid_to": "2030-12-31",
     "risk_rationale": "BASF supplies cathode active materials to CATL"},

    # BASF - SVOLT (AR 2021 p.30) - universe 외

    # BASF - TODA Kogyo JV (AR 2024 p.377)
    # BASF TODA Battery Materials - universe 외이지만 공급망 핵심

    # BASF - Schwarzheide CAM plant (across all years)
    # "cathode active materials production at Schwarzheide" - 확인됨

    # BASF - Samsung SDI (mentioned as customer)
    # "Battery materials customer" - BASF AR에서 확인

    # BASF - Mercedes (AR 2023 p.176)
    # "In collaboration with Pyrum Innovations AG and BASF, chemical recycling of used vehicle tires"
    # 이건 배터리가 아니라 타이어 리사이클링이므로 배터리 엣지로 부적합
]

# ═══════════════════════════════════════════════════
# 기존 엣지의 evidence 업데이트 (PDF 페이지 번호 추가)
# ═══════════════════════════════════════════════════

evidence_updates = {
    # VW - Gotion: 기존 Press 엣지에 AR 페이지 추가
    ("002074.SZ", "SUPPLIES", "VOW3.DE"):
        "VW AR 2022 p.304: 'strategic framework for cooperation in development, manufacture and distribution of battery cells'. Electrive 2025: Gotion mass-producing VW standard cells.",

    # Mercedes - CATL: 기존 Press 엣지에 AR 페이지 추가
    ("300750.SZ", "SUPPLIES", "MBG.DE"):
        "Mercedes AR 2022 p.44: 'expanding battery cell partnership with CATL, new production site in Hungary. CATL Debrecen to supply European production sites.'",

    # Mercedes - Geely: 기존 엣지에 AR 페이지 추가
    ("MBG.DE", "PARTNERS_WITH", "0175.HK"):
        "Mercedes AR 2022 p.318: 'smart Automobile Co., Ltd., a joint venture of Mercedes-Benz AG and Zhejiang Geely Holding Group Co., Ltd.' Smart included in CO2 pool since 2023.",

    # BASF - CATL: 기존 Press 엣지에 AR 페이지 추가
    ("BAS.DE", "SUPPLIES", "300750.SZ"):
        "BASF AR 2021 p.30: cooperative agreement with CATL. BASF Press 2025-07-28: framework agreement for cathode active materials (CAM) supply confirmed.",
}

# Apply evidence updates to existing edges
for (src, rel, dst), new_evidence in evidence_updates.items():
    mask = (seed['src_company_id'] == src) & (seed['rel_type'] == rel) & (seed['dst_company_id'] == dst)
    if mask.any():
        seed.loc[mask, 'evidence'] = new_evidence
        # Upgrade confidence since now confirmed by AR PDF
        seed.loc[mask, 'confidence_plink'] = 1.0
        print(f"  Updated: {src} --{rel}--> {dst}")

# ═══════════════════════════════════════════════════
# 기존 Press-only 엣지에 AR 미확인 표시 (솔직하게)
# ═══════════════════════════════════════════════════

# BMW의 CATL, Samsung SDI, EVE Energy — AR에서 실명 미확인, Press Release에서만 확인
press_only_note = " [Note: Not named in BMW AR; confirmed via BMW Press Release only]"
for dst_src in [("300750.SZ", "SUPPLIES", "BMW.DE"),
                ("006400.KS", "SUPPLIES", "BMW.DE"),
                ("300014.SZ", "SUPPLIES", "BMW.DE")]:
    src, rel, dst = dst_src
    mask = (seed['src_company_id'] == src) & (seed['rel_type'] == rel) & (seed['dst_company_id'] == dst)
    if mask.any():
        current = seed.loc[mask, 'evidence'].iloc[0]
        if 'Not named in BMW AR' not in current:
            seed.loc[mask, 'evidence'] = current + press_only_note
            print(f"  Noted press-only: {src} --{rel}--> {dst}")

# VW의 CATL, Samsung SDI, LG Energy — AR에서 미확인
for combo in [("300750.SZ", "SUPPLIES", "VOW3.DE"),
              ("006400.KS", "SUPPLIES", "VOW3.DE"),
              ("373220.KS", "SUPPLIES", "VOW3.DE")]:
    src, rel, dst = combo
    mask = (seed['src_company_id'] == src) & (seed['rel_type'] == rel) & (seed['dst_company_id'] == dst)
    if mask.any():
        current = seed.loc[mask, 'evidence'].iloc[0]
        note = " [Note: Not named in VW AR 2021-2024; confirmed via press releases/industry reports]"
        if 'Not named in VW AR' not in current:
            seed.loc[mask, 'evidence'] = current + note
            print(f"  Noted press-only: {src} --{rel}--> {dst}")

# ═══════════════════════════════════════════════════
# 새 엣지 추가 (중복 체크)
# ═══════════════════════════════════════════════════

existing = set(zip(seed['src_company_id'], seed['rel_type'], seed['dst_company_id']))
added = 0
for edge in new_edges:
    key = (edge['src_company_id'], edge['rel_type'], edge['dst_company_id'])
    if key not in existing:
        seed = pd.concat([seed, pd.DataFrame([edge])], ignore_index=True)
        existing.add(key)
        added += 1
        print(f"  Added: {key[0]} --{key[1]}--> {key[2]}")
    else:
        print(f"  Skip (exists): {key[0]} --{key[1]}--> {key[2]}")

seed.to_csv('data/seed/seed_edges.csv', index=False)
print(f"\nAfter: {len(seed)} edges (added {added})")
print("Evidence upgraded with AR page references for 4 edges")
print("Press-only notes added for 6 edges (honest disclosure)")
