"""Add final batch of verified edges from EDINET/web search."""
import pandas as pd

seed = pd.read_csv('data/seed/seed_edges.csv')
print(f'Before: {len(seed)} edges')
existing_keys = set(zip(seed['src_company_id'], seed['rel_type'], seed['dst_company_id']))

new_edges = [
    # === Toyota (7203.T) — PPES JV with Panasonic, CATL partnership ===
    {'src_company_id': '300750.SZ', 'rel_type': 'SUPPLIES', 'dst_company_id': '7203.T',
     'confidence_plink': 1.0, 'strength': 0.9,
     'evidence': 'CATL comprehensive partnership with Toyota for EV batteries (utilitydive.com)',
     'source': 'Industry_News_utilitydive', 'valid_from': '2019-01-01', 'valid_to': ''},

    # === Honda (7267.T) — CATL in China, AESC in Japan ===
    {'src_company_id': '300750.SZ', 'rel_type': 'SUPPLIES', 'dst_company_id': '7267.T',
     'confidence_plink': 1.0, 'strength': 0.8,
     'evidence': 'CATL supplies batteries to Honda in China (Honda Global 2023)',
     'source': 'Honda_Global_2023', 'valid_from': '2020-01-01', 'valid_to': ''},

    # === BASF — CATL framework agreement 2025, LG partnership ===
    {'src_company_id': 'BAS.DE', 'rel_type': 'SUPPLIES', 'dst_company_id': '373220.KS',
     'confidence_plink': 0.9, 'strength': 0.7,
     'evidence': 'BASF and LG piloted single-crystal NMC-90 cathode (chargedevs.com)',
     'source': 'Industry_News_chargedevs', 'valid_from': '2022-01-01', 'valid_to': ''},
    {'src_company_id': 'BAS.DE', 'rel_type': 'SUPPLIES', 'dst_company_id': '300750.SZ',
     'confidence_plink': 1.0, 'strength': 0.8,
     'evidence': 'BASF-CATL framework agreement for CAM supply (basf.com 2025-07-28)',
     'source': 'BASF_PR_2025-07', 'valid_from': '2021-01-01', 'valid_to': ''},

    # === BAIC BluePark (600733) — CATL JV confirmed ===
    {'src_company_id': '300750.SZ', 'rel_type': 'SUPPLIES', 'dst_company_id': '600733.SH',
     'confidence_plink': 1.0, 'strength': 0.8,
     'evidence': 'BAIC-CATL-Xiaomi JV for battery manufacturing (stcn.com 2024)',
     'source': 'Industry_News_stcn_2024', 'valid_from': '2023-01-01', 'valid_to': ''},

    # === Great Wall Motor (601633) — Honeycomb Energy (蜂巢能源) 84% share ===
    {'src_company_id': '300750.SZ', 'rel_type': 'SUPPLIES', 'dst_company_id': '601633.SH',
     'confidence_plink': 0.8, 'strength': 0.5,
     'evidence': 'CATL also supplies to Great Wall (secondary to Honeycomb, OFweek)',
     'source': 'Industry_Analysis_OFweek_2023', 'valid_from': '2022-01-01', 'valid_to': ''},

    # === JAC Motors (600418) — Gotion primary supplier since 2008 ===
    {'src_company_id': '002074.SZ', 'rel_type': 'SUPPLIES', 'dst_company_id': '600418.SH',
     'confidence_plink': 1.0, 'strength': 0.9,
     'evidence': 'Gotion primary battery supplier to JAC since 2008 (OFweek 2023)',
     'source': 'Industry_News_OFweek_2023', 'valid_from': '2008-01-01', 'valid_to': ''},

    # === CMOC (603993) — world largest cobalt producer, supplies to CATL/LGES ===
    {'src_company_id': '603993.SH', 'rel_type': 'SUPPLIES', 'dst_company_id': 'GLEN.L',
     'confidence_plink': 0.8, 'strength': 0.6,
     'evidence': 'CMOC and Glencore are top 2 global cobalt producers, trade relationship via IXM',
     'source': 'Industry_Analysis_2023', 'valid_from': '2020-01-01', 'valid_to': ''},
    {'src_company_id': '603993.SH', 'rel_type': 'SUPPLIES', 'dst_company_id': '1211.HK',
     'confidence_plink': 0.8, 'strength': 0.6,
     'evidence': 'CMOC supplies cobalt/copper to BYD supply chain (industry analysis)',
     'source': 'Industry_Analysis_2023', 'valid_from': '2021-01-01', 'valid_to': ''},

    # === L&F (066970.KS) — Samsung SDI, SK On, Tesla confirmed ===
    {'src_company_id': '066970.KS', 'rel_type': 'SUPPLIES', 'dst_company_id': 'TSLA',
     'confidence_plink': 1.0, 'strength': 0.9,
     'evidence': 'L&F $2.91B cathode supply deal with Tesla 2024-2025 (KED Global 2023-02)',
     'source': 'KED_Global_2023', 'valid_from': '2024-01-01', 'valid_to': '2025-12-31'},
    {'src_company_id': '066970.KS', 'rel_type': 'SUPPLIES', 'dst_company_id': '006400.KS',
     'confidence_plink': 1.0, 'strength': 0.9,
     'evidence': 'Samsung SDI 1.6T won LFP cathode supply from L&F (Seoul Economic Daily 2026-03)',
     'source': 'Seoul_Economic_Daily_2026', 'valid_from': '2027-01-01', 'valid_to': ''},
    {'src_company_id': '066970.KS', 'rel_type': 'SUPPLIES', 'dst_company_id': '096770.KS',
     'confidence_plink': 0.9, 'strength': 0.7,
     'evidence': 'L&F counts SK On among major clients (KED Global 2024-11)',
     'source': 'KED_Global_2024', 'valid_from': '2022-01-01', 'valid_to': ''},

    # === Li-Cycle — Glencore acquired Li-Cycle Aug 2025 ===
    {'src_company_id': 'GLEN.L', 'rel_type': 'OWNS', 'dst_company_id': 'LICY',
     'confidence_plink': 1.0, 'strength': 1.0,
     'evidence': 'Glencore acquired Li-Cycle out of bankruptcy for $40M, Aug 2025 (wastedive.com)',
     'source': 'Industry_News_wastedive_2025', 'valid_from': '2025-08-08', 'valid_to': ''},

    # === Jiayuan (688388) — BYD confirmed, EVE Energy ===
    {'src_company_id': '688388.SH', 'rel_type': 'SUPPLIES', 'dst_company_id': '300014.SZ',
     'confidence_plink': 0.8, 'strength': 0.5,
     'evidence': 'Jiayuan supplies copper foil to EVE Energy (industry analysis)',
     'source': 'Industry_Analysis_2023', 'valid_from': '2021-01-01', 'valid_to': ''},
]

truly_new = [e for e in new_edges if (e['src_company_id'], e['rel_type'], e['dst_company_id']) not in existing_keys]
print(f'New edges to add: {len(truly_new)}')
for e in truly_new:
    print(f'  {e["src_company_id"]} --{e["rel_type"]}--> {e["dst_company_id"]}')

if truly_new:
    new_df = pd.DataFrame(truly_new)
    for col in seed.columns:
        if col not in new_df.columns:
            new_df[col] = ''
    new_df = new_df[seed.columns]
    final = pd.concat([seed, new_df], ignore_index=True)
    final.to_csv('data/seed/seed_edges.csv', index=False, encoding='utf-8-sig')
    print(f'After: {len(final)} edges')
else:
    print('No new edges.')
