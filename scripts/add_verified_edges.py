"""Add verified edges from CNINFO web search + industry reports."""
import pandas as pd

seed = pd.read_csv('data/seed/seed_edges.csv')
print(f'Before: {len(seed)} edges')
existing_keys = set(zip(seed['src_company_id'], seed['rel_type'], seed['dst_company_id']))

new_edges = [
    # === Nuode (600110) ===
    {'src_company_id': '600110.SH', 'rel_type': 'SUPPLIES', 'dst_company_id': '373220.KS',
     'confidence_plink': 0.9, 'strength': 0.6,
     'evidence': 'Nuode supplies copper foil to LG Chem/LGES (chinabuses.com 2018-09-11)',
     'source': 'Industry_News_chinabuses_2018', 'valid_from': '2018-01-01', 'valid_to': ''},
    {'src_company_id': '600110.SH', 'rel_type': 'SUPPLIES', 'dst_company_id': '300014.SZ',
     'confidence_plink': 0.9, 'strength': 0.6,
     'evidence': 'Nuode supplies copper foil to EVE Energy (OFweek 2022-04)',
     'source': 'Industry_News_OFweek_2022', 'valid_from': '2020-01-01', 'valid_to': ''},
    {'src_company_id': '600110.SH', 'rel_type': 'SUPPLIES', 'dst_company_id': '3931.HK',
     'confidence_plink': 1.0, 'strength': 0.8,
     'evidence': 'Nuode locked 45k ton copper foil order from CALB 2025 (stcn.com 2025-02-20)',
     'source': 'Industry_News_stcn_2025', 'valid_from': '2025-01-01', 'valid_to': ''},

    # === BTR (920185) ===
    {'src_company_id': '920185.BJ', 'rel_type': 'SUPPLIES', 'dst_company_id': '006400.KS',
     'confidence_plink': 1.0, 'strength': 0.8,
     'evidence': 'BTR primary natural graphite supplier to Samsung SDI (cbea.com 2018)',
     'source': 'Industry_Analysis_cbea_2018', 'valid_from': '2017-01-01', 'valid_to': ''},
    {'src_company_id': '920185.BJ', 'rel_type': 'SUPPLIES', 'dst_company_id': '1211.HK',
     'confidence_plink': 0.9, 'strength': 0.7,
     'evidence': 'BTR supplies anode to BYD (jiemian.com 2024-09)',
     'source': 'Industry_Analysis_jiemian_2024', 'valid_from': '2020-01-01', 'valid_to': ''},
    {'src_company_id': '920185.BJ', 'rel_type': 'SUPPLIES', 'dst_company_id': '300014.SZ',
     'confidence_plink': 0.9, 'strength': 0.6,
     'evidence': 'BTR supplies anode to EVE Energy (jiemian.com 2024-09)',
     'source': 'Industry_Analysis_jiemian_2024', 'valid_from': '2020-01-01', 'valid_to': ''},

    # === Huayou Cobalt (603799) ===
    {'src_company_id': '603799.SH', 'rel_type': 'SUPPLIES', 'dst_company_id': 'TSLA',
     'confidence_plink': 1.0, 'strength': 0.8,
     'evidence': 'Huayou signed precursor supply agreement with Tesla 2022-2025 (zhitongcaijing.com)',
     'source': 'Huayou_Tesla_Agreement_2022', 'valid_from': '2022-07-01', 'valid_to': '2025-12-31'},
    {'src_company_id': '603799.SH', 'rel_type': 'SUPPLIES', 'dst_company_id': '003670.KS',
     'confidence_plink': 1.0, 'strength': 0.9,
     'evidence': 'Huayou supply 160k ton precursor to POSCO Chemical 2023-2025 (sina finance 2023-01)',
     'source': 'Huayou_POSCO_Agreement_2023', 'valid_from': '2023-01-01', 'valid_to': '2025-12-31'},
    {'src_company_id': '603799.SH', 'rel_type': 'SUPPLIES', 'dst_company_id': '1211.HK',
     'confidence_plink': 0.9, 'strength': 0.7,
     'evidence': 'Huayou supplies high-nickel precursor to BYD (21jingji.com)',
     'source': 'Industry_Analysis_21jingji', 'valid_from': '2021-01-01', 'valid_to': ''},
    {'src_company_id': '603799.SH', 'rel_type': 'SUPPLIES', 'dst_company_id': '006400.KS',
     'confidence_plink': 0.9, 'strength': 0.7,
     'evidence': 'Huayou entered Samsung SDI supply chain (21jingji.com)',
     'source': 'Industry_Analysis_21jingji', 'valid_from': '2021-01-01', 'valid_to': ''},

    # === CALB (3931.HK) ===
    {'src_company_id': '3931.HK', 'rel_type': 'SUPPLIES', 'dst_company_id': '601238.SH',
     'confidence_plink': 1.0, 'strength': 0.9,
     'evidence': 'GAC Group is CALB largest customer for 3 consecutive years ~30% rev (stcn.com)',
     'source': 'Industry_News_stcn_2023', 'valid_from': '2020-01-01', 'valid_to': ''},
    {'src_company_id': '3931.HK', 'rel_type': 'SUPPLIES', 'dst_company_id': '601633.SH',
     'confidence_plink': 0.9, 'strength': 0.7,
     'evidence': 'CALB supplies batteries to Great Wall Motor (marklines.com)',
     'source': 'Industry_Analysis_marklines', 'valid_from': '2021-01-01', 'valid_to': ''},
    {'src_company_id': '3931.HK', 'rel_type': 'SUPPLIES', 'dst_company_id': '600006.SH',
     'confidence_plink': 0.9, 'strength': 0.7,
     'evidence': 'CALB supplies batteries to Dongfeng Motor (baike.baidu.com)',
     'source': 'Industry_Analysis_baidu', 'valid_from': '2020-01-01', 'valid_to': ''},

    # === Benchmark/SNE industry reports ===
    {'src_company_id': '300750.SZ', 'rel_type': 'SUPPLIES', 'dst_company_id': 'F',
     'confidence_plink': 1.0, 'strength': 0.8,
     'evidence': 'CATL supplies batteries to Ford (Benchmark Minerals 2024)',
     'source': 'Benchmark_Minerals_2024', 'valid_from': '2022-01-01', 'valid_to': ''},
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
    print('No new edges to add.')
