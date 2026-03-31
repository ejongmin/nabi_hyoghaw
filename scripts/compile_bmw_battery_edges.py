"""
Compile the final curated battery supply chain edges from BMW Annual Reports 2021-2024.
Only includes findings actually verified in the PDF text.
"""

findings = []

# ============================================================
# BMW AR 2021
# ============================================================

findings.append({
    "year": 2021,
    "page": 23,
    "sentence": "We use only green power to produce batteries for our electrified vehicles – as do our battery cell suppliers.",
    "src": "BMW_Battery_Cell_Suppliers",
    "rel": "SUPPLIES",
    "dst": "BMW:Battery_Cell",
    "confidence": 0.8
})

findings.append({
    "year": 2021,
    "page": 55,
    "sentence": "In 2021, the BMW Group, together with the Ford Motor Company, Volta Energy Technologies and other investors, invested in Solid Power, Inc., one of the industry's leading developers of solid-state battery cells with high energy density.",
    "src": "BMW",
    "rel": "INVESTS_IN",
    "dst": "Solid_Power",
    "confidence": 0.95
})

findings.append({
    "year": 2021,
    "page": 55,
    "sentence": "the BMW Group intends to purchase solid-state battery cells from Solid Power, Inc.",
    "src": "Solid_Power",
    "rel": "SUPPLIES",
    "dst": "BMW:Battery_Cell",
    "confidence": 0.85
})

findings.append({
    "year": 2021,
    "page": 70,
    "sentence": "In collaboration with BASF and the ALBA Group, we are currently testing improved methods of recycling automotive plastics as part of a pilot project.",
    "src": "BASF",
    "rel": "PARTNERS_WITH",
    "dst": "BMW:Recycling",
    "confidence": 0.9
})

findings.append({
    "year": 2021,
    "page": 71,
    "sentence": "Lilac is pursuing the goal of extracting lithium from the brine of saltwater deposits using ion exchangers, deploying a far more eco-friendly method that conserves resources more effectively than conventional processes.",
    "src": "Lilac_Solutions",
    "rel": "SUPPLIES",
    "dst": "BMW:Lithium",
    "confidence": 0.7
})

findings.append({
    "year": 2021,
    "page": 78,
    "sentence": "The strategy allows us to fully document both the origin and the path of the lithium and cobalt we use, while creating transparency regarding mining methods at the same time.",
    "src": "BMW",
    "rel": "SOURCES_DIRECTLY",
    "dst": "Lithium",
    "confidence": 0.8
})

findings.append({
    "year": 2021,
    "page": 78,
    "sentence": "The strategy allows us to fully document both the origin and the path of the lithium and cobalt we use, while creating transparency regarding mining methods at the same time.",
    "src": "BMW",
    "rel": "SOURCES_DIRECTLY",
    "dst": "Cobalt",
    "confidence": 0.8
})

findings.append({
    "year": 2021,
    "page": 78,
    "sentence": "we have reduced the use of cobalt in the cathodes of our current generation of battery cells to less than 10 %.",
    "src": "BMW",
    "rel": "USES",
    "dst": "Cobalt:Cathode",
    "confidence": 0.95
})

findings.append({
    "year": 2021,
    "page": 78,
    "sentence": "The BMW Group even goes one step further in this regard because, as a precautionary measure, we are committed to not using minerals such as cobalt that are extracted from the deep sea.",
    "src": "BMW",
    "rel": "POLICY",
    "dst": "Cobalt:No_Deep_Sea_Mining",
    "confidence": 0.95
})

findings.append({
    "year": 2021,
    "page": 79,
    "sentence": "we entered into contractual agreements with battery cell manufacturers to use only energy generated from renewable sources to produce the current generation of battery cells.",
    "src": "BMW",
    "rel": "REQUIRES_GREEN_ENERGY",
    "dst": "BMW:Battery_Cell_Manufacturers",
    "confidence": 0.9
})

findings.append({
    "year": 2021,
    "page": 181,
    "sentence": "the BMW Group, together with the Ford Motor Company and Volta Energy Technologies, participated in an investment round relating to Solid Power, an industry-leading producer of solid-state batteries for electric vehicles.",
    "src": "BMW",
    "rel": "INVESTS_IN",
    "dst": "Solid_Power",
    "confidence": 0.95
})

findings.append({
    "year": 2021,
    "page": 181,
    "sentence": "some existing joint development partnerships with Solid Power have also been expanded with a view to securing the supply of solid-state batteries for future generations of electric vehicles.",
    "src": "Solid_Power",
    "rel": "PARTNERS_WITH",
    "dst": "BMW:Solid_State_Battery",
    "confidence": 0.9
})

findings.append({
    "year": 2021,
    "page": 245,
    "sentence": "Northvolt AB, Stockholm – – 3 [shareholding table]",
    "src": "BMW",
    "rel": "INVESTS_IN",
    "dst": "Northvolt",
    "confidence": 0.85
})

# ============================================================
# BMW AR 2022
# ============================================================

findings.append({
    "year": 2022,
    "page": 45,
    "sentence": "The sixth generation of our electric drivetrain features a completely new high-voltage storage concept, optimised cell design and improved cell chemistry.",
    "src": "BMW",
    "rel": "DEVELOPS",
    "dst": "Gen6_Battery_Cell",
    "confidence": 0.9
})

findings.append({
    "year": 2022,
    "page": 92,
    "sentence": "The sixth generation of our lithium-ion cells represents a giant technological leap forward compared with the previous generation, effectively increasing energy density by more than 20 %, charging speed by up to 30 % and range by around 30 %.",
    "src": "BMW",
    "rel": "DEVELOPS",
    "dst": "Gen6_Lithium_Ion_Cell",
    "confidence": 0.95
})

findings.append({
    "year": 2022,
    "page": 92,
    "sentence": "The carbon emissions generated by cell production will be reduced by up to 60 %.",
    "src": "BMW",
    "rel": "TARGETS",
    "dst": "Cell_Production:CO2_Reduction_60pct",
    "confidence": 0.9
})

findings.append({
    "year": 2022,
    "page": 104,
    "sentence": "The Cell Manufacturing Competence Centre (CMCC) in Parsdorf near Munich, which went into operation in 2022, is making a vital contribution to the next generation of e-drive systems.",
    "src": "BMW",
    "rel": "OPERATES",
    "dst": "CMCC_Parsdorf:Cell_Manufacturing",
    "confidence": 0.95
})

findings.append({
    "year": 2022,
    "page": 107,
    "sentence": "Lilac Solutions is pursuing the goal of extracting lithium from the brine of saltwater deposits using ion exchangers in a far more eco- and resource-friendly way than previously possible.",
    "src": "Lilac_Solutions",
    "rel": "SUPPLIES",
    "dst": "BMW:Lithium",
    "confidence": 0.7
})

findings.append({
    "year": 2022,
    "page": 107,
    "sentence": "During the period under report, we continued to invest in the resource-conserving extraction of lithium by acquiring a stake in the company Mangrove Lithium.",
    "src": "BMW",
    "rel": "INVESTS_IN",
    "dst": "Mangrove_Lithium",
    "confidence": 0.9
})

findings.append({
    "year": 2022,
    "page": 107,
    "sentence": "Its innovative technology makes it possible to refine and process both virgin and recycled lithium directly into battery-grade lithium using a special procedure.",
    "src": "Mangrove_Lithium",
    "rel": "PRODUCES",
    "dst": "Battery_Grade_Lithium",
    "confidence": 0.85
})

findings.append({
    "year": 2022,
    "page": 110,
    "sentence": "With the help of measures such as the direct purchase of lithium and cobalt, the BMW Group is making itself technologically and regionally less dependent on individual deposits and suppliers.",
    "src": "BMW",
    "rel": "SOURCES_DIRECTLY",
    "dst": "Lithium",
    "confidence": 0.95
})

findings.append({
    "year": 2022,
    "page": 110,
    "sentence": "With the help of measures such as the direct purchase of lithium and cobalt, the BMW Group is making itself technologically and regionally less dependent on individual deposits and suppliers.",
    "src": "BMW",
    "rel": "SOURCES_DIRECTLY",
    "dst": "Cobalt",
    "confidence": 0.95
})

findings.append({
    "year": 2022,
    "page": 186,
    "sentence": "The BMW Group holds shares in Solid Power, an industry-leading manufacturer of solid-state batteries for electric vehicles.",
    "src": "BMW",
    "rel": "INVESTS_IN",
    "dst": "Solid_Power",
    "confidence": 0.95
})

findings.append({
    "year": 2022,
    "page": 186,
    "sentence": "Joint development partnerships are in place with Solid Power with a view to securing the supply of solid-state batteries for future generations of electric vehicles.",
    "src": "Solid_Power",
    "rel": "PARTNERS_WITH",
    "dst": "BMW:Solid_State_Battery",
    "confidence": 0.9
})

findings.append({
    "year": 2022,
    "page": 244,
    "sentence": "Northvolt AB, Stockholm [shareholding table]",
    "src": "BMW",
    "rel": "INVESTS_IN",
    "dst": "Northvolt",
    "confidence": 0.85
})

# ============================================================
# BMW AR 2023
# ============================================================

findings.append({
    "year": 2023,
    "page": 25,
    "sentence": "The Gen6 batteries for our plants in Bavaria will be supplied from our planned new location in Irlbach-Strasskirchen from 2026 onwards.",
    "src": "BMW:Irlbach_Strasskirchen",
    "rel": "PRODUCES",
    "dst": "Gen6_Battery",
    "confidence": 0.95
})

findings.append({
    "year": 2023,
    "page": 25,
    "sentence": "we have created the necessary conditions for this with the Battery Cell Competence Centre in Munich and the Cell Manufacturing Competence Centre in Parsdorf, Bavaria, enabling us to span all value creation processes involved in cell production.",
    "src": "BMW",
    "rel": "OPERATES",
    "dst": "CMCC_Parsdorf:Cell_Manufacturing",
    "confidence": 0.95
})

findings.append({
    "year": 2023,
    "page": 46,
    "sentence": "Examples include our Cell Manufacturing Competence Centre (CMCC) in Parsdorf near Munich, where our expertise in designing and manufacturing solid-state cells is continuously expanded through a dedicated prototype line, bringing significant value to our partnership with Solid Power.",
    "src": "Solid_Power",
    "rel": "PARTNERS_WITH",
    "dst": "BMW:CMCC_Parsdorf:Solid_State_Cell",
    "confidence": 0.9
})

findings.append({
    "year": 2023,
    "page": 46,
    "sentence": "For the next generation of high-voltage batteries, new assembly sites will be established in Debrecen (Hungary), San Luis Potosi (Mexico), Woodruff near Spartanburg (USA) and in Germany at the planned new facility in Irlbach-Strasskirchen.",
    "src": "BMW",
    "rel": "BUILDS_PLANT",
    "dst": "Gen6_Battery_Assembly:Debrecen_SLP_Woodruff_Irlbach",
    "confidence": 0.95
})

findings.append({
    "year": 2023,
    "page": 92,
    "sentence": "The carbon emissions generated by cell production will be reduced by up to 60%.",
    "src": "BMW",
    "rel": "TARGETS",
    "dst": "Cell_Production:CO2_Reduction_60pct",
    "confidence": 0.9
})

findings.append({
    "year": 2023,
    "page": 104,
    "sentence": "Production facilities for the sixth generation of the high-voltage battery are being established in Debrecen (Hungary), Woodruff near Spartanburg (USA), San Luis Potosi (Mexico) and Shenyang (China).",
    "src": "BMW",
    "rel": "BUILDS_PLANT",
    "dst": "Gen6_HV_Battery:Debrecen_Woodruff_SLP_Shenyang",
    "confidence": 0.95
})

findings.append({
    "year": 2023,
    "page": 104,
    "sentence": "The BMW Group will take the findings of this pilot scheme and apply them in close collaboration with its mass production partners for battery cells at a later stage.",
    "src": "BMW:CMCC_Parsdorf",
    "rel": "COLLABORATES_WITH",
    "dst": "Battery_Cell_Production_Partners",
    "confidence": 0.8
})

findings.append({
    "year": 2023,
    "page": 112,
    "sentence": "Measures taken by the BMW Group to increase supply security and promote the procurement of raw materials from responsible sources include, among other things, sourcing lithium and cobalt directly.",
    "src": "BMW",
    "rel": "SOURCES_DIRECTLY",
    "dst": "Lithium",
    "confidence": 0.95
})

findings.append({
    "year": 2023,
    "page": 112,
    "sentence": "Measures taken by the BMW Group to increase supply security and promote the procurement of raw materials from responsible sources include, among other things, sourcing lithium and cobalt directly.",
    "src": "BMW",
    "rel": "SOURCES_DIRECTLY",
    "dst": "Cobalt",
    "confidence": 0.95
})

findings.append({
    "year": 2023,
    "page": 112,
    "sentence": "Secondary raw materials will be increasingly deployed to cover the BMW Group's cobalt, lithium and nickel requirements, together with purchased green electricity for the latest generation of battery cells.",
    "src": "BMW",
    "rel": "USES_SECONDARY",
    "dst": "Cobalt",
    "confidence": 0.85
})

findings.append({
    "year": 2023,
    "page": 112,
    "sentence": "Secondary raw materials will be increasingly deployed to cover the BMW Group's cobalt, lithium and nickel requirements, together with purchased green electricity for the latest generation of battery cells.",
    "src": "BMW",
    "rel": "USES_SECONDARY",
    "dst": "Lithium",
    "confidence": 0.85
})

findings.append({
    "year": 2023,
    "page": 112,
    "sentence": "Secondary raw materials will be increasingly deployed to cover the BMW Group's cobalt, lithium and nickel requirements, together with purchased green electricity for the latest generation of battery cells.",
    "src": "BMW",
    "rel": "USES_SECONDARY",
    "dst": "Nickel",
    "confidence": 0.85
})

findings.append({
    "year": 2023,
    "page": 185,
    "sentence": "The BMW Group holds shares in Solid Power, an industry-leading manufacturer of solid-state batteries for electric vehicles.",
    "src": "BMW",
    "rel": "INVESTS_IN",
    "dst": "Solid_Power",
    "confidence": 0.95
})

findings.append({
    "year": 2023,
    "page": 185,
    "sentence": "Joint development partnerships are in place with Solid Power with a view to securing the supply of solid-state batteries for future generations of electric vehicles.",
    "src": "Solid_Power",
    "rel": "PARTNERS_WITH",
    "dst": "BMW:Solid_State_Battery",
    "confidence": 0.9
})

findings.append({
    "year": 2023,
    "page": 241,
    "sentence": "Northvolt AB, Stockholm [shareholding table]",
    "src": "BMW",
    "rel": "INVESTS_IN",
    "dst": "Northvolt",
    "confidence": 0.85
})

# ============================================================
# BMW AR 2024
# ============================================================

findings.append({
    "year": 2024,
    "page": 28,
    "sentence": "Assembly of the sixth generation (Gen6) of our high-voltage batteries for the NEUE KLASSE will take place at five new cutting-edge production facilities being built close to our vehicle plants: in Irlbach-Strasskirchen (Lower Bavaria), Debrecen (Hungary), Shenyang (China), San Luis Potosi (Mexico) and Woodruff, near Spartanburg (USA).",
    "src": "BMW",
    "rel": "BUILDS_PLANT",
    "dst": "Gen6_HV_Battery:Irlbach_Debrecen_Shenyang_SLP_Woodruff",
    "confidence": 0.95
})

findings.append({
    "year": 2024,
    "page": 28,
    "sentence": "With the sixth generation of BMW eDrive technology, your Company has achieved a quantum leap in technology, offering up to 30% faster charging and around 30% more range.",
    "src": "BMW",
    "rel": "DEVELOPS",
    "dst": "Gen6_eDrive:Battery_Cell",
    "confidence": 0.95
})

findings.append({
    "year": 2024,
    "page": 28,
    "sentence": "For the first time, the Gen6 high-voltage batteries also feature the latest 800-volt technology and enable bidirectional charging.",
    "src": "BMW",
    "rel": "DEVELOPS",
    "dst": "Gen6_Battery:800V_Bidirectional",
    "confidence": 0.9
})

findings.append({
    "year": 2024,
    "page": 48,
    "sentence": "the development partnership between the BMW Group and Solid Power offers benefits to both companies.",
    "src": "Solid_Power",
    "rel": "PARTNERS_WITH",
    "dst": "BMW:Battery_Development",
    "confidence": 0.9
})

findings.append({
    "year": 2024,
    "page": 56,
    "sentence": "The energy density in the sixth generation of our lithium-ion cells is around 20% higher than in previous generations, with charging speed improved by up to 30% and range by around 30%.",
    "src": "BMW",
    "rel": "DEVELOPS",
    "dst": "Gen6_Lithium_Ion_Cell",
    "confidence": 0.95
})

findings.append({
    "year": 2024,
    "page": 60,
    "sentence": "The new Cell Manufacturing Competence Centre (CMCC) in Parsdorf near Munich plays a key role for the BMW Group.",
    "src": "BMW",
    "rel": "OPERATES",
    "dst": "CMCC_Parsdorf:Cell_Manufacturing",
    "confidence": 0.95
})

findings.append({
    "year": 2024,
    "page": 64,
    "sentence": "The BMW Group relies on its close cooperation with partners in the supply chain and can also secure raw materials such as lithium and cobalt itself directly, where required, in order to increase its security of supply.",
    "src": "BMW",
    "rel": "SOURCES_DIRECTLY",
    "dst": "Lithium",
    "confidence": 0.95
})

findings.append({
    "year": 2024,
    "page": 64,
    "sentence": "The BMW Group relies on its close cooperation with partners in the supply chain and can also secure raw materials such as lithium and cobalt itself directly, where required, in order to increase its security of supply.",
    "src": "BMW",
    "rel": "SOURCES_DIRECTLY",
    "dst": "Cobalt",
    "confidence": 0.95
})

findings.append({
    "year": 2024,
    "page": 64,
    "sentence": "The use of secondary materials and the use of electricity from renewable sources in particular in battery cell production significantly reduce the Group's carbon equivalent footprint.",
    "src": "BMW",
    "rel": "REQUIRES_GREEN_ENERGY",
    "dst": "Battery_Cell_Production",
    "confidence": 0.85
})

findings.append({
    "year": 2024,
    "page": 144,
    "sentence": "Beginning in 2025 for example, requirements for the proportionate use of secondary materials for battery cell materials such as cobalt, lithium and nickel will be imposed as soon as the contract is awarded.",
    "src": "BMW",
    "rel": "REQUIRES_SECONDARY",
    "dst": "Battery_Cell_Suppliers:Cobalt_Lithium_Nickel",
    "confidence": 0.9
})

findings.append({
    "year": 2024,
    "page": 145,
    "sentence": "Lilac Solutions is developing an ion-exchange lithium extraction technology with enhanced recovery rates, reduced impurities, and lower acid consumption.",
    "src": "Lilac_Solutions",
    "rel": "SUPPLIES",
    "dst": "BMW:Lithium",
    "confidence": 0.7
})

findings.append({
    "year": 2024,
    "page": 145,
    "sentence": "Cyclic Materials is developing a recycling process that reintegrates rare earth elements (REEs) into manufacturing.",
    "src": "Cyclic_Materials",
    "rel": "RECYCLES",
    "dst": "BMW:Rare_Earth_Elements",
    "confidence": 0.7
})

findings.append({
    "year": 2024,
    "page": 301,
    "sentence": "The BMW Group holds shares in Solid Power, an industry-leading manufacturer of solid-state batteries for electric vehicles.",
    "src": "BMW",
    "rel": "INVESTS_IN",
    "dst": "Solid_Power",
    "confidence": 0.95
})

findings.append({
    "year": 2024,
    "page": 301,
    "sentence": "Joint development partnerships are in place with Solid Power with a view to securing the supply of solid-state batteries for future generations of electric vehicles.",
    "src": "Solid_Power",
    "rel": "PARTNERS_WITH",
    "dst": "BMW:Solid_State_Battery",
    "confidence": 0.9
})

findings.append({
    "year": 2024,
    "page": 301,
    "sentence": "The BMW Group has significant influence as it is represented on Solid Power's Board of Directors.",
    "src": "BMW",
    "rel": "BOARD_SEAT",
    "dst": "Solid_Power",
    "confidence": 0.95
})

findings.append({
    "year": 2024,
    "page": 359,
    "sentence": "Northvolt AB, Stockholm [shareholding table]",
    "src": "BMW",
    "rel": "INVESTS_IN",
    "dst": "Northvolt",
    "confidence": 0.85
})


# ============================================================
# Print all findings
# ============================================================
print("=" * 100)
print("BMW ANNUAL REPORTS 2021-2024: BATTERY SUPPLY CHAIN KNOWLEDGE GRAPH EDGES")
print("=" * 100)

current_year = None
for f in findings:
    if f["year"] != current_year:
        current_year = f["year"]
        print(f"\n{'─' * 100}")
        print(f"  BMW ANNUAL REPORT {current_year}")
        print(f"{'─' * 100}")

    sent = f["sentence"][:300]
    print(f"\n[BMW AR {f['year']} p.{f['page']}] {sent}")
    print(f"EDGE: {f['src']} --{f['rel']}--> {f['dst']} | confidence: {f['confidence']}")

# Summary
print(f"\n{'=' * 100}")
print("SUMMARY")
print(f"{'=' * 100}")

from collections import Counter
year_counts = Counter(f["year"] for f in findings)
rel_counts = Counter(f["rel"] for f in findings)
entity_counts = Counter()
for f in findings:
    entity_counts[f["src"]] += 1
    entity_counts[f["dst"]] += 1

print(f"\nTotal edges: {len(findings)}")
print(f"\nBy year:")
for y in sorted(year_counts):
    print(f"  {y}: {year_counts[y]} edges")
print(f"\nBy relationship type:")
for rel, count in rel_counts.most_common():
    print(f"  {rel}: {count}")
print(f"\nTop entities:")
for ent, count in entity_counts.most_common(20):
    print(f"  {ent}: {count}")

print(f"\n{'=' * 100}")
print("NOTE: BMW Annual Reports do NOT explicitly name CATL, Samsung SDI, EVE Energy,")
print("LG Energy, BYD, Gotion, Umicore, or Glencore as battery cell suppliers.")
print("They refer generically to 'battery cell manufacturers/suppliers' without naming them.")
print("Named battery-related entities found: Solid Power, Northvolt, BASF (plastics recycling),")
print("Lilac Solutions (lithium extraction), Mangrove Lithium, Cyclic Materials (REE recycling).")
print(f"{'=' * 100}")
