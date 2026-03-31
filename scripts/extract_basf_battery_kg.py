"""
Extract battery supply chain relationships from BASF Annual Reports (2021-2024)
for Knowledge Graph construction.
"""
import fitz
import re
import os

BASE_DIR = r"C:\Users\john9\nabi_hyoghaw"
PDF_FILES = {
    2021: os.path.join(BASE_DIR, "data", "raw", "filings", "basf_ar_2021.pdf"),
    2022: os.path.join(BASE_DIR, "data", "raw", "filings", "basf_ar_2022.pdf"),
    2023: os.path.join(BASE_DIR, "data", "raw", "filings", "basf_ar_2023.pdf"),
    2024: os.path.join(BASE_DIR, "data", "raw", "filings", "basf_ar_2024.pdf"),
}

# Keywords for section detection (TOC / section headers)
SECTION_KEYWORDS = [
    r"surface\s+technolog",
    r"battery\s+material",
    r"catalyst",
    r"materials\s+segment",
    r"procurement",
    r"supply\s+chain",
    r"industrial\s+solutions",
    r"coatings",
    r"chemicals\s+segment",
    r"segment\s+data",
    r"automotive",
]

# Entity / topic keywords to search for in text
SEARCH_TERMS = [
    "cathode active material", "CAM", "pCAM", "precursor cathode",
    "battery recycling", "battery material", "battery cell",
    "Schwarzheide", "TODA", "Glencore", "Huayou", "CATL",
    "BMW", "Volkswagen", "Toyota", "Panasonic", "LG Chem", "LG Energy",
    "Samsung SDI", "Samsung", "Umicore",
    "cobalt", "nickel", "lithium", "manganese",
    "Harjavalta", "cell manufacturer", "cell production",
    "BASF Toda", "BASF Shanshan", "Norilsk", "Nornickel",
    "Eramet", "recycling", "black mass",
    "electrode", "electrolyte", "anode", "cathode",
    "gigafactory", "EV battery", "electric vehicle",
    "joint venture", "partnership", "agreement", "collaboration",
    "supply agreement", "offtake", "long-term supply",
]

# Compile a combined regex pattern (case-insensitive)
SEARCH_PATTERN = re.compile(
    "|".join(re.escape(t) for t in SEARCH_TERMS),
    re.IGNORECASE
)


def extract_toc_pages(doc):
    """Try to find TOC and identify relevant section page numbers."""
    toc = doc.get_toc()
    relevant_pages = set()
    if toc:
        for level, title, page in toc:
            title_lower = title.lower()
            for kw in SECTION_KEYWORDS:
                if re.search(kw, title_lower):
                    # Add the page and surrounding pages
                    for p in range(max(0, page - 1), min(len(doc), page + 15)):
                        relevant_pages.add(p)
                    break
    return relevant_pages, toc


def split_into_sentences(text):
    """Split text into rough sentences."""
    # Split on period followed by space and uppercase, or newlines that look like sentence breaks
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return sentences


def find_edge(sentence, year):
    """Attempt to extract a knowledge graph edge from a sentence."""
    edges = []
    s_lower = sentence.lower()

    # Pattern: BASF + partner entity
    partners = {
        "TODA": "TODA KOGYO",
        "Glencore": "Glencore",
        "Huayou": "Huayou Cobalt",
        "CATL": "CATL",
        "BMW": "BMW",
        "Volkswagen": "Volkswagen",
        "Toyota": "Toyota",
        "Panasonic": "Panasonic",
        "LG Chem": "LG Chem",
        "LG Energy": "LG Energy Solution",
        "Samsung SDI": "Samsung SDI",
        "Samsung": "Samsung SDI",
        "Umicore": "Umicore",
        "Norilsk": "Nornickel",
        "Nornickel": "Nornickel",
        "Eramet": "Eramet",
        "BASF Shanshan": "BASF Shanshan",
        "BASF Toda": "BASF TODA",
    }

    # Location-based edges
    locations = {
        "Schwarzheide": "BASF_Schwarzheide_Plant",
        "Harjavalta": "BASF_Harjavalta_Plant",
    }

    # Material edges
    materials = {
        "cathode active material": "Cathode_Active_Materials",
        "CAM": "Cathode_Active_Materials",
        "pCAM": "Precursor_CAM",
        "precursor cathode": "Precursor_CAM",
        "cobalt": "Cobalt",
        "nickel": "Nickel",
        "lithium": "Lithium",
        "manganese": "Manganese",
        "black mass": "Battery_Recycling_Feedstock",
    }

    # Relationship keywords
    rel_keywords = {
        "joint venture": "JOINT_VENTURE_WITH",
        "partnership": "PARTNERS_WITH",
        "agreement": "HAS_AGREEMENT_WITH",
        "collaboration": "COLLABORATES_WITH",
        "supply": "SUPPLIES_TO",
        "offtake": "HAS_OFFTAKE_WITH",
        "recycl": "RECYCLES_WITH",
        "acqui": "ACQUIRED",
        "invest": "INVESTED_IN",
        "produc": "PRODUCES",
        "manufactur": "MANUFACTURES",
        "operat": "OPERATES",
        "locat": "LOCATED_AT",
        "site": "LOCATED_AT",
        "plant": "OPERATES_PLANT",
        "customer": "SUPPLIES_TO",
    }

    found_partners = []
    found_locations = []
    found_materials = []
    found_rels = []

    for key, val in partners.items():
        if key.lower() in s_lower:
            found_partners.append(val)

    for key, val in locations.items():
        if key.lower() in s_lower:
            found_locations.append(val)

    for key, val in materials.items():
        if key.lower() in s_lower:
            found_materials.append(val)

    for key, val in rel_keywords.items():
        if key.lower() in s_lower:
            found_rels.append(val)

    # Determine best relationship
    rel = found_rels[0] if found_rels else "RELATED_TO"

    # Build edges
    for partner in found_partners:
        conf = 0.8 if len(found_rels) > 0 else 0.5
        edges.append(("BASF", rel, partner, conf))

    for loc in found_locations:
        conf = 0.8
        edges.append(("BASF", "OPERATES_PLANT", loc, conf))
        for mat in found_materials:
            edges.append((loc, "PRODUCES", mat, 0.7))

    for mat in found_materials:
        if not found_locations:
            edges.append(("BASF", "PRODUCES", mat, 0.6))

    # If we found a partner + material, link them
    for partner in found_partners:
        for mat in found_materials:
            conf = 0.7 if len(found_rels) > 0 else 0.4
            edges.append(("BASF", rel, f"{partner}|{mat}", conf))

    return edges


def process_pdf(year, filepath):
    """Process a single PDF and extract relevant sentences + edges."""
    results = []
    doc = fitz.open(filepath)
    total_pages = len(doc)

    # Step 1: Get TOC-based relevant pages
    relevant_pages, toc = extract_toc_pages(doc)

    # Step 2: If TOC didn't give us pages, scan all pages
    if not relevant_pages:
        relevant_pages = set(range(total_pages))

    # Also always do a full scan for our keywords on ALL pages
    # (TOC-based filtering is just for optimization hints)
    pages_to_scan = set(range(total_pages))

    for page_num in sorted(pages_to_scan):
        page = doc[page_num]
        text = page.get_text()
        if not text.strip():
            continue

        # Check if this page has any of our search terms
        if not SEARCH_PATTERN.search(text):
            continue

        # Split into sentences and check each
        sentences = split_into_sentences(text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue

            if SEARCH_PATTERN.search(sent):
                # Truncate to 300 chars
                display = sent[:300].replace('\n', ' ').strip()
                # Clean up excessive whitespace
                display = re.sub(r'\s+', ' ', display)

                edges = find_edge(sent, year)

                if edges:
                    for src, rel, dst, conf in edges:
                        results.append({
                            "year": year,
                            "page": page_num + 1,  # 1-indexed
                            "sentence": display,
                            "src": src,
                            "rel": rel,
                            "dst": dst,
                            "confidence": conf,
                        })
                else:
                    # Still record the finding even without a structured edge
                    results.append({
                        "year": year,
                        "page": page_num + 1,
                        "sentence": display,
                        "src": "BASF",
                        "rel": "MENTIONS",
                        "dst": "battery_supply_chain",
                        "confidence": 0.3,
                    })

    doc.close()
    return results


def deduplicate_results(results):
    """Remove duplicate sentences (keep first occurrence)."""
    seen = set()
    deduped = []
    for r in results:
        key = (r["year"], r["sentence"][:100], r["src"], r["rel"], r["dst"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def main():
    all_results = []

    for year, filepath in sorted(PDF_FILES.items()):
        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found, skipping.")
            continue
        print(f"\n{'='*80}")
        print(f"Processing BASF Annual Report {year}...")
        print(f"{'='*80}")

        results = process_pdf(year, filepath)
        all_results.extend(results)
        print(f"  Found {len(results)} raw matches for {year}")

    # Deduplicate
    all_results = deduplicate_results(all_results)
    print(f"\nTotal unique findings: {len(all_results)}")

    # Print results
    print(f"\n{'='*80}")
    print("EXTRACTED FINDINGS")
    print(f"{'='*80}\n")

    # Group by year
    for year in sorted(PDF_FILES.keys()):
        year_results = [r for r in all_results if r["year"] == year]
        if not year_results:
            continue

        print(f"\n--- BASF AR {year} ({len(year_results)} findings) ---\n")

        # Sort by page
        year_results.sort(key=lambda x: x["page"])

        # Filter: prioritize actual edges over MENTIONS
        edges_only = [r for r in year_results if r["rel"] != "MENTIONS"]
        mentions_only = [r for r in year_results if r["rel"] == "MENTIONS"]

        # Print edges first
        for r in edges_only:
            print(f'[BASF AR {r["year"]} p.{r["page"]}] {r["sentence"]}')
            print(f'EDGE: {r["src"]} --{r["rel"]}--> {r["dst"]} | confidence: {r["confidence"]}')
            print()

        # Print a limited number of MENTIONS (context sentences)
        if mentions_only:
            print(f"  ... plus {len(mentions_only)} contextual mentions (showing up to 30):\n")
            for r in mentions_only[:30]:
                print(f'[BASF AR {r["year"]} p.{r["page"]}] {r["sentence"]}')
                print(f'EDGE: {r["src"]} --{r["rel"]}--> {r["dst"]} | confidence: {r["confidence"]}')
                print()

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    edge_types = {}
    entity_pairs = set()
    for r in all_results:
        if r["rel"] != "MENTIONS":
            edge_types[r["rel"]] = edge_types.get(r["rel"], 0) + 1
            entity_pairs.add((r["src"], r["dst"]))

    print(f"Total findings: {len(all_results)}")
    print(f"Unique entity pairs: {len(entity_pairs)}")
    print(f"Edge type distribution:")
    for et, count in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"  {et}: {count}")


if __name__ == "__main__":
    main()
