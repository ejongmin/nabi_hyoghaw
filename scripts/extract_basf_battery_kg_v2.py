"""
Extract battery supply chain relationships from BASF Annual Reports (2021-2024)
for Knowledge Graph construction. V2: focused output, reduced noise.
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

# High-value search terms (battery supply chain specific)
BATTERY_TERMS = [
    "cathode active material", "CAM ", " CAM,", " CAM.", "pCAM", "precursor cathode",
    "battery recycling", "battery material", "battery cell",
    "Schwarzheide", "TODA", "Glencore", "Huayou", "CATL",
    "BMW", "Volkswagen", "Toyota", "Panasonic", "LG Chem", "LG Energy",
    "Samsung SDI", "Umicore",
    "Harjavalta", "cell manufacturer", "cell production",
    "BASF Toda", "BASF Shanshan", "Nornickel",
    "Eramet", "black mass",
    "cathode material", "anode material",
    "gigafactory", "EV battery",
    "joint venture", "battery",
    "supply agreement", "offtake",
    "SVOLT", "Porsche",
]

# Build pattern that requires at least one battery-specific term
BATTERY_PATTERN = re.compile(
    "|".join(re.escape(t) for t in BATTERY_TERMS),
    re.IGNORECASE
)

# Entity patterns for edge extraction
PARTNER_PATTERNS = {
    r"\bTODA\b": "TODA_KOGYO",
    r"\bGlencore\b": "Glencore",
    r"\bHuayou\b": "Huayou_Cobalt",
    r"\bCATL\b": "CATL",
    r"\bBMW\b": "BMW",
    r"\bVolkswagen\b": "Volkswagen",
    r"\bToyota\b": "Toyota",
    r"\bPanasonic\b": "Panasonic",
    r"\bLG\s*Chem\b": "LG_Chem",
    r"\bLG\s*Energy\b": "LG_Energy_Solution",
    r"\bSamsung\s*SDI\b": "Samsung_SDI",
    r"\bUmicore\b": "Umicore",
    r"\bNornickel\b": "Nornickel",
    r"\bNorilsk\b": "Nornickel",
    r"\bEramet\b": "Eramet",
    r"\bBASF\s*Shanshan\b": "BASF_Shanshan",
    r"\bBASF\s*Toda\b": "BASF_TODA",
    r"\bSVOLT\b": "SVOLT",
    r"\bPorsche\b": "Porsche",
    r"\bTesla\b": "Tesla",
    r"\bFord\b": "Ford",
    r"\bStellantis\b": "Stellantis",
    r"\bPrime\s*Planet\b": "Prime_Planet_Energy",
}

LOCATION_PATTERNS = {
    r"\bSchwarzhei?de\b": "Schwarzheide_Plant",
    r"\bHarjavalta\b": "Harjavalta_Plant",
    r"\bOnoda\b": "Onoda_Plant",
    r"\bNumazu\b": "Numazu_Plant",
    r"\bSparks\b": "Sparks_Plant",
}

MATERIAL_PATTERNS = {
    r"cathode\s+active\s+material": "CAM",
    r"\bCAM\b": "CAM",
    r"\bpCAM\b": "pCAM",
    r"precursor\s+cathode": "pCAM",
    r"\bcobalt\b": "Cobalt",
    r"\bnickel\b": "Nickel",
    r"\blithium\b": "Lithium",
    r"\bmanganese\b": "Manganese",
    r"black\s+mass": "Black_Mass",
    r"cathode\s+material": "Cathode_Materials",
}

RELATION_PATTERNS = [
    (r"joint\s+venture", "JOINT_VENTURE"),
    (r"partnership", "PARTNERS_WITH"),
    (r"agreement|contract", "HAS_AGREEMENT"),
    (r"collaborat", "COLLABORATES_WITH"),
    (r"suppli|supply", "SUPPLIES"),
    (r"offtake", "HAS_OFFTAKE"),
    (r"recycl", "RECYCLES"),
    (r"acqui|formation|formed", "ACQUIRED"),
    (r"invest", "INVESTED_IN"),
    (r"produc|manufactur|operat", "PRODUCES_AT"),
    (r"customer", "SUPPLIES_TO"),
    (r"build|construct", "BUILDING"),
]


def find_matches(pattern_dict, text):
    found = []
    for pat, label in pattern_dict.items():
        if re.search(pat, text, re.IGNORECASE):
            found.append(label)
    return list(set(found))


def find_relation(text):
    for pat, rel in RELATION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return rel
    return "RELATED_TO"


def is_battery_relevant(sentence):
    """Check if sentence is actually about battery supply chain, not just mentions a keyword in passing."""
    s_lower = sentence.lower()
    battery_context = any(w in s_lower for w in [
        "battery", "cathode", "anode", "cam ", "pcam", "cell",
        "recycl", "electr", "ev ", "lithium", "cobalt", "nickel",
        "schwarzheide", "harjavalta", "shanshan", "toda",
        "catl", "svolt", "gigafact",
    ])
    return battery_context


def split_sentences(text):
    """Split text into sentences, handling PDF quirks."""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [p.strip() for p in parts if len(p.strip()) > 30]


def process_pdf(year, filepath):
    results = []
    doc = fitz.open(filepath)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if not text.strip():
            continue

        if not BATTERY_PATTERN.search(text):
            continue

        sentences = split_sentences(text)
        for sent in sentences:
            if not BATTERY_PATTERN.search(sent):
                continue

            if not is_battery_relevant(sent):
                continue

            partners = find_matches(PARTNER_PATTERNS, sent)
            locations = find_matches(LOCATION_PATTERNS, sent)
            materials = find_matches(MATERIAL_PATTERNS, sent)
            rel = find_relation(sent)

            display = sent[:300].strip()

            edges = []

            # Partner edges
            for p in partners:
                conf = 0.85 if rel != "RELATED_TO" else 0.6
                edges.append(("BASF", rel, p, conf))

            # Location edges
            for loc in locations:
                edges.append(("BASF", "OPERATES_PLANT", loc, 0.8))
                for mat in materials:
                    edges.append((loc, "PRODUCES", mat, 0.75))

            # Material edges (when no location/partner found)
            if not partners and not locations and materials:
                for mat in materials:
                    edges.append(("BASF", "PRODUCES", mat, 0.5))

            # Partner + material combo
            for p in partners:
                for mat in materials:
                    conf = 0.75 if rel != "RELATED_TO" else 0.5
                    edges.append((p, "INVOLVED_IN", mat, conf))

            # If no edges at all but sentence is battery-relevant
            if not edges:
                edges.append(("BASF", "MENTIONS", "Battery_Supply_Chain", 0.3))

            for src, r, dst, conf in edges:
                results.append({
                    "year": year,
                    "page": page_num + 1,
                    "sentence": display,
                    "src": src,
                    "rel": r,
                    "dst": dst,
                    "confidence": conf,
                })

    doc.close()
    return results


def deduplicate(results):
    seen = set()
    out = []
    for r in results:
        # Deduplicate on (year, truncated sentence, edge)
        key = (r["year"], r["sentence"][:80], r["src"], r["rel"], r["dst"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main():
    all_results = []

    for year, filepath in sorted(PDF_FILES.items()):
        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found, skipping.")
            continue
        print(f"Processing BASF AR {year}...", flush=True)
        results = process_pdf(year, filepath)
        all_results.extend(results)
        print(f"  Raw matches: {len(results)}")

    all_results = deduplicate(all_results)

    # Filter out low-value MENTIONS
    structured = [r for r in all_results if r["rel"] != "MENTIONS"]
    mentions = [r for r in all_results if r["rel"] == "MENTIONS"]

    print(f"\nTotal structured edges: {len(structured)}")
    print(f"Total context mentions: {len(mentions)}")

    # Output file
    outpath = os.path.join(BASE_DIR, "data", "processed", "basf_battery_kg_extractions.txt")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(outpath, "w", encoding="utf-8") as f:
        for year in sorted(PDF_FILES.keys()):
            yr = [r for r in structured if r["year"] == year]
            if not yr:
                continue
            yr.sort(key=lambda x: x["page"])
            f.write(f"\n{'='*80}\n")
            f.write(f"BASF ANNUAL REPORT {year}\n")
            f.write(f"{'='*80}\n\n")

            for r in yr:
                line1 = f'[BASF AR {r["year"]} p.{r["page"]}] {r["sentence"]}'
                line2 = f'EDGE: {r["src"]} --{r["rel"]}--> {r["dst"]} | confidence: {r["confidence"]}'
                f.write(line1 + "\n")
                f.write(line2 + "\n\n")

        # Also write mentions section
        f.write(f"\n{'='*80}\n")
        f.write(f"CONTEXTUAL MENTIONS (lower confidence)\n")
        f.write(f"{'='*80}\n\n")
        for r in mentions:
            line1 = f'[BASF AR {r["year"]} p.{r["page"]}] {r["sentence"]}'
            line2 = f'EDGE: {r["src"]} --{r["rel"]}--> {r["dst"]} | confidence: {r["confidence"]}'
            f.write(line1 + "\n")
            f.write(line2 + "\n\n")

    print(f"\nResults written to: {outpath}")

    # Print to stdout as well (structured edges only)
    print(f"\n{'='*80}")
    print("STRUCTURED EDGE EXTRACTIONS")
    print(f"{'='*80}\n")

    for year in sorted(PDF_FILES.keys()):
        yr = [r for r in structured if r["year"] == year]
        if not yr:
            continue
        yr.sort(key=lambda x: x["page"])
        print(f"\n--- BASF AR {year} ({len(yr)} edges) ---\n")
        for r in yr:
            print(f'[BASF AR {r["year"]} p.{r["page"]}] {r["sentence"]}')
            print(f'EDGE: {r["src"]} --{r["rel"]}--> {r["dst"]} | confidence: {r["confidence"]}')
            print()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    entities = set()
    edge_types = {}
    for r in structured:
        entities.add(r["src"])
        entities.add(r["dst"])
        edge_types[r["rel"]] = edge_types.get(r["rel"], 0) + 1

    print(f"Total structured edges: {len(structured)}")
    print(f"Unique entities: {len(entities)}")
    print(f"\nEdge types:")
    for et, c in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"  {et}: {c}")
    print(f"\nEntities found:")
    for e in sorted(entities):
        print(f"  {e}")


if __name__ == "__main__":
    main()
