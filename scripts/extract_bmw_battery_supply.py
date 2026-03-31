"""
Extract battery supply chain mentions from BMW Annual Reports (2021-2024).
Uses PyMuPDF (fitz) to read PDFs, searches for relevant sections and keywords.
"""
import fitz
import re
import os

BASE = r"C:\Users\john9\nabi_hyoghaw\data\raw\filings"

YEARS = [2021, 2022, 2023, 2024]

# Keywords to search for (case-insensitive)
KEYWORDS = [
    "CATL", "Samsung SDI", "EVE Energy", "Northvolt", "BASF", "Panasonic",
    "LG Energy", "BYD", "Gotion", "Umicore", "Glencore",
    "cobalt", "lithium", "nickel", "cathode", "battery cell",
    "separator", "battery", "cell supplier", "cell production",
    "raw material", "recycling", "anode", "electrolyte",
    "mining", "supply chain", "circular", "Gen6", "Gen5",
    "Neue Klasse", "round cell", "prismatic"
]

# Compile a combined pattern for quick scanning
kw_pattern = re.compile(
    "|".join(re.escape(kw) for kw in KEYWORDS),
    re.IGNORECASE
)

# Section header patterns to identify relevant sections
SECTION_PATTERNS = [
    re.compile(r"purchas\w*\s+and\s+supplier", re.IGNORECASE),
    re.compile(r"production\s+and\s+supplier", re.IGNORECASE),
    re.compile(r"supplier\s+network", re.IGNORECASE),
    re.compile(r"raw\s+materials?\s+and\s+supply", re.IGNORECASE),
    re.compile(r"procurement", re.IGNORECASE),
    re.compile(r"supply\s+chain", re.IGNORECASE),
    re.compile(r"battery", re.IGNORECASE),
    re.compile(r"electromobility", re.IGNORECASE),
    re.compile(r"e-mobility", re.IGNORECASE),
    re.compile(r"sustainability", re.IGNORECASE),
]


def classify_edge(sentence, year):
    """Attempt to classify the relationship described in a sentence."""
    s_lower = sentence.lower()
    edges = []

    # Map supplier names
    suppliers = {
        "CATL": "CATL",
        "Samsung SDI": "Samsung SDI",
        "EVE Energy": "EVE Energy",
        "Northvolt": "Northvolt",
        "BASF": "BASF",
        "Panasonic": "Panasonic",
        "LG Energy": "LG Energy",
        "BYD": "BYD",
        "Gotion": "Gotion",
        "Umicore": "Umicore",
        "Glencore": "Glencore",
    }

    materials = {
        "cobalt": "Cobalt",
        "lithium": "Lithium",
        "nickel": "Nickel",
        "cathode": "Cathode",
        "battery cell": "Battery_Cell",
        "separator": "Separator",
        "anode": "Anode",
        "electrolyte": "Electrolyte",
    }

    # Find which suppliers are mentioned
    found_suppliers = []
    for key, name in suppliers.items():
        if key.lower() in s_lower:
            found_suppliers.append(name)

    # Find which materials are mentioned
    found_materials = []
    for key, name in materials.items():
        if key.lower() in s_lower:
            found_materials.append(name)

    # Determine relationship type
    rel = "SUPPLIES"
    if any(w in s_lower for w in ["recycl", "circular", "closed-loop", "closed loop"]):
        rel = "RECYCLES"
    elif any(w in s_lower for w in ["partner", "cooperation", "collaborat", "joint", "agreement"]):
        rel = "PARTNERS_WITH"
    elif any(w in s_lower for w in ["invest", "stake", "acquisition"]):
        rel = "INVESTS_IN"
    elif any(w in s_lower for w in ["produc", "manufactur", "plant", "factory", "facility"]):
        rel = "PRODUCES_FOR"
    elif any(w in s_lower for w in ["supply", "suppli", "sourc", "procur", "deliver"]):
        rel = "SUPPLIES"

    # Build edges
    confidence = 0.7  # base confidence for keyword co-occurrence
    if found_suppliers and found_materials:
        confidence = 0.85
        for sup in found_suppliers:
            for mat in found_materials:
                edges.append((sup, rel, f"BMW:{mat}", confidence))
    elif found_suppliers:
        for sup in found_suppliers:
            edges.append((sup, rel, "BMW", confidence))
    elif found_materials:
        for mat in found_materials:
            edges.append(("BMW", "USES", mat, 0.6))

    return edges


def extract_sentences_with_keywords(text, page_num, year):
    """Split text into sentences and find those containing keywords."""
    # Split into sentences (rough but workable)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    results = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 15:
            continue
        if kw_pattern.search(sent):
            # Truncate to 300 chars
            display = sent[:300]
            results.append((page_num, display, sent))
    return results


def find_toc_sections(doc):
    """Find relevant page ranges by scanning TOC or early pages."""
    relevant_pages = set()

    # Strategy 1: Check the PDF's built-in TOC
    toc = doc.get_toc()
    if toc:
        for level, title, page in toc:
            for pat in SECTION_PATTERNS:
                if pat.search(title):
                    # Add a range around this page
                    for p in range(max(0, page - 2), min(len(doc), page + 25)):
                        relevant_pages.add(p)

    # Strategy 2: Scan pages 2-15 for TOC-like content
    for pg_idx in range(min(15, len(doc))):
        page = doc[pg_idx]
        text = page.get_text()
        # Look for TOC entries with page numbers
        for line in text.split('\n'):
            for pat in SECTION_PATTERNS:
                if pat.search(line):
                    # Try to extract page number from the line
                    nums = re.findall(r'\b(\d{2,3})\b', line)
                    for n in nums:
                        n = int(n)
                        if 10 < n < len(doc):
                            for p in range(max(0, n - 2), min(len(doc), n + 25)):
                                relevant_pages.add(p)

    # Strategy 3: Always scan pages 80-160 (common location for these sections)
    for p in range(min(80, len(doc)), min(170, len(doc))):
        relevant_pages.add(p)

    # Strategy 4: Full scan of all pages for battery-specific keywords
    # (only add pages that actually contain high-value keywords)
    high_value = re.compile(
        r"CATL|Samsung SDI|EVE Energy|Northvolt|Gotion|battery cell|cathode|"
        r"Umicore|Glencore|cell supplier|round cell|Gen[56]|Neue Klasse",
        re.IGNORECASE
    )
    for pg_idx in range(len(doc)):
        if pg_idx not in relevant_pages:
            page = doc[pg_idx]
            text = page.get_text()
            if high_value.search(text):
                relevant_pages.add(pg_idx)

    return sorted(relevant_pages)


def process_pdf(year):
    """Process a single BMW annual report PDF."""
    path = os.path.join(BASE, f"bmw_ar_{year}.pdf")
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return

    doc = fitz.open(path)
    print(f"\n{'='*80}")
    print(f"BMW Annual Report {year} - {len(doc)} pages")
    print(f"{'='*80}")

    # Find relevant pages
    relevant_pages = find_toc_sections(doc)
    print(f"Scanning {len(relevant_pages)} candidate pages...")

    seen_sentences = set()
    findings = []

    for pg_idx in relevant_pages:
        page = doc[pg_idx]
        text = page.get_text()
        results = extract_sentences_with_keywords(text, pg_idx + 1, year)  # 1-indexed

        for page_num, display, full_sent in results:
            # Deduplicate
            sig = full_sent[:100].lower().strip()
            if sig in seen_sentences:
                continue
            seen_sentences.add(sig)

            edges = classify_edge(full_sent, year)
            findings.append((page_num, display, edges))

    # Print findings
    count = 0
    for page_num, display, edges in findings:
        print(f"\n[BMW AR {year} p.{page_num}] {display}")
        if edges:
            for src, rel, dst, conf in edges:
                print(f"EDGE: {src} --{rel}--> {dst} | confidence: {conf}")
                count += 1
        else:
            print(f"EDGE: (context-only, no entity pair extracted)")

    doc.close()
    print(f"\n--- {year}: {len(findings)} sentences, {count} edges extracted ---")
    return findings


if __name__ == "__main__":
    all_findings = {}
    for year in YEARS:
        all_findings[year] = process_pdf(year)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    total_sentences = 0
    total_edges = 0
    for year in YEARS:
        if all_findings[year]:
            n_sent = len(all_findings[year])
            n_edges = sum(len(edges) for _, _, edges in all_findings[year])
            total_sentences += n_sent
            total_edges += n_edges
            print(f"  {year}: {n_sent} sentences, {n_edges} edges")
    print(f"  TOTAL: {total_sentences} sentences, {total_edges} edges")
