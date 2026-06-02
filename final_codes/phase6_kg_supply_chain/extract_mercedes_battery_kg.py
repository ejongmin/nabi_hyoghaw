"""
Extract battery supply chain relationships from Mercedes-Benz Annual Reports (2021-2024).
Uses PyMuPDF to extract text, then searches for specific entities and keywords.
"""

import fitz  # PyMuPDF
import re
import os
import json

BASE_DIR = r"C:\Users\john9\nabi_hyoghaw"
PDF_FILES = {
    2021: os.path.join(BASE_DIR, "data", "raw", "filings", "mercedes_ar_2021.pdf"),
    2022: os.path.join(BASE_DIR, "data", "raw", "filings", "mercedes_ar_2022.pdf"),
    2023: os.path.join(BASE_DIR, "data", "raw", "filings", "mercedes_ar_2023.pdf"),
    2024: os.path.join(BASE_DIR, "data", "raw", "filings", "mercedes_ar_2024.pdf"),
}

# Target entities
ENTITIES = [
    "CATL", "Samsung SDI", "LG Energy", "Farasis", "ACC",
    "BYD", "Panasonic", "Gotion", "BASF", "Umicore",
    "Glencore", "Geely", "smart",
]

# Battery supply chain keywords
KEYWORDS = [
    "battery cell", "cathode", "lithium", "cobalt", "separator",
    "battery", "cell factory", "cell production", "cell supplier",
    "anode", "electrolyte", "nickel", "recycling",
    "raw material", "electric drive", "e-drive",
]

# TOC section names to look for (case-insensitive partial matches)
TOC_SECTIONS = [
    "procurement", "supply chain", "battery", "electric",
    "risk", "technology", "research", "development",
    "purchasing", "supplier", "raw material", "drive system",
    "powertrain", "sustainability", "environment",
    "production", "strategy", "innovation",
]


def find_relevant_pages_via_toc(doc, year):
    """Try to find relevant pages from TOC entries."""
    toc = doc.get_toc()
    relevant_pages = set()

    if toc:
        for level, title, page in toc:
            title_lower = title.lower()
            for section in TOC_SECTIONS:
                if section in title_lower:
                    # Add the page and a range around it
                    for p in range(max(0, page - 1), min(len(doc), page + 30)):
                        relevant_pages.add(p)
                    break

    return relevant_pages


def find_relevant_pages_via_scan(doc):
    """Scan all pages for keyword mentions to find relevant pages."""
    relevant_pages = set()
    all_terms = ENTITIES + KEYWORDS

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        text_lower = text.lower()

        for term in all_terms:
            if term.lower() in text_lower:
                relevant_pages.add(page_num)
                break

    return relevant_pages


def extract_sentences(text):
    """Split text into sentences, handling common PDF artifacts."""
    # Clean up text
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def classify_relationship(sentence, entity):
    """Classify the relationship type based on sentence context."""
    s_lower = sentence.lower()
    e_lower = entity.lower()

    # Supply / procurement
    if any(w in s_lower for w in ["supply", "supplier", "procur", "purchas", "sourc"]):
        return "supplies_to"
    # Joint venture / partnership
    if any(w in s_lower for w in ["joint venture", "partnership", "cooperation", "collaborat", "together with"]):
        return "joint_venture_with"
    # Investment
    if any(w in s_lower for w in ["invest", "stake", "shareholding", "equity", "acquired"]):
        return "invested_in"
    # Production / manufacturing
    if any(w in s_lower for w in ["produc", "manufactur", "factory", "plant", "facility"]):
        return "produces_for"
    # Technology
    if any(w in s_lower for w in ["technolog", "develop", "innovat", "research", "patent"]):
        return "technology_partner_of"
    # Recycling
    if any(w in s_lower for w in ["recycl", "circular", "second life", "reuse"]):
        return "recycling_partner_of"
    # Raw materials
    if any(w in s_lower for w in ["raw material", "lithium", "cobalt", "nickel", "mining"]):
        return "raw_material_supplier_to"
    # Battery cell
    if any(w in s_lower for w in ["battery cell", "cell production", "cell supplier"]):
        return "battery_cell_supplier_to"
    # General mention
    return "mentioned_with"


def estimate_confidence(sentence, entity, rel):
    """Estimate confidence based on directness of the mention."""
    s_lower = sentence.lower()
    conf = 0.5

    # Direct mention of Mercedes relationship
    if any(w in s_lower for w in ["mercedes", "daimler", "we ", "our ", "the group", "the company"]):
        conf += 0.2

    # Specific relationship language
    if rel != "mentioned_with":
        conf += 0.1

    # Entity appears with battery context
    if any(w in s_lower for w in ["battery", "cell", "electric", "ev"]):
        conf += 0.1

    # Contract / agreement language
    if any(w in s_lower for w in ["contract", "agreement", "signed", "awarded"]):
        conf += 0.1

    return min(conf, 1.0)


def determine_edge_nodes(entity, rel, sentence):
    """Determine source and destination nodes."""
    s_lower = sentence.lower()

    # For supply relationships, the entity typically supplies to Mercedes
    if rel in ["supplies_to", "battery_cell_supplier_to", "raw_material_supplier_to", "produces_for"]:
        return entity, "Mercedes-Benz"
    # For joint ventures, it's bidirectional but we pick a direction
    if rel in ["joint_venture_with", "technology_partner_of", "recycling_partner_of"]:
        return "Mercedes-Benz", entity
    # For investments, Mercedes invests in entity
    if rel == "invested_in":
        return "Mercedes-Benz", entity

    return "Mercedes-Benz", entity


def process_pdf(year, filepath):
    """Process a single PDF and extract findings."""
    findings = []

    if not os.path.exists(filepath):
        print(f"WARNING: File not found: {filepath}")
        return findings

    doc = fitz.open(filepath)
    print(f"\n{'='*80}")
    print(f"Processing Mercedes AR {year} ({len(doc)} pages)")
    print(f"{'='*80}")

    # Get TOC
    toc = doc.get_toc()
    if toc:
        print(f"  TOC entries: {len(toc)}")
        for level, title, page in toc:
            title_lower = title.lower()
            for section in TOC_SECTIONS:
                if section in title_lower:
                    print(f"    [p.{page}] {'  '*level}{title}")
                    break
    else:
        print("  No TOC found, will scan all pages.")

    # Find relevant pages
    toc_pages = find_relevant_pages_via_toc(doc, year)
    scan_pages = find_relevant_pages_via_scan(doc)
    relevant_pages = toc_pages | scan_pages

    if not relevant_pages:
        # Fall back to scanning all pages
        relevant_pages = set(range(len(doc)))

    print(f"  Relevant pages: {len(relevant_pages)} (TOC: {len(toc_pages)}, Scan: {len(scan_pages)})")

    # Process relevant pages
    for page_num in sorted(relevant_pages):
        page = doc[page_num]
        text = page.get_text()

        if not text or len(text.strip()) < 50:
            continue

        sentences = extract_sentences(text)

        for sentence in sentences:
            s_lower = sentence.lower()

            for entity in ENTITIES:
                # Special handling for "smart" - require context to avoid false positives
                if entity == "smart":
                    # Look for "smart" as a brand (car brand, joint venture context)
                    if not re.search(r'\bsmart\b(?:\s+(?:brand|car|vehicle|joint venture|Automobile|EQ|fortwo|forfour|#1))', sentence, re.IGNORECASE):
                        # Also check for "smart Automobile" or specific patterns
                        if not re.search(r'\bsmart\s+Automobile\b', sentence):
                            continue

                elif entity == "ACC":
                    # ACC could be an abbreviation for many things - require battery/cell context
                    if not re.search(r'\bACC\b', sentence):
                        continue
                    if not any(w in s_lower for w in ["battery", "cell", "automotive cells", "gigafactory"]):
                        continue

                else:
                    if entity.lower() not in s_lower:
                        continue

                # Found a match
                rel = classify_relationship(sentence, entity)
                conf = estimate_confidence(sentence, entity, rel)
                src, dst = determine_edge_nodes(entity, rel, sentence)

                # Truncate sentence to 300 chars
                display_sentence = sentence[:300]

                findings.append({
                    "year": year,
                    "page": page_num + 1,  # 1-indexed
                    "sentence": display_sentence,
                    "entity": entity,
                    "rel": rel,
                    "src": src,
                    "dst": dst,
                    "confidence": conf,
                })

    doc.close()
    return findings


def main():
    all_findings = []

    for year, filepath in sorted(PDF_FILES.items()):
        findings = process_pdf(year, filepath)
        all_findings.extend(findings)

    # Deduplicate similar findings (same entity, same page, similar sentence)
    seen = set()
    unique_findings = []
    for f in all_findings:
        key = (f["year"], f["page"], f["entity"], f["sentence"][:80])
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    # Print results
    print(f"\n\n{'='*80}")
    print(f"RESULTS: {len(unique_findings)} unique findings across all years")
    print(f"{'='*80}\n")

    for f in unique_findings:
        print(f"[Mercedes AR {f['year']} p.{f['page']}] {f['sentence']}")
        print(f"EDGE: {f['src']} --{f['rel']}--> {f['dst']} | confidence: {f['confidence']:.1f}")
        print()

    # Also save as JSON for downstream use
    output_path = os.path.join(BASE_DIR, "data", "raw", "filings", "mercedes_battery_kg_findings.json")
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(unique_findings, fp, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(unique_findings)} findings to {output_path}")


if __name__ == "__main__":
    main()
