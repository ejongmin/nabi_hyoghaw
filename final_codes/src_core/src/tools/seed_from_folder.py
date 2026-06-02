from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from src.tools.seed_from_xlsx import seed_edges_from_excel, SeedImportResult
from src.tools.seed_from_docx import seed_edges_from_docx_tables

@dataclass
class FolderImportRow:
    file: str
    kind: str
    rows_in: int
    rows_out: int
    mapped_src_rate: float
    mapped_dst_rate: float
    dropped_rows: int

@dataclass
class FolderImportResult:
    merged_edges: pd.DataFrame
    per_file: pd.DataFrame
    unmapped_preview: pd.DataFrame

def import_seed_from_folder(folder: Path,
                            universe: pd.DataFrame,
                            mapping_yaml: Path,
                            normalize_rules: Dict[str,str],
                            allowed_relations: List[str],
                            min_fuzzy_score: int = 88) -> FolderImportResult:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"folder not found: {folder}")

    rows: List[FolderImportRow] = []
    merged = []
    unmapped_all = []

    # xlsx first
    for p in sorted(folder.glob("*.xlsx")):
        res = seed_edges_from_excel(
            xlsx_path=p,
            universe=universe,
            mapping_yaml=mapping_yaml,
            normalize_rules=normalize_rules,
            allowed_relations=allowed_relations,
            sheet=0,
            min_fuzzy_score=min_fuzzy_score,
        )
        merged.append(res.edges)
        unmapped_all.append(res.unmapped_examples.assign(file=p.name))
        rows.append(FolderImportRow(p.name, "xlsx", res.rows_in, res.rows_out, res.mapped_src_rate, res.mapped_dst_rate, res.dropped_rows))

    # docx (tables)
    for p in sorted(folder.glob("*.docx")):
        tmp_xlsx = folder / f"__tmp_{p.stem}.xlsx"
        try:
            res = seed_edges_from_docx_tables(
                docx_path=p,
                temp_xlsx_path=tmp_xlsx,
                universe=universe,
                mapping_yaml=mapping_yaml,
                normalize_rules=normalize_rules,
                allowed_relations=allowed_relations,
                sheet=0,
                min_fuzzy_score=min_fuzzy_score,
            )
        finally:
            if tmp_xlsx.exists():
                tmp_xlsx.unlink()
        merged.append(res.edges)
        unmapped_all.append(res.unmapped_examples.assign(file=p.name))
        rows.append(FolderImportRow(p.name, "docx", res.rows_in, res.rows_out, res.mapped_src_rate, res.mapped_dst_rate, res.dropped_rows))

    merged_df = pd.concat(merged, ignore_index=True) if merged else pd.DataFrame(columns=[
        "src_company_id","rel_type","dst_company_id","confidence_plink","strength","evidence","source","valid_from","valid_to"
    ])
    # de-duplicate
    if len(merged_df) > 0:
        merged_df = merged_df.drop_duplicates(subset=["src_company_id","rel_type","dst_company_id"], keep="last")

    per_file = pd.DataFrame([r.__dict__ for r in rows])
    unmapped_preview = pd.concat(unmapped_all, ignore_index=True).head(20) if unmapped_all else pd.DataFrame()
    return FolderImportResult(merged_edges=merged_df, per_file=per_file, unmapped_preview=unmapped_preview)
