from __future__ import annotations
from pathlib import Path
import pandas as pd

def write_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def build_index(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Reports index", ""]
    for f in sorted(reports_dir.glob("*.md")):
        if f.name == "index.md":
            continue
        md.append(f"- {f.name}")
    write_md(reports_dir / "index.md", "\n".join(md) + "\n")
