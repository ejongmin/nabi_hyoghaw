from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import importlib.util

@dataclass
class DoctorResult:
    ok: bool
    messages: List[str]

def _has_pkg(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

def doctor(cfg_raw: Dict[str, Any], root_dir: Path, paths: Dict[str,str]) -> DoctorResult:
    msgs: List[str] = []
    ok = True

    # Required files
    def need(path_key: str, human: str):
        nonlocal ok
        rel = paths.get(path_key)
        if not rel:
            ok = False
            msgs.append(f"❌ config.paths.{path_key} 누락: {human}")
            return
        p = (root_dir / rel).resolve()
        if not p.exists():
            ok = False
            msgs.append(f"❌ 파일 없음: {human} -> {p}")
        else:
            msgs.append(f"✅ 파일 OK: {human} -> {p}")

    need("universe_csv", "Universe(기업 마스터)")
    # seed_edges는 자동 생성할 수 있으므로 warn 처리
    seed_rel = paths.get("seed_edges_csv")
    if seed_rel:
        seed_path = (root_dir/seed_rel).resolve()
        if seed_path.exists():
            msgs.append(f"✅ 파일 OK: seed_edges -> {seed_path}")
        else:
            msgs.append(f"⚠️ seed_edges 없음(자동 생성 가능): {seed_path}")
    else:
        msgs.append("⚠️ config.paths.seed_edges_csv 누락(추가 권장)")

    # Optional deps based on config
    risk = cfg_raw.get("risk", {})
    if bool(risk.get("use_finbert", False)):
        if _has_pkg("torch") and _has_pkg("transformers"):
            msgs.append("✅ FinBERT 의존성(torch/transformers) OK")
        else:
            ok = False
            msgs.append("❌ FinBERT 사용 설정인데 torch/transformers가 없습니다. requirements_nlp.txt 설치 필요")

    gnn = cfg_raw.get("gnn", {})
    if bool(gnn.get("enabled", False)):
        if _has_pkg("torch") and _has_pkg("torch_geometric"):
            msgs.append("✅ GNN 의존성(torch/torch_geometric) OK")
        else:
            ok = False
            msgs.append("❌ GNN 사용 설정인데 torch_geometric가 없습니다. requirements_gnn.txt 설치 필요")

    # tabulate check (markdown tables)
    if _has_pkg("tabulate"):
        msgs.append("✅ tabulate OK (markdown table)")
    else:
        msgs.append("⚠️ tabulate 없음: markdown table이 CSV 텍스트로 대체됩니다(동작은 함)")

    return DoctorResult(ok=ok, messages=msgs)
