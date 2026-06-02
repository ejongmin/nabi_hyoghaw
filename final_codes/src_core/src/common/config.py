from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]
    config_path: Path
    root_dir: Path

    @staticmethod
    def load(path: str | Path) -> "Config":
        p = Path(path).expanduser().resolve()
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        root = data.get("project", {}).get("root_dir", None)
        if root is None:
            # assume config is in ./configs/
            root_dir = p.parent.parent
        else:
            root_dir = (p.parent / root).resolve() if root != "." else p.parent.parent
        return Config(raw=data, config_path=p, root_dir=root_dir)

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def path(self, *keys: str) -> Path:
        """Resolve a path from config.paths.* relative to root_dir."""
        rel = self.get("paths", *keys)
        if rel is None:
            raise KeyError(f"Missing paths.{'.'.join(keys)} in config")
        return (self.root_dir / rel).resolve()

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
