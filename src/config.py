"""Configuration loader for risk keywords and project paths."""

from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "gdelt"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
UNIVERSE_DIR = PROJECT_ROOT / "data" / "universe"
CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"

ARTICLES_CSV = RAW_DIR / "articles.csv"
RISK_EVENTS_PARQUET = PROCESSED_DIR / "risk_events.parquet"
RISK_KEYWORDS_YAML = CONFIGS_DIR / "risk_keywords.yaml"


def load_risk_keywords() -> dict:
    """Load risk taxonomy keywords from YAML config."""
    with open(RISK_KEYWORDS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
