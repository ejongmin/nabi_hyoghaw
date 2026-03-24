from src.common.config import Config
from pathlib import Path

def test_config_load():
    cfg = Config.load("configs/base.yaml")
    assert cfg.get("project","name") is not None
