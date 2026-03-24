from __future__ import annotations
import logging
from rich.logging import RichHandler

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

def get_logger(name: str = "nabi") -> logging.Logger:
    return logging.getLogger(name)
