"""Configuration loader for the Fraud Detection Agent."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and resolve project-relative paths."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # Allow environment variable overrides for DB and LLM
    if os.getenv("DATABASE_URL"):
        cfg["paths"]["database_url"] = os.environ["DATABASE_URL"]
    if os.getenv("OPENAI_API_KEY"):
        cfg.setdefault("llm", {})["enabled"] = True

    return cfg


def resolve_path(rel_or_abs: str | Path) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not already absolute."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else PROJECT_ROOT / p
