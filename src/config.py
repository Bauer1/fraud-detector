"""Configuration loader for the Fraud Detection Service."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and apply environment variable overrides."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    if os.getenv("DATABASE_URL"):
        cfg.setdefault("paths", {})["database_url"] = os.environ["DATABASE_URL"]
    if os.getenv("OPENAI_API_KEY"):
        cfg.setdefault("llm", {})["enabled"] = True
    if os.getenv("FRAUD_API_KEY"):
        sec = cfg.setdefault("security", {})
        sec["require_api_key"] = True
        sec["api_key"] = os.environ["FRAUD_API_KEY"]
    if os.getenv("FRAUD_PSEUDO_SALT"):
        cfg.setdefault("security", {})["pseudonymization_salt"] = os.environ["FRAUD_PSEUDO_SALT"]
    if os.getenv("FRAUD_SHADOW_MODE"):
        cfg.setdefault("scoring", {})["shadow_mode"] = os.environ["FRAUD_SHADOW_MODE"].lower() in ("1", "true", "yes")

    return cfg


def resolve_path(rel_or_abs: str | Path) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not already absolute."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else PROJECT_ROOT / p
