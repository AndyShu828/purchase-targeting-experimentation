"""Load environment-specific settings from configs/<env>.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(env: str | None = None) -> dict[str, Any]:
    """Read configs/<env>.yaml; env defaults to $PTE_ENV, then 'dev'."""
    env = env or os.environ.get("PTE_ENV", "dev")
    path = PROJECT_ROOT / "configs" / f"{env}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path
