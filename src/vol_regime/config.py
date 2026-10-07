"""Shared paths and parameter loading. Paths are relative to the repo root (where DVC runs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PARAMS_PATH = Path("params.yaml")
RAW_DATA_PATH = Path("data/raw/market.csv")
DATASET_PATH = Path("data/processed/dataset.csv")
METRICS_PATH = Path("reports/metrics.json")
REFERENCE_PATH = Path("reports/reference_window.csv")


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
