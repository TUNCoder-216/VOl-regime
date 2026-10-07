"""Loading the model bundle (model + metadata + reference window) from MODEL_DIR/current."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelBundle:
    model: Any
    metadata: dict[str, Any]
    reference: pd.DataFrame | None = None

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x))[:, 1]


def load_bundle(model_dir: Path) -> ModelBundle | None:
    current = model_dir / "current"
    meta_path = current / "metadata.json"
    if not meta_path.exists():
        return None
    mlmodel = next((current / "model").rglob("MLmodel"), None)
    if mlmodel is None:
        raise FileNotFoundError(f"No MLmodel file under {current / 'model'}")

    import mlflow.sklearn  # imported here so tests with a dummy model never need it

    model = mlflow.sklearn.load_model(str(mlmodel.parent))
    ref_path = next(current.rglob("reference_window.csv"), None)
    reference = pd.read_csv(ref_path, parse_dates=["date"], index_col="date") if ref_path else None
    return ModelBundle(model=model, metadata=json.loads(meta_path.read_text()), reference=reference)
