"""Pure modelling logic (no MLflow): candidates, time split, metrics, quality gate."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vol_regime.features import FEATURE_COLUMNS


def build_candidates(train_params: dict[str, Any]) -> dict[str, Pipeline]:
    seed = train_params["random_state"]
    lr, hgb = train_params["logreg"], train_params["hgb"]
    return {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=lr["C"], class_weight="balanced", max_iter=2000, random_state=seed
                    ),
                ),
            ]
        ),
        "hgb": Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=hgb["max_iter"],
                        learning_rate=hgb["learning_rate"],
                        max_depth=hgb["max_depth"],
                        l2_regularization=hgb["l2_regularization"],
                        class_weight="balanced",
                        random_state=seed,
                    ),
                )
            ]
        ),
    }


def time_split(
    dataset: pd.DataFrame, test_size: float, gap: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological split. `gap` rows are dropped before the test set because the last
    training labels look `horizon` days ahead, i.e. into the test period (purging)."""
    split = int(len(dataset) * (1 - test_size))
    train = dataset.iloc[: max(split - gap, 0)]
    test = dataset.iloc[split:]
    if train.empty or test.empty:
        raise ValueError("Not enough rows for a train/test split")
    return train[FEATURE_COLUMNS], test[FEATURE_COLUMNS], train["label"], test["label"]


def evaluate(y_true: np.ndarray, proba: np.ndarray, decision_threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= decision_threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if both_classes else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
        "positive_rate": float(y_true.mean()),
    }


def select_best(results: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    return max(results, key=lambda r: r["metrics"][metric])


def passes_quality_gate(metrics: dict[str, float], gate: dict[str, Any]) -> bool:
    value = metrics.get(gate["metric"], float("nan"))
    return bool(np.isfinite(value) and value >= gate["min_value"])
