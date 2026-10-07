import numpy as np
import pytest

from vol_regime.features import FEATURE_COLUMNS, build_dataset
from vol_regime.modeling import (
    build_candidates,
    evaluate,
    passes_quality_gate,
    select_best,
    time_split,
)

TRAIN_PARAMS = {
    "random_state": 42,
    "logreg": {"C": 0.5},
    "hgb": {"max_iter": 50, "learning_rate": 0.1, "max_depth": 3, "l2_regularization": 1.0},
}


@pytest.fixture
def dataset(raw_market):
    return build_dataset(raw_market, horizon=5, threshold_window=252, threshold_quantile=0.75)


def test_time_split_is_chronological_with_gap(dataset):
    x_train, x_test, y_train, y_test = time_split(dataset, test_size=0.2, gap=5)
    assert x_train.index.max() < x_test.index.min()
    gap_rows = dataset.index.get_loc(x_test.index.min()) - dataset.index.get_loc(
        x_train.index.max()
    )
    assert gap_rows == 6  # 5 purged rows + the next row
    assert len(x_test) == len(dataset) - int(len(dataset) * 0.8)


def test_candidates_fit_and_predict(dataset):
    x_train, x_test, y_train, _ = time_split(dataset, test_size=0.2, gap=5)
    for model in build_candidates(TRAIN_PARAMS).values():
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test[FEATURE_COLUMNS])[:, 1]
        assert proba.shape == (len(x_test),)
        assert ((proba >= 0) & (proba <= 1)).all()


def test_evaluate_returns_bounded_metrics():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.8, 0.6, 0.3, 0.9])
    metrics = evaluate(y, p, 0.5)
    assert metrics["roc_auc"] == 1.0
    assert all(0 <= v <= 1 for v in metrics.values())


def test_evaluate_handles_single_class():
    metrics = evaluate(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]), 0.5)
    assert np.isnan(metrics["roc_auc"])


def test_quality_gate():
    gate = {"metric": "roc_auc", "min_value": 0.6}
    assert passes_quality_gate({"roc_auc": 0.65}, gate)
    assert not passes_quality_gate({"roc_auc": 0.55}, gate)
    assert not passes_quality_gate({"roc_auc": float("nan")}, gate)


def test_select_best():
    results = [
        {"name": "a", "metrics": {"roc_auc": 0.7}},
        {"name": "b", "metrics": {"roc_auc": 0.8}},
    ]
    assert select_best(results, "roc_auc")["name"] == "b"
