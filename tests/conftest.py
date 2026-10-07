"""Shared fixtures: synthetic market data and a dummy model (no network, no MLflow server)."""

import numpy as np
import pandas as pd
import pytest

from api.model_store import ModelBundle


@pytest.fixture
def raw_market() -> pd.DataFrame:
    """~3 years of business days with alternating calm/turbulent regimes."""
    rng = np.random.default_rng(7)
    n = 750
    dates = pd.bdate_range("2020-01-01", periods=n, name="date")
    sigma = np.where(np.sin(np.arange(n) / 40) > 0.5, 0.025, 0.008)
    close = 3000 * np.exp(np.cumsum(rng.normal(0.0003, sigma)))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, sigma / 2)),
            "high": close * (1 + np.abs(rng.normal(0, sigma))),
            "low": close * (1 - np.abs(rng.normal(0, sigma))),
            "close": close,
            "volume": rng.integers(2_000_000_000, 5_000_000_000, n).astype(float),
            "vix_close": 100 * sigma * np.sqrt(252) * (1 + rng.normal(0, 0.05, n)),
        },
        index=dates,
    )


class DummyModel:
    """Stands in for the sklearn pipeline: always predicts P(high vol) = 0.7."""

    def predict_proba(self, x):
        p = np.full(len(x), 0.7)
        return np.column_stack([1 - p, p])


@pytest.fixture
def bundle(raw_market) -> ModelBundle:
    return ModelBundle(
        model=DummyModel(),
        metadata={
            "name": "vol-regime",
            "version": 3,
            "alias": "champion",
            "decision_threshold": 0.5,
            "metrics": {"roc_auc": 0.81},
        },
        reference=raw_market.tail(400),
    )
