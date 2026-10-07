"""Feature engineering and labelling (DVC stage: features).

Shared by training AND the API, so features are computed identically in both places.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from vol_regime.config import DATASET_PATH, load_params
from vol_regime.data import load_raw

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "rv_5d",
    "rv_10d",
    "rv_21d",
    "rv_ratio_5_21",
    "range_pct",
    "range_5d_mean",
    "momentum_21d",
    "drawdown_252d",
    "vix",
    "vix_chg_5d",
    "vix_rv_spread",
    "volume_z_21d",
]


def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking features only: the value at date t uses data up to t, never after."""
    close = raw["close"]
    log_close = np.log(close)
    logret = log_close.diff()

    f = pd.DataFrame(index=raw.index)
    f["ret_1d"] = logret
    f["ret_5d"] = log_close.diff(5)
    for w in (5, 10, 21):
        f[f"rv_{w}d"] = logret.rolling(w).std() * np.sqrt(TRADING_DAYS)
    f["rv_ratio_5_21"] = f["rv_5d"] / f["rv_21d"]
    f["range_pct"] = (raw["high"] - raw["low"]) / close
    f["range_5d_mean"] = f["range_pct"].rolling(5).mean()
    f["momentum_21d"] = log_close.diff(21)
    f["drawdown_252d"] = close / close.rolling(TRADING_DAYS, min_periods=60).max() - 1.0
    f["vix"] = raw["vix_close"]
    f["vix_chg_5d"] = raw["vix_close"] / raw["vix_close"].shift(5) - 1.0
    f["vix_rv_spread"] = raw["vix_close"] / 100.0 - f["rv_21d"]
    volume = raw["volume"].astype(float)
    f["volume_z_21d"] = (volume - volume.rolling(21).mean()) / volume.rolling(21).std()
    return f.replace([np.inf, -np.inf], np.nan)[FEATURE_COLUMNS]


def compute_labels(
    raw: pd.DataFrame, horizon: int, threshold_window: int, threshold_quantile: float
) -> pd.DataFrame:
    """label(t) = 1 if realized vol over t+1..t+horizon beats the quantile of PAST horizon-vols."""
    logret = np.log(raw["close"]).diff()
    rv_h = logret.rolling(horizon).std() * np.sqrt(TRADING_DAYS)  # window t-h+1..t
    fwd_rv = rv_h.shift(-horizon)  # window t+1..t+h (the future we predict)
    threshold = rv_h.rolling(threshold_window, min_periods=threshold_window // 2).quantile(
        threshold_quantile
    )  # past only
    label = (fwd_rv > threshold).astype(float)
    label[fwd_rv.isna() | threshold.isna()] = np.nan
    return pd.DataFrame({"fwd_rv": fwd_rv, "threshold": threshold, "label": label})


def build_dataset(
    raw: pd.DataFrame, horizon: int, threshold_window: int, threshold_quantile: float
) -> pd.DataFrame:
    labels = compute_labels(raw, horizon, threshold_window, threshold_quantile)
    dataset = compute_features(raw).join(labels).dropna()
    dataset["label"] = dataset["label"].astype(int)
    return dataset


def latest_features(raw: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Most recent complete feature row, used at inference time."""
    features = compute_features(raw).dropna()
    if features.empty:
        raise ValueError("Not enough history to compute features")
    return features.index[-1], features.iloc[[-1]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params()["features"]
    dataset = build_dataset(
        load_raw(), params["horizon"], params["threshold_window"], params["threshold_quantile"]
    )
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATASET_PATH)
    logger.info(
        "Dataset: %d rows, positive rate %.1f%%", len(dataset), 100 * dataset["label"].mean()
    )


if __name__ == "__main__":
    main()
