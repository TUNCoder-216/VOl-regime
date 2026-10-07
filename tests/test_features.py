import numpy as np
import pandas as pd

from vol_regime.features import (
    FEATURE_COLUMNS,
    build_dataset,
    compute_features,
    compute_labels,
    latest_features,
)

PARAMS = {"horizon": 5, "threshold_window": 252, "threshold_quantile": 0.75}


def test_feature_columns_and_order(raw_market):
    assert list(compute_features(raw_market).columns) == FEATURE_COLUMNS


def test_features_have_no_lookahead(raw_market):
    """Deleting future rows must not change any past feature value."""
    cut = 500
    full = compute_features(raw_market)
    truncated = compute_features(raw_market.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], truncated)


def test_threshold_has_no_lookahead(raw_market):
    cut = 500
    full = compute_labels(raw_market, **PARAMS)["threshold"]
    truncated = compute_labels(raw_market.iloc[:cut], **PARAMS)["threshold"]
    pd.testing.assert_series_equal(full.iloc[:cut], truncated)


def test_label_uses_the_next_horizon_days(raw_market):
    labels = compute_labels(raw_market, **PARAMS)
    t = 400
    logret = np.log(raw_market["close"]).diff()
    expected_fwd = logret.iloc[t + 1 : t + 6].std() * np.sqrt(252)
    assert np.isclose(labels["fwd_rv"].iloc[t], expected_fwd)
    expected_label = float(expected_fwd > labels["threshold"].iloc[t])
    assert labels["label"].iloc[t] == expected_label


def test_last_horizon_rows_have_no_label(raw_market):
    labels = compute_labels(raw_market, **PARAMS)
    assert labels["label"].tail(PARAMS["horizon"]).isna().all()


def test_dataset_is_clean_and_balanced_enough(raw_market):
    dataset = build_dataset(raw_market, **PARAMS)
    assert not dataset.isna().any().any()
    assert set(dataset["label"].unique()) <= {0, 1}
    assert 0.05 < dataset["label"].mean() < 0.6


def test_latest_features_returns_last_complete_row(raw_market):
    as_of, row = latest_features(raw_market)
    assert as_of == raw_market.index[-1]
    assert row.shape == (1, len(FEATURE_COLUMNS))
    assert not row.isna().any().any()
