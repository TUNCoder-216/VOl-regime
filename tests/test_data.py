import pandas as pd
import pytest

from vol_regime import data
from vol_regime.data import RAW_COLUMNS, fetch_market_data, validate_market_data


def test_valid_data_passes(raw_market):
    validate_market_data(raw_market)


def test_missing_column_is_rejected(raw_market):
    with pytest.raises(ValueError, match="Missing columns"):
        validate_market_data(raw_market.drop(columns=["vix_close"]))


def test_duplicate_dates_are_rejected(raw_market):
    duplicated = pd.concat([raw_market, raw_market.tail(1)])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_market_data(duplicated)


def test_non_positive_prices_are_rejected(raw_market):
    broken = raw_market.copy()
    broken.iloc[10, broken.columns.get_loc("close")] = -1.0
    with pytest.raises(ValueError, match="positive"):
        validate_market_data(broken)


def test_too_few_rows_are_rejected(raw_market):
    with pytest.raises(ValueError, match="at least"):
        validate_market_data(raw_market.head(50))


def test_fetch_joins_index_and_vix(monkeypatch, raw_market):
    def fake_history(ticker, start, end):
        frame = raw_market.rename(columns=str.capitalize)
        if ticker == "^VIX":
            return frame.assign(Close=raw_market["vix_close"])
        return frame

    monkeypatch.setattr(data, "_history", fake_history)
    df = fetch_market_data("2020-01-01", "2023-01-01")
    assert list(df.columns) == RAW_COLUMNS
    assert df.index.name == "date"
    assert len(df) == len(raw_market)
    pd.testing.assert_series_equal(
        df["vix_close"], raw_market["vix_close"], check_names=False, check_freq=False
    )
