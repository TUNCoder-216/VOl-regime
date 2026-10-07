"""Market data acquisition and validation (DVC stage: download)."""

from __future__ import annotations

import logging

import pandas as pd

from vol_regime.config import RAW_DATA_PATH, load_params

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
RAW_COLUMNS = [*PRICE_COLUMNS, "vix_close"]
MIN_ROWS = 300


def _history(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # lazy import: offline code paths and tests never need it

    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} between {start} and {end}")
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    df.index = index.normalize()
    return df


def fetch_market_data(
    start: str, end: str, index_ticker: str = "^GSPC", vix_ticker: str = "^VIX"
) -> pd.DataFrame:
    """Daily OHLCV of the index joined with the VIX close, one row per trading day."""
    idx = _history(index_ticker, start, end)[["Open", "High", "Low", "Close", "Volume"]].copy()
    idx.columns = PRICE_COLUMNS
    vix = _history(vix_ticker, start, end)[["Close"]].rename(columns={"Close": "vix_close"})
    df = idx.join(vix, how="inner").dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "date"
    return df


def validate_market_data(df: pd.DataFrame, min_rows: int = MIN_ROWS) -> None:
    """Fail fast on bad data instead of training a silently broken model."""
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) < min_rows:
        raise ValueError(f"Expected at least {min_rows} rows, got {len(df)}")
    if df.index.has_duplicates:
        raise ValueError("Duplicate dates in market data")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted")
    if df[RAW_COLUMNS].isna().any().any():
        raise ValueError("Market data contains NaN values")
    if (df[["open", "high", "low", "close", "vix_close"]] <= 0).any().any():
        raise ValueError("Prices must be strictly positive")
    if (df["high"] < df["low"]).any():
        raise ValueError("Found rows where high < low")


def load_raw(path=RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params()["data"]
    df = fetch_market_data(
        params["start_date"], params["end_date"], params["index_ticker"], params["vix_ticker"]
    )
    validate_market_data(df)
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH)
    logger.info(
        "Saved %d rows (%s -> %s) to %s",
        len(df),
        df.index.min().date(),
        df.index.max().date(),
        RAW_DATA_PATH,
    )


if __name__ == "__main__":
    main()
