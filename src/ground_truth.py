"""Ground truth data fetching: EIA Cushing stocks and WTI futures prices."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from src.config import (
    DATA_START,
    DATA_END,
    EIA_CUSHING_STOCKS,
    EIA_DIR,
    TRAIN_END,
    WTI_FUTURES_TICKER,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EIA Cushing crude oil stocks
# ---------------------------------------------------------------------------

_EIA_API_V2_URL = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"

# Fallback: direct XLS download from EIA website (no API key required)
_EIA_XLS_URL = (
    "https://www.eia.gov/dnav/pet/hist_xls/W_EPC0_SAX_YCUOK_MBBLw.xls"
)


def fetch_eia_stocks(
    api_key: str | None = None,
    start: str = DATA_START,
    end: str = DATA_END,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch weekly Cushing, OK crude oil stocks from EIA API v2.

    Parameters
    ----------
    api_key : str or None
        EIA API key.  If *None*, the function tries the ``EIA_API_KEY``
        environment variable, then falls back to an unauthenticated request.
    start, end : str
        ISO date strings (``YYYY-MM-DD``).  Only the ``YYYY-MM`` portion is
        sent to the EIA API.
    cache_path : str or Path or None
        Where to cache the result as parquet.  Defaults to
        ``data/eia/cushing_stocks.parquet``.

    Returns
    -------
    pd.DataFrame
        Columns: ``stocks_mbbl`` (thousand barrels).
        Index: ``DatetimeIndex`` named ``date``.
    """
    if cache_path is None:
        cache_path = Path(EIA_DIR) / "cushing_stocks.parquet"
    cache_path = Path(cache_path)

    if cache_path.exists():
        logger.info("Loading cached EIA stocks from %s", cache_path)
        return pd.read_parquet(cache_path)

    # Resolve API key
    if api_key is None:
        api_key = os.environ.get("EIA_API_KEY", "")

    start_ym = start[:7]  # YYYY-MM
    end_ym = end[:7]

    params = {
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": EIA_CUSHING_STOCKS,
        "start": start_ym,
        "end": end_ym,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }
    if api_key:
        params["api_key"] = api_key

    logger.info("Fetching EIA Cushing stocks (series %s) ...", EIA_CUSHING_STOCKS)

    try:
        resp = requests.get(_EIA_API_V2_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if "response" not in payload or "data" not in payload["response"]:
            raise ValueError(
                f"Unexpected EIA response structure: {list(payload.keys())}"
            )

        records = payload["response"]["data"]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["period"])
        df = df.rename(columns={"value": "stocks_mbbl"})
        df["stocks_mbbl"] = pd.to_numeric(df["stocks_mbbl"], errors="coerce")
        df = df[["date", "stocks_mbbl"]].dropna(subset=["stocks_mbbl"])
        df = df.set_index("date").sort_index()

    except Exception:
        logger.warning(
            "EIA API v2 request failed; trying CSV fallback ...", exc_info=True
        )
        df = _fetch_eia_xls_fallback(start, end)

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info("Cached EIA stocks to %s  (%d rows)", cache_path, len(df))
    return df


def _fetch_eia_xls_fallback(start: str, end: str) -> pd.DataFrame:
    """Fallback: download Cushing stocks XLS directly from eia.gov (no key)."""
    import io

    logger.info("Downloading EIA XLS from %s", _EIA_XLS_URL)
    resp = requests.get(_EIA_XLS_URL, timeout=30)
    resp.raise_for_status()

    # The data is on the "Data 1" sheet; first two rows are headers
    raw = pd.read_excel(
        io.BytesIO(resp.content),
        sheet_name="Data 1",
        skiprows=2,
        engine="xlrd",
    )
    raw.columns = ["date", "stocks_mbbl"]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["stocks_mbbl"] = pd.to_numeric(raw["stocks_mbbl"], errors="coerce")
    raw = raw.dropna(subset=["date", "stocks_mbbl"])
    raw = raw.set_index("date").sort_index()

    # Filter to requested date range
    raw = raw.loc[start:end]
    return raw


# ---------------------------------------------------------------------------
# WTI front-month futures prices (via yfinance)
# ---------------------------------------------------------------------------


def fetch_wti_prices(
    start: str = DATA_START,
    end: str = DATA_END,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch daily WTI front-month futures (CL=F) close prices.

    Parameters
    ----------
    start, end : str
        ISO date strings.
    cache_path : str or Path or None
        Parquet cache location. Defaults to ``data/eia/wti_prices.parquet``.

    Returns
    -------
    pd.DataFrame
        Columns: ``wti_close`` (USD/bbl).
        Index: ``DatetimeIndex`` named ``date``.
    """
    if cache_path is None:
        cache_path = Path(EIA_DIR) / "wti_prices.parquet"
    cache_path = Path(cache_path)

    if cache_path.exists():
        logger.info("Loading cached WTI prices from %s", cache_path)
        return pd.read_parquet(cache_path)

    ticker = WTI_FUTURES_TICKER
    logger.info("Fetching yfinance ticker %s ...", ticker)

    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if data.empty:
        logger.warning("No data returned for ticker %s", ticker)
        return pd.DataFrame(columns=["wti_close"])

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    df = close.to_frame(name="wti_close")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.dropna().sort_index()

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info("Cached WTI prices to %s  (%d rows)", cache_path, len(df))
    return df


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------


def align_sar_eia(
    sar_df: pd.DataFrame,
    eia_df: pd.DataFrame,
    tolerance_days: int = 3,
) -> pd.DataFrame:
    """Merge SAR observations with nearest EIA weekly report.

    Parameters
    ----------
    sar_df : pd.DataFrame
        Must have a ``DatetimeIndex`` (or a ``date`` column).
    eia_df : pd.DataFrame
        EIA stocks with ``DatetimeIndex``.
    tolerance_days : int
        Maximum number of days between SAR and EIA observation for a match.

    Returns
    -------
    pd.DataFrame
        Merged on nearest date within tolerance.
    """
    sar = sar_df.copy()
    eia = eia_df.copy()

    # Ensure DatetimeIndex
    if "date" in sar.columns:
        sar = sar.set_index("date")
    if "date" in eia.columns:
        eia = eia.set_index("date")

    sar = sar.sort_index()
    eia = eia.sort_index()

    merged = pd.merge_asof(
        sar,
        eia,
        left_index=True,
        right_index=True,
        tolerance=pd.Timedelta(days=tolerance_days),
        direction="nearest",
    )
    return merged


def split_train_test(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Date-based train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``DatetimeIndex``.
    train_end : str
        Last date (inclusive) for the training set.

    Returns
    -------
    (train_df, test_df)
    """
    cutoff = pd.Timestamp(train_end)
    train = df[df.index <= cutoff]
    test = df[df.index > cutoff]
    return train, test


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_summary(name: str, df: pd.DataFrame) -> None:
    """Print summary statistics for a DataFrame."""
    if df.empty:
        print(f"\n  {name}: NO DATA")
        return
    print(f"\n  {name}:")
    print(f"    Date range : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"    Observations: {len(df)}")
    for col in df.columns:
        print(f"    {col}  min={df[col].min():.2f}  max={df[col].max():.2f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("  Cushing Ground Truth Data Fetch")
    print("=" * 60)

    # --- EIA Cushing stocks ---
    eia_df = fetch_eia_stocks()
    _print_summary("EIA Cushing Crude Stocks (thousand bbl)", eia_df)

    # --- WTI futures prices ---
    wti_df = fetch_wti_prices()
    _print_summary("WTI Front-Month Futures (USD/bbl)", wti_df)

    # --- Quick train/test split demo ---
    if not eia_df.empty:
        train, test = split_train_test(eia_df)
        print(f"\n  Train/test split (cutoff {TRAIN_END}):")
        print(f"    Train: {len(train)} rows")
        print(f"    Test : {len(test)} rows")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
