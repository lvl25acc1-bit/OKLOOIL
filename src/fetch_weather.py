"""Fetch ERA5 weather data for Cushing, Oklahoma via Open-Meteo Archive API.

Open-Meteo provides free, no-auth access to ERA5 reanalysis data.
We fetch hourly weather variables relevant to SAR backscatter correction:
  - wind_speed_10m  : 10-m wind speed (km/h from API, stored as m/s)
  - temperature_2m  : 2-m air temperature (deg C)
  - precipitation   : hourly precipitation (mm)
  - soil_moisture_0_to_7cm : top-layer volumetric soil moisture (m3/m3)

Data is chunked by year to respect API date-range limits, cached as parquet.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CUSHING_LAT = 35.985
CUSHING_LON = -96.77  # West longitude is negative
START_DATE = "2017-01-01"
END_DATE = "2026-03-01"

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "wind_speed_10m",
    "temperature_2m",
    "precipitation",
    "soil_moisture_0_to_7cm",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_HOURLY = PROJECT_ROOT / "data" / "era5" / "cushing_weather_hourly.parquet"
CACHE_DAILY = PROJECT_ROOT / "data" / "era5" / "cushing_weather_daily.parquet"
FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "cushing_weather_overview.png"


# ---------------------------------------------------------------------------
# Year-chunking helper (from Port Volume project)
# ---------------------------------------------------------------------------
def _year_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split a date range into per-year chunks for Open-Meteo API limits."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks = []
    current = start
    while current <= end:
        year_end = min(pd.Timestamp(f"{current.year}-12-31"), end)
        chunks.append((current.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d")))
        current = pd.Timestamp(f"{current.year + 1}-01-01")
    return chunks


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------
def fetch_cushing_weather(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    cache_path: Path = CACHE_HOURLY,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch hourly weather for Cushing from Open-Meteo and cache as parquet.

    Returns a DataFrame indexed by datetime (UTC) with weather columns.
    """
    if cache_path.exists() and not force:
        logger.info("Loading cached hourly weather from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching Cushing weather from Open-Meteo: %s to %s", start_date, end_date)
    hourly_var_str = ",".join(HOURLY_VARS)

    all_chunks: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _year_chunks(start_date, end_date):
        logger.info("  chunk %s  ->  %s", chunk_start, chunk_end)
        params = {
            "latitude": CUSHING_LAT,
            "longitude": CUSHING_LON,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "hourly": hourly_var_str,
            "timezone": "UTC",
        }
        try:
            resp = requests.get(BASE_URL, params=params, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("API request failed for %s-%s: %s", chunk_start, chunk_end, exc)
            continue

        data = resp.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            logger.warning("No hourly data returned for %s-%s", chunk_start, chunk_end)
            continue

        chunk_df = pd.DataFrame({"time": pd.to_datetime(times, utc=True)})
        for var in HOURLY_VARS:
            chunk_df[var] = hourly.get(var)

        all_chunks.append(chunk_df)

    if not all_chunks:
        raise RuntimeError("No weather data returned from Open-Meteo for any chunk")

    df = pd.concat(all_chunks, ignore_index=True).sort_values("time").reset_index(drop=True)

    # Convert wind speed from km/h to m/s
    if "wind_speed_10m" in df.columns:
        df["wind_speed_10m_mps"] = df["wind_speed_10m"] / 3.6
        df = df.drop(columns=["wind_speed_10m"])
        df = df.rename(columns={"wind_speed_10m_mps": "wind_speed_10m"})

    df = df.set_index("time")

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info("Cached hourly weather: %d rows -> %s", len(df), cache_path)

    return df


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------
def build_daily(hourly_df: pd.DataFrame, cache_path: Path = CACHE_DAILY) -> pd.DataFrame:
    """Aggregate hourly weather to daily stats and cache as parquet."""
    daily = hourly_df.resample("D").agg(
        wind_speed_10m_mean=("wind_speed_10m", "mean"),
        wind_speed_10m_max=("wind_speed_10m", "max"),
        temperature_2m_mean=("temperature_2m", "mean"),
        temperature_2m_min=("temperature_2m", "min"),
        temperature_2m_max=("temperature_2m", "max"),
        precipitation_sum=("precipitation", "sum"),
        soil_moisture_mean=("soil_moisture_0_to_7cm", "mean"),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(cache_path)
    logger.info("Cached daily weather: %d rows -> %s", len(daily), cache_path)
    return daily


# ---------------------------------------------------------------------------
# Interpolate to SAR timestamps
# ---------------------------------------------------------------------------
def get_weather_for_timestamps(
    timestamps: list,
    hourly_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return weather conditions at each SAR scene timestamp.

    Performs linear interpolation between bracketing hourly observations.

    Parameters
    ----------
    timestamps : list of datetime-like
        SAR acquisition times.
    hourly_df : DataFrame, optional
        Pre-loaded hourly weather. If None, loads from cache.

    Returns
    -------
    DataFrame with timestamp index and interpolated weather columns.
    """
    if hourly_df is None:
        if not CACHE_HOURLY.exists():
            raise FileNotFoundError(
                f"No cached hourly data at {CACHE_HOURLY}. Run fetch first."
            )
        hourly_df = pd.read_parquet(CACHE_HOURLY)
        if "time" in hourly_df.columns:
            hourly_df = hourly_df.set_index("time")

    weather_cols = [c for c in hourly_df.columns if c != "time"]

    results = []
    for ts in timestamps:
        ts_pd = pd.Timestamp(ts)
        if ts_pd.tzinfo is None:
            ts_pd = ts_pd.tz_localize("UTC")

        row = {"timestamp": ts_pd}
        before = hourly_df.index[hourly_df.index <= ts_pd]
        after = hourly_df.index[hourly_df.index >= ts_pd]

        if len(before) > 0 and len(after) > 0:
            t0 = before[-1]
            t1 = after[0]
            if t0 == t1:
                for col in weather_cols:
                    row[col] = hourly_df.loc[t0, col]
            else:
                frac = (ts_pd - t0).total_seconds() / (t1 - t0).total_seconds()
                for col in weather_cols:
                    v0 = hourly_df.loc[t0, col]
                    v1 = hourly_df.loc[t1, col]
                    if pd.notna(v0) and pd.notna(v1):
                        row[col] = v0 + frac * (v1 - v0)
                    else:
                        row[col] = np.nan
        else:
            for col in weather_cols:
                row[col] = np.nan

        results.append(row)

    return pd.DataFrame(results).set_index("timestamp")


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------
def plot_weather_overview(
    hourly_df: pd.DataFrame,
    out_path: Path = FIGURE_PATH,
) -> None:
    """Plot wind speed and precipitation time series."""
    # Resample to daily for cleaner visuals
    daily = hourly_df.resample("D").agg({
        "wind_speed_10m": "mean",
        "precipitation": "sum",
    })

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Wind speed
    ax = axes[0]
    ax.plot(daily.index, daily["wind_speed_10m"], color="steelblue", linewidth=0.6, alpha=0.8)
    ax.set_ylabel("Wind speed 10m (m/s)")
    ax.set_title("Cushing, OK — ERA5 Weather (Open-Meteo)")
    ax.grid(True, alpha=0.3)

    # Precipitation
    ax = axes[1]
    ax.bar(daily.index, daily["precipitation"], width=1.0, color="teal", alpha=0.7)
    ax.set_ylabel("Daily precipitation (mm)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved weather overview plot -> %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def print_summary(hourly_df: pd.DataFrame) -> None:
    """Print a summary of the fetched weather data."""
    print("\n" + "=" * 60)
    print("CUSHING WEATHER DATA SUMMARY")
    print("=" * 60)
    print(f"Date range : {hourly_df.index.min()} -> {hourly_df.index.max()}")
    print(f"Total rows : {len(hourly_df):,}")
    print()

    ws = hourly_df["wind_speed_10m"]
    print(f"Wind speed 10m (m/s):")
    print(f"  mean = {ws.mean():.2f},  std = {ws.std():.2f},  "
          f"min = {ws.min():.2f},  max = {ws.max():.2f}")

    precip = hourly_df["precipitation"]
    precip_hours = (precip > 0).sum()
    print(f"Precipitation:")
    print(f"  hours with precip = {precip_hours:,} / {len(precip):,}  "
          f"({100 * precip_hours / len(precip):.1f}%)")
    print(f"  hourly mean (when > 0) = {precip[precip > 0].mean():.2f} mm")

    temp = hourly_df["temperature_2m"]
    print(f"Temperature 2m (C):")
    print(f"  mean = {temp.mean():.1f},  std = {temp.std():.1f},  "
          f"min = {temp.min():.1f},  max = {temp.max():.1f}")

    sm = hourly_df.get("soil_moisture_0_to_7cm")
    if sm is not None and sm.notna().any():
        print(f"Soil moisture 0-7cm (m3/m3):")
        print(f"  mean = {sm.mean():.4f},  std = {sm.std():.4f}")
    else:
        print("Soil moisture: not available in this dataset")

    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Step 1: Fetch hourly data
    hourly = fetch_cushing_weather(force="--force" in sys.argv)

    # Step 2: Print summary
    print_summary(hourly)

    # Step 3: Build daily aggregation
    daily = build_daily(hourly)

    # Step 4: Diagnostic plot
    plot_weather_overview(hourly)

    print(f"\nOutputs:")
    print(f"  Hourly parquet : {CACHE_HOURLY}")
    print(f"  Daily parquet  : {CACHE_DAILY}")
    print(f"  Overview plot  : {FIGURE_PATH}")
