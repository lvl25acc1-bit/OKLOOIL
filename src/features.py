"""Feature engineering for Cushing SAR oil storage estimation.

Transforms per-scene backscatter statistics into weekly analysis-ready features
aligned to EIA reporting cadence (Wednesday). Includes wind correction via RLM,
seasonal decomposition, and control-difference computation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL

from src.config import (
    FEATURES_DIR,
    EIA_DIR,
    ERA5_DIR,
    MIN_OBSERVATIONS,
)

logger = logging.getLogger(__name__)

# ── Backscatter stat columns ──────────────────────────────────────────────────
BACKSCATTER_STAT_COLS = [
    "mean_db",
    "median_db",
    "std_db",
    "p10_db",
    "p25_db",
    "p75_db",
    "p90_db",
    "skewness",
    "kurtosis",
    "bright_pixel_ratio",
]

SCENE_KIND_SUFFIXES = {
    "rtc_vv": "_rtc_vv",
    "vv_crop": "_vv_crop",
}


def infer_scene_kind(scene_id: object) -> str:
    """Infer the source product family from a scene identifier."""
    if scene_id is None:
        return "unknown"

    scene_str = str(scene_id)
    for scene_kind, suffix in SCENE_KIND_SUFFIXES.items():
        if scene_str.endswith(suffix):
            return scene_kind
    return "unknown"


def ensure_scene_kind_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure backscatter data carries a scene_kind provenance column."""
    out = df.copy()
    inferred = out["scene_id"].map(infer_scene_kind) if "scene_id" in out.columns else pd.Series("unknown", index=out.index)

    if "scene_kind" in out.columns:
        out["scene_kind"] = out["scene_kind"].fillna(inferred).astype(str)
    else:
        out["scene_kind"] = inferred.astype(str)

    return out


def validate_scene_kinds(df: pd.DataFrame, allow_mixed_scene_kinds: bool = False) -> list[str]:
    """Validate that the backscatter table does not mix incompatible scene kinds."""
    if "scene_kind" not in df.columns:
        return []

    scene_kinds = sorted({kind for kind in df["scene_kind"].dropna().astype(str) if kind != "unknown"})
    if not scene_kinds:
        logger.warning("Backscatter data has no recognized scene_kind values")
        return []

    logger.info("Backscatter scene kinds present: %s", ", ".join(scene_kinds))
    if len(scene_kinds) > 1 and not allow_mixed_scene_kinds:
        raise ValueError(
            "Mixed scene kinds detected in backscatter input: "
            f"{scene_kinds}. Rebuild with a single product family or pass allow_mixed_scene_kinds=True."
        )

    return scene_kinds


# ── 1. Load backscatter ──────────────────────────────────────────────────────

def load_backscatter(path: str | Path | None = None) -> pd.DataFrame:
    """Load roi_backscatter.parquet, parse timestamps, set DatetimeIndex.

    Parameters
    ----------
    path : Path to the parquet file.  Defaults to
        ``data/features/roi_backscatter.parquet``.

    Returns
    -------
    DataFrame with DatetimeIndex (UTC).
    """
    if path is None:
        path = Path(FEATURES_DIR) / "roi_backscatter.parquet"
    path = Path(path)

    logger.info("Loading backscatter from %s", path)
    df = pd.read_parquet(path)
    df = ensure_scene_kind_column(df)

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

    logger.info("Loaded %d scenes spanning %s to %s", len(df), df.index.min(), df.index.max())
    return df


# ── 2. Weekly aggregation ────────────────────────────────────────────────────

def compute_weekly_features(
    df: pd.DataFrame,
    roi_name: str = "all_tanks",
    freq: str = "W-WED",
) -> pd.DataFrame:
    """Aggregate per-scene stats to weekly, aligned to Wednesday for EIA.

    Parameters
    ----------
    df : Backscatter DataFrame with DatetimeIndex.
    roi_name : Filter to this ROI.  Use ``"all"`` for no filter.
    freq : Resampling frequency string.

    Returns
    -------
    Weekly DataFrame with mean of each backscatter stat and n_scenes count.
    """
    subset = df.copy()
    if roi_name != "all" and "roi_name" in subset.columns:
        subset = subset[subset["roi_name"] == roi_name]

    if len(subset) == 0:
        logger.warning("No data for roi_name='%s'", roi_name)
        return pd.DataFrame()

    # Identify numeric stat columns present
    stat_cols = [c for c in BACKSCATTER_STAT_COLS if c in subset.columns]

    # Resample to weekly
    weekly = subset[stat_cols].resample(freq).mean()
    weekly["n_scenes"] = subset[stat_cols[0]].resample(freq).count()

    # Drop weeks with no observations
    weekly = weekly.dropna(subset=stat_cols, how="all")

    logger.info(
        "Weekly aggregation for '%s': %d weeks, median %.1f scenes/week",
        roi_name,
        len(weekly),
        weekly["n_scenes"].median(),
    )
    return weekly


# ── 3. First differences ─────────────────────────────────────────────────────

def add_first_differences(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Add delta columns: delta_{col} = col[t] - col[t-1].

    THIS IS THE MOST CRITICAL FEATURE.  First-differencing removes shared
    trends and exposes whether week-to-week changes in SAR track week-to-week
    changes in inventory.

    Parameters
    ----------
    df : Weekly feature DataFrame.
    columns : Columns to difference.  Defaults to all backscatter stat cols present.

    Returns
    -------
    DataFrame with added delta_* columns.
    """
    if columns is None:
        columns = [c for c in BACKSCATTER_STAT_COLS if c in df.columns]

    out = df.copy()
    for col in columns:
        out[f"delta_{col}"] = out[col].diff()

    logger.info("Added first differences for %d columns", len(columns))
    return out


# ── 4. Rolling features ──────────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and std for each column at each window size.

    Parameters
    ----------
    df : Weekly feature DataFrame.
    columns : Columns to compute rolling stats for.
    windows : Rolling window sizes in weeks.  Default ``[4, 8]``.

    Returns
    -------
    DataFrame with added rolling_mean_* and rolling_std_* columns.
    """
    if columns is None:
        columns = [c for c in BACKSCATTER_STAT_COLS if c in df.columns]
    if windows is None:
        windows = [4, 8]

    out = df.copy()
    for col in columns:
        for w in windows:
            out[f"rolling_mean_{col}_{w}w"] = out[col].rolling(window=w, min_periods=max(w // 2, 2)).mean()
            out[f"rolling_std_{col}_{w}w"] = out[col].rolling(window=w, min_periods=max(w // 2, 2)).std()

    logger.info("Added rolling features: %d columns x %d windows", len(columns), len(windows))
    return out


# ── 5. Seasonal decomposition ────────────────────────────────────────────────

def add_seasonal_decomposition(
    df: pd.DataFrame,
    column: str = "mean_db",
    period: int = 52,
) -> pd.DataFrame:
    """Use statsmodels STL to extract trend, seasonal, and residual components.

    Parameters
    ----------
    df : Weekly feature DataFrame.
    column : Column to decompose.
    period : Seasonal period in weeks (52 = annual).

    Returns
    -------
    DataFrame with added {column}_trend, {column}_seasonal, {column}_residual columns.
    """
    out = df.copy()

    if column not in out.columns:
        logger.warning("Column '%s' not found for STL decomposition", column)
        return out

    series = out[column].dropna()

    if len(series) < 2 * period:
        logger.warning(
            "Only %d observations for STL with period=%d; need >= %d. Skipping.",
            len(series), period, 2 * period,
        )
        out[f"{column}_trend"] = np.nan
        out[f"{column}_seasonal"] = np.nan
        out[f"{column}_residual"] = np.nan
        return out

    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    out.loc[series.index, f"{column}_trend"] = result.trend
    out.loc[series.index, f"{column}_seasonal"] = result.seasonal
    out.loc[series.index, f"{column}_residual"] = result.resid

    logger.info(
        "STL decomposition on '%s': trend var=%.4f, seasonal var=%.4f, resid var=%.4f",
        column,
        result.trend.var(),
        result.seasonal.var(),
        result.resid.var(),
    )
    return out


# ── 6. Wind correction via RLM ───────────────────────────────────────────────

def fit_wind_correction(
    df: pd.DataFrame,
    target_col: str = "mean_db",
    wind_col: str = "wind_speed_10m",
) -> tuple[object, pd.Series]:
    """Fit Robust Linear Model to remove weather/orbit confounds.

    Model: target ~ wind_speed + precipitation + is_ascending

    Uses HuberT M-estimator to down-weight outlier weeks.

    Parameters
    ----------
    df : DataFrame with target, wind, precipitation, and flight_direction columns.
    target_col : SAR backscatter column to correct.
    wind_col : Wind speed column name.

    Returns
    -------
    Tuple of (fitted RLM model, Series of residuals = wind-corrected values).
    """
    required = [target_col, wind_col]
    available = [c for c in required if c in df.columns]
    if len(available) < len(required):
        missing = set(required) - set(available)
        logger.warning("Missing columns for wind correction: %s. Returning raw values.", missing)
        return None, df[target_col].copy() if target_col in df.columns else pd.Series(dtype=float)

    work = df[[target_col]].copy()
    work[wind_col] = df[wind_col]

    # Precipitation (optional)
    if "precipitation" in df.columns:
        work["precipitation"] = df["precipitation"]
    else:
        work["precipitation"] = 0.0

    # Ascending flag (optional)
    if "flight_direction" in df.columns:
        work["is_ascending"] = (df["flight_direction"].str.upper() == "ASCENDING").astype(float)
    elif "is_ascending" in df.columns:
        work["is_ascending"] = df["is_ascending"].astype(float)
    else:
        work["is_ascending"] = 0.0

    work = work.dropna()

    if len(work) < 10:
        logger.warning("Too few observations (%d) for wind RLM; returning raw values.", len(work))
        return None, df[target_col].copy()

    X = work[[wind_col, "precipitation", "is_ascending"]]
    X = sm.add_constant(X)
    y = work[target_col]

    rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    result = rlm_model.fit()

    # Residuals = target - nuisance prediction (keeps mean level)
    residuals = pd.Series(np.nan, index=df.index, name="wind_corrected")
    residuals.loc[work.index] = result.resid + y.mean()

    # Log model fit
    ss_res = np.sum(result.resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    pseudo_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    logger.info(
        "Wind correction RLM: wind=%.3f, precip=%.3f, asc=%.3f, pseudo-R2=%.3f",
        result.params.get(wind_col, 0),
        result.params.get("precipitation", 0),
        result.params.get("is_ascending", 0),
        pseudo_r2,
    )

    return result, residuals


# ── 7. Control difference ────────────────────────────────────────────────────

def compute_control_difference(
    tank_df: pd.DataFrame,
    control_df: pd.DataFrame,
    col: str = "mean_db",
) -> pd.Series:
    """Compute tank_farm_db - control_grassland_db per week.

    This cancels atmospheric effects that are spatially uniform (weather,
    ionosphere, calibration drift).

    Parameters
    ----------
    tank_df : Weekly features for tank farm ROI.
    control_df : Weekly features for control grassland ROI.
    col : Backscatter column to difference.

    Returns
    -------
    Series named 'control_diff' with same index as tank_df.
    """
    # Align on shared index
    aligned_tank = tank_df[col].reindex(tank_df.index)
    aligned_ctrl = control_df[col].reindex(tank_df.index)

    diff = aligned_tank - aligned_ctrl
    diff.name = "control_diff"

    n_valid = diff.notna().sum()
    logger.info(
        "Control difference: %d weeks with valid values (of %d tank weeks)",
        n_valid,
        len(tank_df),
    )
    return diff


# ── 8. Build analysis-ready dataset ──────────────────────────────────────────

def build_analysis_ready(
    backscatter_path: str | Path | None = None,
    eia_path: str | Path | None = None,
    weather_path: str | Path | None = None,
    output_path: str | Path | None = None,
    control_output_path: str | Path | None = None,
    allow_mixed_scene_kinds: bool = False,
) -> pd.DataFrame:
    """Main pipeline: build the analysis-ready dataset.

    Steps:
    1. Load backscatter, compute weekly features for all_tanks and control_grassland
    2. Load EIA stocks, merge with SAR via pd.merge_asof (tolerance 3 days)
    3. Load weather, match to SAR timestamps
    4. Add first differences, rolling features, seasonal decomposition
    5. Fit wind correction
    6. Compute control difference
    7. Save to output_path

    Returns
    -------
    Analysis-ready DataFrame.
    """
    # Defaults
    if backscatter_path is None:
        backscatter_path = Path(FEATURES_DIR) / "roi_backscatter.parquet"
    if eia_path is None:
        eia_path = Path(EIA_DIR) / "cushing_stocks.parquet"
    if weather_path is None:
        weather_path = Path(ERA5_DIR) / "cushing_weather_hourly.parquet"
    if output_path is None:
        output_path = Path(FEATURES_DIR) / "analysis_ready.parquet"
    if control_output_path is None:
        control_output_path = Path(FEATURES_DIR) / "control_weekly.parquet"

    backscatter_path = Path(backscatter_path)
    eia_path = Path(eia_path)
    weather_path = Path(weather_path)
    output_path = Path(output_path)
    control_output_path = Path(control_output_path)

    # ── Step 1: Load backscatter, compute weekly features ─────────────────
    logger.info("Step 1: Loading backscatter and computing weekly features")
    bs = load_backscatter(backscatter_path)
    scene_kinds = validate_scene_kinds(bs, allow_mixed_scene_kinds=allow_mixed_scene_kinds)

    tank_weekly = compute_weekly_features(bs, roi_name="all_tanks", freq="W-WED")
    control_weekly = compute_weekly_features(bs, roi_name="control_grassland", freq="W-WED")

    if len(tank_weekly) == 0:
        raise ValueError("No tank weekly data produced. Check roi_name values in backscatter data.")

    # ── Step 2: Load EIA stocks, merge via merge_asof ─────────────────────
    logger.info("Step 2: Loading EIA stocks and merging")
    eia = pd.read_parquet(eia_path)

    # Ensure DatetimeIndex
    if "date" in eia.columns:
        eia["date"] = pd.to_datetime(eia["date"])
        eia = eia.set_index("date")
    if not isinstance(eia.index, pd.DatetimeIndex):
        eia.index = pd.to_datetime(eia.index)
    eia = eia.sort_index()

    # Timezone alignment: remove tz from both for merge_asof
    tank_idx = tank_weekly.index.tz_localize(None) if tank_weekly.index.tz else tank_weekly.index
    eia_idx = eia.index.tz_localize(None) if eia.index.tz else eia.index

    tank_for_merge = tank_weekly.copy()
    tank_for_merge.index = tank_idx

    eia_for_merge = eia.copy()
    eia_for_merge.index = eia_idx

    # merge_asof needs sorted indexes and a column (not index) for 'on'
    tank_for_merge = tank_for_merge.reset_index().rename(columns={"index": "week", "timestamp": "week"})
    if "week" not in tank_for_merge.columns:
        tank_for_merge = tank_for_merge.reset_index().rename(columns={tank_for_merge.index.name or "index": "week"})
    tank_for_merge["week"] = pd.to_datetime(tank_for_merge["week"]).dt.as_unit("ns")
    tank_for_merge = tank_for_merge.sort_values("week")

    eia_for_merge = eia_for_merge.reset_index()
    eia_for_merge.columns = ["eia_date"] + list(eia_for_merge.columns[1:])
    eia_for_merge["eia_date"] = pd.to_datetime(eia_for_merge["eia_date"]).dt.as_unit("ns")
    eia_for_merge = eia_for_merge.sort_values("eia_date")

    merged = pd.merge_asof(
        tank_for_merge,
        eia_for_merge,
        left_on="week",
        right_on="eia_date",
        tolerance=pd.Timedelta("3D"),
        direction="nearest",
    )

    merged = merged.set_index("week")
    merged.index.name = "timestamp"

    logger.info("Merged SAR+EIA: %d weeks with EIA match", merged["stocks_mbbl"].notna().sum())

    # ── Step 3: Load weather, match to SAR timestamps ─────────────────────
    logger.info("Step 3: Loading weather data")
    try:
        weather = pd.read_parquet(weather_path)
        if not isinstance(weather.index, pd.DatetimeIndex):
            if "time" in weather.columns:
                weather = weather.set_index("time")
            elif "timestamp" in weather.columns:
                weather = weather.set_index("timestamp")
            weather.index = pd.to_datetime(weather.index)

        # Remove tz for alignment
        if weather.index.tz is not None:
            weather.index = weather.index.tz_localize(None)

        weather = weather.sort_index()

        # Resample weather to weekly to match SAR
        weather_cols = [c for c in ["wind_speed_10m", "temperature_2m", "precipitation", "soil_moisture_0_to_7cm"]
                        if c in weather.columns]
        weather_weekly = weather[weather_cols].resample("W-WED").mean()

        # Merge weather
        for col in weather_cols:
            if col in weather_weekly.columns:
                merged[col] = weather_weekly[col].reindex(merged.index)

        logger.info("Weather merged: %d columns", len(weather_cols))
    except FileNotFoundError:
        logger.warning("Weather file not found at %s; skipping weather features.", weather_path)
    except Exception as exc:
        logger.warning("Error loading weather: %s; skipping.", exc)

    # ── Step 4: First differences, rolling features, seasonal decomposition
    logger.info("Step 4: Adding derived features")
    merged = add_first_differences(merged)
    merged = add_rolling_features(merged)
    merged = add_seasonal_decomposition(merged, column="mean_db", period=52)

    # ── Step 5: Wind correction ───────────────────────────────────────────
    logger.info("Step 5: Fitting wind correction")
    wind_model, wind_corrected = fit_wind_correction(merged, target_col="mean_db")
    merged["wind_corrected"] = wind_corrected

    # Add first difference of wind-corrected
    merged["delta_wind_corrected"] = merged["wind_corrected"].diff()

    # ── Step 6: Control difference ────────────────────────────────────────
    logger.info("Step 6: Computing control difference")
    if len(control_weekly) > 0:
        # Ensure matching tz-naive indexes for control difference
        ctrl = control_weekly.copy()
        if ctrl.index.tz is not None:
            ctrl.index = ctrl.index.tz_localize(None)
        tank_for_ctrl = tank_weekly.copy()
        if tank_for_ctrl.index.tz is not None:
            tank_for_ctrl.index = tank_for_ctrl.index.tz_localize(None)
        ctrl_diff = compute_control_difference(tank_for_ctrl, ctrl)
        # Align to merged index
        merged_idx_naive = merged.index.tz_localize(None) if merged.index.tz else merged.index
        merged["control_diff"] = ctrl_diff.reindex(merged_idx_naive).values
        merged["delta_control_diff"] = merged["control_diff"].diff()
    else:
        logger.warning("No control ROI data; skipping control difference.")
        merged["control_diff"] = np.nan
        merged["delta_control_diff"] = np.nan

    # Also add delta of seasonal residual
    if "mean_db_residual" in merged.columns:
        merged["delta_seasonal_residual"] = merged["mean_db_residual"].diff()

    # EIA first difference
    if "stocks_mbbl" in merged.columns:
        merged["delta_stocks_mbbl"] = merged["stocks_mbbl"].diff()

    # ── Step 7: Save ──────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path)
    logger.info("Saved analysis-ready dataset: %s (%d rows, %d columns)", output_path, len(merged), len(merged.columns))
    if scene_kinds:
        logger.info("Analysis-ready dataset built from scene_kind=%s", ",".join(scene_kinds))

    # Also save control weekly for null tests
    if len(control_weekly) > 0:
        control_output_path.parent.mkdir(parents=True, exist_ok=True)
        control_weekly.to_parquet(control_output_path)
        logger.info("Saved control weekly: %s", control_output_path)

    return merged


# ── CLI entry point ───────────────────────────────────────────────────────────

def build_cli_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for feature engineering."""
    parser = argparse.ArgumentParser(description="Build the weekly analysis-ready SAR dataset.")
    parser.add_argument(
        "--backscatter-path",
        type=Path,
        default=Path(FEATURES_DIR) / "roi_backscatter.parquet",
        help="Input parquet from preprocess.py.",
    )
    parser.add_argument(
        "--eia-path",
        type=Path,
        default=Path(EIA_DIR) / "cushing_stocks.parquet",
        help="Input EIA stocks parquet.",
    )
    parser.add_argument(
        "--weather-path",
        type=Path,
        default=Path(ERA5_DIR) / "cushing_weather_hourly.parquet",
        help="Input weather parquet.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(FEATURES_DIR) / "analysis_ready.parquet",
        help="Output analysis-ready parquet.",
    )
    parser.add_argument(
        "--control-output-path",
        type=Path,
        default=Path(FEATURES_DIR) / "control_weekly.parquet",
        help="Output control-weekly parquet used by the analysis battery.",
    )
    parser.add_argument(
        "--allow-mixed-scene-kinds",
        action="store_true",
        help="Allow mixed scene kinds in the input backscatter parquet. Default is to fail closed.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = build_cli_parser().parse_args()

    result = build_analysis_ready(
        backscatter_path=args.backscatter_path,
        eia_path=args.eia_path,
        weather_path=args.weather_path,
        output_path=args.output_path,
        control_output_path=args.control_output_path,
        allow_mixed_scene_kinds=args.allow_mixed_scene_kinds,
    )

    print(f"\nAnalysis-ready dataset shape: {result.shape}")
    print(f"\nColumns:\n{list(result.columns)}")
    print(f"\nDate range: {result.index.min()} to {result.index.max()}")
    print(f"\nEIA coverage: {result['stocks_mbbl'].notna().sum()} / {len(result)} weeks")
    print(f"\nFirst few rows:")
    print(result.head())
