"""Statistical test battery for Cushing SAR oil storage estimation.

Runs ALL 8 tests on SAR-vs-EIA relationships. A signal that fails these
tests is spurious. From Santos: rho=0.597 was spurious -- first-diff was 0.010.

Tests:
  1. Raw correlation (levels)          -- Pearson + Spearman
  2. First-difference correlation      -- THE KILL TEST
  3. Detrended correlation             -- STL residuals
  4. Within-year stability             -- sign/magnitude per year
  5. Out-of-time validation            -- train 2017-2023, test 2024-2025
  6. Granger causality                 -- SAR should lead or be contemporaneous
  7. Wind confound (partial corr)      -- relationship must survive wind removal
  8. Multiple comparison correction    -- BH FDR across all features

Plus: control null test (grassland ROI must show NO correlation).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm

from src.config import (
    TRAIN_END,
    TEST_START,
    FDR_THRESHOLD,
    KILL_THRESHOLD,
    MIN_OBSERVATIONS,
    FEATURES_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
)

logger = logging.getLogger(__name__)

# ── Feature columns to test ──────────────────────────────────────────────────
SAR_FEATURE_COLS = [
    "mean_db",
    "median_db",
    "std_db",
    "bright_pixel_ratio",
    "wind_corrected",
    "control_diff",
    "mean_db_residual",
]

DELTA_KILL_COLS = [
    "delta_mean_db",
    "delta_control_diff",
    "delta_wind_corrected",
    "delta_seasonal_residual",
]


# ── Helper: safe correlation ─────────────────────────────────────────────────

def _safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> tuple[float, float]:
    """Compute correlation with NaN safety and minimum observation check."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) < MIN_OBSERVATIONS:
        return np.nan, np.nan
    if np.std(x_clean) == 0 or np.std(y_clean) == 0:
        return np.nan, np.nan
    if method == "pearson":
        return stats.pearsonr(x_clean, y_clean)
    elif method == "spearman":
        return stats.spearmanr(x_clean, y_clean)
    else:
        raise ValueError(f"Unknown method: {method}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Raw Correlation (Levels)
# ═════════════════════════════════════════════════════════════════════════════

def test_raw_correlation(
    df: pd.DataFrame,
    sar_cols: list[str],
    eia_col: str = "stocks_mbbl",
) -> pd.DataFrame:
    """Test 1: Pearson and Spearman correlation on levels.

    WARNING: High correlation on levels can be spurious due to shared trends.
    This is a necessary but NOT sufficient condition.

    Parameters
    ----------
    df : Analysis-ready DataFrame.
    sar_cols : SAR feature column names to test.
    eia_col : EIA inventory column.

    Returns
    -------
    DataFrame with correlation results per SAR feature.
    """
    records = []
    eia = df[eia_col].values if eia_col in df.columns else None
    if eia is None:
        logger.error("EIA column '%s' not found", eia_col)
        return pd.DataFrame()

    for col in sar_cols:
        if col not in df.columns:
            continue

        sar = df[col].values
        pr, pp = _safe_corr(sar, eia, "pearson")
        sr, sp = _safe_corr(sar, eia, "spearman")

        mask = np.isfinite(sar) & np.isfinite(eia)
        n = int(mask.sum())

        records.append({
            "test": "raw_correlation",
            "feature": col,
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
            "n_obs": n,
        })

    result = pd.DataFrame(records)
    logger.info("Test 1 (raw correlation): %d features tested", len(result))
    return result


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: First-Difference Correlation — THE KILL TEST
# ═════════════════════════════════════════════════════════════════════════════

def test_first_difference(
    df: pd.DataFrame,
    sar_cols: list[str],
    eia_col: str = "stocks_mbbl",
) -> pd.DataFrame:
    """Test 2: Correlate delta(SAR) with delta(EIA).

    THIS IS THE KILL TEST.  If ALL features have |rho| < 0.1, the signal
    is dead — levels correlation was spurious (shared trend / confound).

    Parameters
    ----------
    df : Analysis-ready DataFrame (must contain delta_* columns).
    sar_cols : Delta SAR column names (e.g. delta_mean_db).
    eia_col : EIA column (will use delta_stocks_mbbl if available).

    Returns
    -------
    DataFrame with first-difference correlation results.
    """
    # Use delta of EIA
    delta_eia_col = f"delta_{eia_col}" if f"delta_{eia_col}" in df.columns else eia_col
    if delta_eia_col not in df.columns:
        # Compute on the fly
        df = df.copy()
        df[delta_eia_col] = df[eia_col].diff()

    eia = df[delta_eia_col].values

    records = []
    seen_delta_cols = set()
    for col in sar_cols:
        dcol = col if col.startswith("delta_") else f"delta_{col}"
        if dcol in seen_delta_cols:
            continue
        seen_delta_cols.add(dcol)
        if dcol not in df.columns:
            continue

        sar = df[dcol].values
        pr, pp = _safe_corr(sar, eia, "pearson")
        sr, sp = _safe_corr(sar, eia, "spearman")

        mask = np.isfinite(sar) & np.isfinite(eia)
        n = int(mask.sum())

        records.append({
            "test": "first_difference",
            "feature": dcol,
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
            "n_obs": n,
        })

    result = pd.DataFrame(records)

    if len(result) > 0:
        max_rho = result["pearson_r"].abs().max()
        logger.info(
            "Test 2 (first-difference): max |rho| = %.4f (%s threshold=%.2f)",
            max_rho,
            "ABOVE" if max_rho >= KILL_THRESHOLD else "BELOW",
            KILL_THRESHOLD,
        )
    return result


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Detrended Correlation (STL Residuals)
# ═════════════════════════════════════════════════════════════════════════════

def test_detrended(
    df: pd.DataFrame,
    sar_cols: list[str],
    eia_col: str = "stocks_mbbl",
) -> pd.DataFrame:
    """Test 3: Remove STL trend+seasonal from both series, correlate residuals.

    If the correlation disappears after detrending, it was driven by shared
    macro trends (e.g. both SAR and stocks declined 2020-2023).

    Parameters
    ----------
    df : Analysis-ready DataFrame.
    sar_cols : SAR feature columns to test.
    eia_col : EIA inventory column.

    Returns
    -------
    DataFrame with detrended correlation results.
    """
    # Detrend EIA
    eia_series = df[eia_col].dropna()
    eia_resid = _detrend_stl(eia_series, period=52)

    records = []
    for col in sar_cols:
        if col not in df.columns:
            continue

        sar_series = df[col].dropna()
        sar_resid = _detrend_stl(sar_series, period=52)

        if sar_resid is None or eia_resid is None:
            continue

        # Align
        combined = pd.DataFrame({"sar": sar_resid, "eia": eia_resid}).dropna()
        if len(combined) < MIN_OBSERVATIONS:
            continue

        pr, pp = stats.pearsonr(combined["sar"].values, combined["eia"].values)
        sr, sp = stats.spearmanr(combined["sar"].values, combined["eia"].values)

        records.append({
            "test": "detrended",
            "feature": col,
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
            "n_obs": len(combined),
        })

    result = pd.DataFrame(records)
    logger.info("Test 3 (detrended): %d features tested", len(result))
    return result


def _detrend_stl(series: pd.Series, period: int = 52) -> Optional[pd.Series]:
    """Remove trend and seasonal via STL, return residuals."""
    series = series.dropna()
    if len(series) < 2 * period:
        # Fall back to simple linear detrend
        x = np.arange(len(series), dtype=float)
        y = series.values.astype(float)
        mask = np.isfinite(y)
        if mask.sum() < MIN_OBSERVATIONS:
            return None
        slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])
        resid = y - (slope * x + intercept)
        return pd.Series(resid, index=series.index, name=series.name)

    try:
        stl = STL(series, period=period, robust=True)
        result = stl.fit()
        return result.resid
    except Exception as exc:
        logger.warning("STL failed for '%s': %s; using linear detrend", series.name, exc)
        x = np.arange(len(series), dtype=float)
        y = series.values.astype(float)
        slope, intercept, _, _, _ = stats.linregress(x, y)
        resid = y - (slope * x + intercept)
        return pd.Series(resid, index=series.index, name=series.name)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Within-Year Stability
# ═════════════════════════════════════════════════════════════════════════════

def test_within_year_stability(
    df: pd.DataFrame,
    sar_col: str,
    eia_col: str = "stocks_mbbl",
) -> pd.DataFrame:
    """Test 4: Run correlation per calendar year. Check sign/magnitude stability.

    If the sign flips across years, the relationship is unstable and
    likely driven by confounds that vary year-to-year.

    Parameters
    ----------
    df : Analysis-ready DataFrame.
    sar_col : Single SAR feature column to test.
    eia_col : EIA column.

    Returns
    -------
    DataFrame with per-year correlation and stability assessment.
    """
    records = []
    df_work = df[[sar_col, eia_col]].dropna()

    years = sorted(df_work.index.year.unique())
    for year in years:
        subset = df_work[df_work.index.year == year]
        if len(subset) < 10:
            continue

        pr, pp = stats.pearsonr(subset[sar_col].values, subset[eia_col].values)
        records.append({
            "test": "within_year_stability",
            "feature": sar_col,
            "year": int(year),
            "pearson_r": pr,
            "pearson_p": pp,
            "n_obs": len(subset),
        })

    result = pd.DataFrame(records)

    if len(result) > 0:
        signs = np.sign(result["pearson_r"].values)
        sign_flips = int(np.sum(np.abs(np.diff(signs)) > 0))
        all_same_sign = bool(np.all(signs == signs[0]))
        result["sign_flips"] = sign_flips
        result["all_same_sign"] = all_same_sign

        logger.info(
            "Test 4 (year stability) for '%s': %d years, %d sign flips, stable=%s",
            sar_col, len(result), sign_flips, all_same_sign,
        )

    return result


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Out-of-Time Validation
# ═════════════════════════════════════════════════════════════════════════════

def test_out_of_time(
    df: pd.DataFrame,
    sar_cols: list[str],
    eia_col: str = "stocks_mbbl",
    train_end: str | None = None,
) -> pd.DataFrame:
    """Test 5: Fit OLS on training period, predict test period. Report R-squared.

    R-squared must be positive. Negative R-squared means the model is worse
    than predicting the mean — a clear sign of overfitting or spurious correlation.

    Parameters
    ----------
    df : Analysis-ready DataFrame.
    sar_cols : SAR features to use as predictors.
    eia_col : EIA target column.
    train_end : End of training period (default from config.TRAIN_END).

    Returns
    -------
    DataFrame with out-of-time R-squared results.
    """
    if train_end is None:
        train_end = TRAIN_END
    test_start = TEST_START

    train = df.loc[:train_end].dropna(subset=[eia_col])
    test = df.loc[test_start:].dropna(subset=[eia_col])

    records = []
    for col in sar_cols:
        if col not in df.columns:
            continue

        train_clean = train[[col, eia_col]].dropna()
        test_clean = test[[col, eia_col]].dropna()

        if len(train_clean) < MIN_OBSERVATIONS or len(test_clean) < 10:
            continue

        X_train = sm.add_constant(train_clean[col].values)
        y_train = train_clean[eia_col].values
        X_test = sm.add_constant(test_clean[col].values)
        y_test = test_clean[eia_col].values

        try:
            model = sm.OLS(y_train, X_train).fit()
            y_pred = model.predict(X_test)

            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            r2_oot = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            records.append({
                "test": "out_of_time",
                "feature": col,
                "train_r2": float(model.rsquared),
                "test_r2": float(r2_oot),
                "train_n": len(train_clean),
                "test_n": len(test_clean),
                "positive_r2": r2_oot > 0,
            })
        except Exception as exc:
            logger.warning("OLS failed for '%s': %s", col, exc)

    result = pd.DataFrame(records)
    if len(result) > 0:
        best = result.loc[result["test_r2"].idxmax()]
        logger.info(
            "Test 5 (out-of-time): best feature='%s', train R2=%.3f, test R2=%.3f",
            best["feature"], best["train_r2"], best["test_r2"],
        )
    return result


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: Granger Causality
# ═════════════════════════════════════════════════════════════════════════════

def test_granger_causality(
    df: pd.DataFrame,
    sar_col: str,
    eia_col: str = "stocks_mbbl",
    max_lag: int = 4,
) -> dict:
    """Test 6: Granger causality in both directions.

    SAR should Granger-cause EIA (SAR leads) or at least be contemporaneous.
    If EIA Granger-causes SAR but not vice versa, SAR is just reacting to
    price-driven operational changes.

    Parameters
    ----------
    df : Analysis-ready DataFrame.
    sar_col : SAR feature column.
    eia_col : EIA column.
    max_lag : Maximum lag in weeks to test.

    Returns
    -------
    Dict with Granger test results in both directions.
    """
    work = df[[sar_col, eia_col]].dropna()

    if len(work) < MIN_OBSERVATIONS + max_lag:
        logger.warning("Insufficient data for Granger test: %d rows", len(work))
        return {"test": "granger_causality", "feature": sar_col, "error": "insufficient_data"}

    results = {"test": "granger_causality", "feature": sar_col}

    # Direction 1: SAR -> EIA (SAR Granger-causes EIA)
    try:
        gc_sar_to_eia = grangercausalitytests(
            work[[eia_col, sar_col]].values, maxlag=max_lag, verbose=False
        )
        sar_to_eia_pvals = {}
        for lag in range(1, max_lag + 1):
            f_test = gc_sar_to_eia[lag][0]
            # Use ssr_ftest p-value
            sar_to_eia_pvals[f"lag_{lag}"] = float(f_test["ssr_ftest"][1])
        results["sar_causes_eia"] = sar_to_eia_pvals
        results["sar_causes_eia_min_p"] = min(sar_to_eia_pvals.values())
    except Exception as exc:
        logger.warning("Granger SAR->EIA failed: %s", exc)
        results["sar_causes_eia"] = {"error": str(exc)}
        results["sar_causes_eia_min_p"] = np.nan

    # Direction 2: EIA -> SAR (EIA Granger-causes SAR)
    try:
        gc_eia_to_sar = grangercausalitytests(
            work[[sar_col, eia_col]].values, maxlag=max_lag, verbose=False
        )
        eia_to_sar_pvals = {}
        for lag in range(1, max_lag + 1):
            f_test = gc_eia_to_sar[lag][0]
            eia_to_sar_pvals[f"lag_{lag}"] = float(f_test["ssr_ftest"][1])
        results["eia_causes_sar"] = eia_to_sar_pvals
        results["eia_causes_sar_min_p"] = min(eia_to_sar_pvals.values())
    except Exception as exc:
        logger.warning("Granger EIA->SAR failed: %s", exc)
        results["eia_causes_sar"] = {"error": str(exc)}
        results["eia_causes_sar_min_p"] = np.nan

    logger.info(
        "Test 6 (Granger) for '%s': SAR->EIA min_p=%.4f, EIA->SAR min_p=%.4f",
        sar_col,
        results.get("sar_causes_eia_min_p", np.nan),
        results.get("eia_causes_sar_min_p", np.nan),
    )

    return results


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7: Wind Confound (Partial Correlation)
# ═════════════════════════════════════════════════════════════════════════════

def test_wind_confound(
    df: pd.DataFrame,
    sar_col: str,
    eia_col: str = "stocks_mbbl",
    wind_col: str = "wind_speed_10m",
) -> dict:
    """Test 7: Partial correlation controlling for wind speed.

    If the relationship disappears after removing wind, backscatter isn't
    measuring tank fill — it's measuring weather-driven surface roughness.

    Formula: r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))

    Parameters
    ----------
    df : Analysis-ready DataFrame.
    sar_col : SAR feature column.
    eia_col : EIA column.
    wind_col : Wind speed column to control for.

    Returns
    -------
    Dict with raw and partial correlation results.
    """
    work = df[[sar_col, eia_col]].copy()
    if wind_col in df.columns:
        work[wind_col] = df[wind_col]
    else:
        logger.warning("Wind column '%s' not found; cannot compute partial correlation", wind_col)
        return {
            "test": "wind_confound",
            "feature": sar_col,
            "error": "wind_column_missing",
        }

    work = work.dropna()

    if len(work) < MIN_OBSERVATIONS:
        return {
            "test": "wind_confound",
            "feature": sar_col,
            "error": "insufficient_data",
            "n_obs": len(work),
        }

    x = work[sar_col].values
    y = work[eia_col].values
    z = work[wind_col].values

    # Raw correlations
    r_xy, p_xy = stats.pearsonr(x, y)
    r_xz, _ = stats.pearsonr(x, z)
    r_yz, _ = stats.pearsonr(y, z)

    # Partial correlation: r_xy.z
    numerator = r_xy - r_xz * r_yz
    denominator = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))

    if denominator < 1e-10:
        partial_r = np.nan
        partial_p = np.nan
    else:
        partial_r = numerator / denominator
        # Approximate p-value via t-test
        n = len(work)
        t_stat = partial_r * np.sqrt((n - 3) / (1 - partial_r ** 2)) if abs(partial_r) < 1 else np.inf
        partial_p = float(2 * stats.t.sf(abs(t_stat), df=n - 3))

    result = {
        "test": "wind_confound",
        "feature": sar_col,
        "raw_r": float(r_xy),
        "raw_p": float(p_xy),
        "partial_r": float(partial_r) if not np.isnan(partial_r) else None,
        "partial_p": float(partial_p) if not np.isnan(partial_p) else None,
        "r_sar_wind": float(r_xz),
        "r_eia_wind": float(r_yz),
        "relationship_survives_wind": bool(
            not np.isnan(partial_r) and abs(partial_r) > KILL_THRESHOLD
        ),
        "n_obs": len(work),
    }

    logger.info(
        "Test 7 (wind confound) for '%s': raw r=%.3f, partial r=%.3f, survives=%s",
        sar_col,
        r_xy,
        partial_r if not np.isnan(partial_r) else 0.0,
        result["relationship_survives_wind"],
    )

    return result


# ═════════════════════════════════════════════════════════════════════════════
# TEST 8: Multiple Comparison Correction (BH FDR)
# ═════════════════════════════════════════════════════════════════════════════

def test_multiple_comparison(results_df: pd.DataFrame) -> pd.DataFrame:
    """Test 8: Apply Benjamini-Hochberg FDR correction across all tested features.

    Only features surviving q < FDR_THRESHOLD are considered real.

    Parameters
    ----------
    results_df : DataFrame with 'pearson_p' column from tests 1-3.

    Returns
    -------
    DataFrame with added 'bh_q' and 'survives_fdr' columns.
    """
    if len(results_df) == 0 or "pearson_p" not in results_df.columns:
        logger.warning("No p-values to correct")
        return results_df

    out = results_df.copy()
    valid_mask = out["pearson_p"].notna()

    if valid_mask.sum() == 0:
        out["bh_q"] = np.nan
        out["survives_fdr"] = False
        return out

    _, q_values, _, _ = multipletests(
        out.loc[valid_mask, "pearson_p"].values,
        alpha=FDR_THRESHOLD,
        method="fdr_bh",
    )

    out["bh_q"] = np.nan
    out.loc[valid_mask, "bh_q"] = q_values
    out["survives_fdr"] = out["bh_q"] < FDR_THRESHOLD

    n_survive = out["survives_fdr"].sum()
    logger.info(
        "Test 8 (BH FDR): %d / %d features survive q < %.2f",
        n_survive, len(out), FDR_THRESHOLD,
    )

    return out


# ═════════════════════════════════════════════════════════════════════════════
# CONTROL NULL TEST
# ═════════════════════════════════════════════════════════════════════════════

def test_control_null(
    df_control: pd.DataFrame,
    eia_col: str = "stocks_mbbl",
) -> pd.DataFrame:
    """Run Tests 1-3 on control grassland ROI. Must show NO significant correlation.

    If the control (grassland) correlates with EIA stocks, the signal is
    atmospheric/calibration, not tank-fill.

    Parameters
    ----------
    df_control : Control ROI weekly features merged with EIA.
    eia_col : EIA column name.

    Returns
    -------
    DataFrame with control null test results.
    """
    control_cols = [c for c in ["mean_db", "median_db", "std_db", "bright_pixel_ratio"]
                    if c in df_control.columns]

    if len(control_cols) == 0 or eia_col not in df_control.columns:
        logger.warning("Insufficient data for control null test")
        return pd.DataFrame()

    # Test 1 on control: raw correlation
    raw = test_raw_correlation(df_control, control_cols, eia_col)
    if len(raw) > 0:
        raw["test"] = "control_null_raw"

    # Test 2 on control: first difference
    df_ctrl_diff = df_control.copy()
    for col in control_cols:
        df_ctrl_diff[f"delta_{col}"] = df_ctrl_diff[col].diff()
    if eia_col in df_ctrl_diff.columns:
        df_ctrl_diff[f"delta_{eia_col}"] = df_ctrl_diff[eia_col].diff()

    diff = test_first_difference(df_ctrl_diff, control_cols, eia_col)
    if len(diff) > 0:
        diff["test"] = "control_null_first_diff"

    # Test 3 on control: detrended
    detr = test_detrended(df_control, control_cols, eia_col)
    if len(detr) > 0:
        detr["test"] = "control_null_detrended"

    result = pd.concat([raw, diff, detr], ignore_index=True)

    if len(result) > 0:
        max_rho = result["pearson_r"].abs().max()
        any_sig = (result["pearson_p"] < 0.05).any()
        logger.info(
            "Control null test: max |rho|=%.3f, any_significant=%s (should be False)",
            max_rho, any_sig,
        )

    return result


# ═════════════════════════════════════════════════════════════════════════════
# KILL CRITERION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_kill_criterion(first_diff_results: pd.DataFrame) -> dict:
    """Check if first-difference |rho| < KILL_THRESHOLD for ALL critical features.

    Kill columns: delta_mean_db, delta_control_diff, delta_wind_corrected,
    delta_seasonal_residual.

    If ALL have |rho| < 0.1, the signal is dead.

    Parameters
    ----------
    first_diff_results : Results from test_first_difference().

    Returns
    -------
    Dict with kill decision and per-feature rho values.
    """
    kill_features = {}
    for col in DELTA_KILL_COLS:
        row = first_diff_results[first_diff_results["feature"] == col]
        if len(row) > 0:
            rho = float(row["pearson_r"].iloc[0])
            kill_features[col] = {
                "rho": rho,
                "abs_rho": abs(rho),
                "below_threshold": abs(rho) < KILL_THRESHOLD,
            }
        else:
            kill_features[col] = {
                "rho": None,
                "abs_rho": None,
                "below_threshold": True,  # Missing = cannot validate = treat as below
            }

    all_below = all(v["below_threshold"] for v in kill_features.values())
    any_tested = any(v["rho"] is not None for v in kill_features.values())

    result = {
        "kill": all_below and any_tested,
        "threshold": KILL_THRESHOLD,
        "features": kill_features,
        "reason": (
            "ALL first-difference correlations below kill threshold — "
            "levels correlation is spurious (shared trend / confound)"
            if all_below and any_tested
            else "At least one feature shows first-difference signal above threshold"
            if not all_below
            else "No kill features could be tested"
        ),
    }

    logger.info(
        "Kill criterion: KILL=%s — %s",
        result["kill"],
        result["reason"],
    )

    return result


# ═════════════════════════════════════════════════════════════════════════════
# LEAD-LAG CROSS-CORRELATION (diagnostic)
# ═════════════════════════════════════════════════════════════════════════════

def lead_lag_crosscorrelation(
    sar_series: pd.Series,
    econ_series: pd.Series,
    max_lag_weeks: int = 12,
) -> pd.DataFrame:
    """Compute cross-correlation at lags from -max_lag to +max_lag weeks.

    Convention:
    - Positive lag = SAR leads economic data (SAR is predictive).
    - Negative lag = economic data leads SAR (SAR is lagging).

    Adapted from Port Volume project.
    """
    combined = pd.DataFrame({"sar": sar_series, "econ": econ_series}).dropna()

    if len(combined) < 20:
        logger.warning("Insufficient data for lead-lag analysis: %d rows", len(combined))
        return pd.DataFrame(columns=["lag_weeks", "correlation", "p_value", "is_best"])

    sar_vals = combined["sar"].values
    econ_vals = combined["econ"].values
    n = len(sar_vals)

    records = []
    for lag in range(-max_lag_weeks, max_lag_weeks + 1):
        if lag > 0:
            x = sar_vals[: n - lag]
            y = econ_vals[lag:]
        elif lag < 0:
            x = sar_vals[-lag:]
            y = econ_vals[: n + lag]
        else:
            x = sar_vals
            y = econ_vals

        if len(x) < 10:
            continue

        r, p = stats.pearsonr(x, y)
        records.append({"lag_weeks": lag, "correlation": r, "p_value": p})

    result = pd.DataFrame(records)
    if not result.empty:
        best_idx = result["correlation"].abs().idxmax()
        result["is_best"] = False
        result.loc[best_idx, "is_best"] = True

    return result


# ═════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def _plot_first_diff_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot of first-diff SAR vs first-diff EIA."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Test 2: First-Difference Scatter — THE KILL TEST", fontsize=14, fontweight="bold")

    delta_eia = "delta_stocks_mbbl"
    if delta_eia not in df.columns:
        return

    for ax, col in zip(axes.flat, DELTA_KILL_COLS):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        mask = df[col].notna() & df[delta_eia].notna()
        x = df.loc[mask, col].values
        y = df.loc[mask, delta_eia].values

        if len(x) < 5:
            ax.set_visible(False)
            continue

        r, p = stats.pearsonr(x, y)
        ax.scatter(x, y, alpha=0.4, s=15, edgecolors="none")
        ax.set_xlabel(col)
        ax.set_ylabel(delta_eia)
        ax.set_title(f"r={r:.3f}, p={p:.3f}")

        # Regression line
        z = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "r-", linewidth=1.5)

        # Kill threshold annotation
        color = "red" if abs(r) < KILL_THRESHOLD else "green"
        ax.text(
            0.05, 0.95, f"|rho|={'BELOW' if abs(r) < KILL_THRESHOLD else 'ABOVE'} {KILL_THRESHOLD}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            color=color, fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(output_dir / "first_diff_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved first-diff scatter plot")


def _plot_year_stability(year_results: pd.DataFrame, output_dir: Path) -> None:
    """Per-year correlation stability plot."""
    if len(year_results) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    features = year_results["feature"].unique()
    for feat in features:
        sub = year_results[year_results["feature"] == feat]
        ax.plot(sub["year"], sub["pearson_r"], "o-", label=feat, markersize=6)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Pearson r")
    ax.set_title("Test 4: Within-Year Correlation Stability")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "year_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved year stability plot")


def _plot_lead_lag(ll_results: pd.DataFrame, feature_name: str, output_dir: Path) -> None:
    """Lead-lag cross-correlation plot."""
    if len(ll_results) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(ll_results["lag_weeks"], ll_results["correlation"], color="steelblue", alpha=0.7)

    best = ll_results[ll_results["is_best"]]
    if len(best) > 0:
        ax.bar(best["lag_weeks"], best["correlation"], color="red", alpha=0.9)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag (weeks)  [+ = SAR leads, - = EIA leads]")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Lead-Lag Cross-Correlation: {feature_name}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"lead_lag_{feature_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved lead-lag plot for %s", feature_name)


def _plot_wind_confound(wind_results: list[dict], output_dir: Path) -> None:
    """Wind confound partial correlation plot."""
    valid = [r for r in wind_results if "error" not in r and r.get("partial_r") is not None]
    if len(valid) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    features = [r["feature"] for r in valid]
    raw_r = [r["raw_r"] for r in valid]
    partial_r = [r["partial_r"] for r in valid]

    x = np.arange(len(features))
    width = 0.35

    ax.bar(x - width / 2, raw_r, width, label="Raw r", color="steelblue", alpha=0.7)
    ax.bar(x + width / 2, partial_r, width, label="Partial r (wind removed)", color="darkorange", alpha=0.7)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=KILL_THRESHOLD, color="red", linewidth=0.8, linestyle="--", label=f"|r| threshold = {KILL_THRESHOLD}")
    ax.axhline(y=-KILL_THRESHOLD, color="red", linewidth=0.8, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Correlation")
    ax.set_title("Test 7: Wind Confound — Does Relationship Survive Wind Removal?")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "wind_confound.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved wind confound plot")


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR: RUN FULL BATTERY
# ═════════════════════════════════════════════════════════════════════════════

def run_full_battery(
    analysis_ready_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    control_path: str | Path | None = None,
    eia_path: str | Path | None = None,
    figures_dir: str | Path | None = None,
) -> dict:
    """Run all 8 statistical tests + control null test.

    This is the most important function. A signal that fails these tests is spurious.

    Parameters
    ----------
    analysis_ready_path : Path to analysis_ready.parquet.
    output_dir : Directory for results.  Default: results/

    Returns
    -------
    Dict with all test results and GO/NO-GO verdict.
    """
    default_output_dir = Path(RESULTS_DIR)
    if analysis_ready_path is None:
        analysis_ready_path = Path(FEATURES_DIR) / "analysis_ready.parquet"
    if output_dir is None:
        output_dir = default_output_dir
    if control_path is None:
        control_path = Path(FEATURES_DIR) / "control_weekly.parquet"
    if eia_path is None:
        eia_path = Path(FEATURES_DIR).parent / "eia" / "cushing_stocks.parquet"

    analysis_ready_path = Path(analysis_ready_path)
    output_dir = Path(output_dir)
    if figures_dir is None:
        figures_dir = Path(FIGURES_DIR) if output_dir == default_output_dir else output_dir / "figures"
    figures_dir = Path(figures_dir)
    control_path = Path(control_path)
    eia_path = Path(eia_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading analysis-ready dataset from %s", analysis_ready_path)
    df = pd.read_parquet(analysis_ready_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    logger.info("Dataset: %d rows, %d columns, %s to %s", len(df), len(df.columns), df.index.min(), df.index.max())

    # Filter to available SAR columns
    sar_cols = [c for c in SAR_FEATURE_COLS if c in df.columns]
    delta_cols = [c for c in DELTA_KILL_COLS if c in df.columns]

    if len(sar_cols) == 0:
        raise ValueError(f"No SAR feature columns found. Available: {list(df.columns)}")

    results = {"dataset_info": {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "date_range": [str(df.index.min()), str(df.index.max())],
        "sar_features_tested": sar_cols,
        "delta_features_tested": delta_cols,
    }}

    print("\n" + "=" * 70)
    print("  CUSHING SAR OIL STORAGE — STATISTICAL TEST BATTERY")
    print("=" * 70)

    # ── TEST 1: Raw Correlation ───────────────────────────────────────────
    print("\n--- TEST 1: Raw Correlation (Levels) ---")
    raw_corr = test_raw_correlation(df, sar_cols)
    if len(raw_corr) > 0:
        for _, row in raw_corr.iterrows():
            print(f"  {row['feature']:30s}  pearson_r={row['pearson_r']:+.3f}  p={row['pearson_p']:.4f}")
    results["test_1_raw_correlation"] = raw_corr.to_dict(orient="records") if len(raw_corr) > 0 else []

    # ── TEST 2: First Difference — THE KILL TEST ──────────────────────────
    print("\n--- TEST 2: First-Difference Correlation — THE KILL TEST ---")
    first_diff = test_first_difference(df, sar_cols + delta_cols)
    if len(first_diff) > 0:
        for _, row in first_diff.iterrows():
            flag = " *** BELOW KILL THRESHOLD ***" if abs(row["pearson_r"]) < KILL_THRESHOLD else ""
            print(f"  {row['feature']:30s}  pearson_r={row['pearson_r']:+.4f}  p={row['pearson_p']:.4f}{flag}")
    results["test_2_first_difference"] = first_diff.to_dict(orient="records") if len(first_diff) > 0 else []

    # ── TEST 3: Detrended Correlation ─────────────────────────────────────
    print("\n--- TEST 3: Detrended Correlation (STL Residuals) ---")
    detrended = test_detrended(df, sar_cols)
    if len(detrended) > 0:
        for _, row in detrended.iterrows():
            print(f"  {row['feature']:30s}  pearson_r={row['pearson_r']:+.3f}  p={row['pearson_p']:.4f}")
    results["test_3_detrended"] = detrended.to_dict(orient="records") if len(detrended) > 0 else []

    # ── TEST 4: Within-Year Stability ─────────────────────────────────────
    print("\n--- TEST 4: Within-Year Stability ---")
    year_stability_all = []
    primary_sar = sar_cols[0] if sar_cols else None
    for col in sar_cols[:3]:  # Test top 3 features
        yr = test_within_year_stability(df, col)
        if len(yr) > 0:
            year_stability_all.append(yr)
            sign_flips = yr["sign_flips"].iloc[0] if "sign_flips" in yr.columns else "N/A"
            print(f"  {col}: {len(yr)} years tested, sign_flips={sign_flips}")

    year_stability_df = pd.concat(year_stability_all, ignore_index=True) if year_stability_all else pd.DataFrame()
    results["test_4_year_stability"] = year_stability_df.to_dict(orient="records") if len(year_stability_df) > 0 else []

    # ── TEST 5: Out-of-Time Validation ────────────────────────────────────
    print("\n--- TEST 5: Out-of-Time Validation ---")
    oot = test_out_of_time(df, sar_cols)
    if len(oot) > 0:
        for _, row in oot.iterrows():
            flag = " *** NEGATIVE R2 ***" if not row["positive_r2"] else ""
            print(f"  {row['feature']:30s}  train_R2={row['train_r2']:+.3f}  test_R2={row['test_r2']:+.3f}{flag}")
    results["test_5_out_of_time"] = oot.to_dict(orient="records") if len(oot) > 0 else []

    # ── TEST 6: Granger Causality ─────────────────────────────────────────
    print("\n--- TEST 6: Granger Causality ---")
    granger_results = []
    for col in sar_cols[:3]:  # Test top 3 features
        gc = test_granger_causality(df, col)
        granger_results.append(gc)
        sar_p = gc.get("sar_causes_eia_min_p", np.nan)
        eia_p = gc.get("eia_causes_sar_min_p", np.nan)
        print(f"  {col:30s}  SAR->EIA min_p={sar_p:.4f}  EIA->SAR min_p={eia_p:.4f}")
    results["test_6_granger"] = granger_results

    # ── TEST 7: Wind Confound ─────────────────────────────────────────────
    print("\n--- TEST 7: Wind Confound (Partial Correlation) ---")
    wind_results = []
    for col in sar_cols:
        wc = test_wind_confound(df, col)
        wind_results.append(wc)
        if "error" not in wc:
            print(
                f"  {col:30s}  raw_r={wc['raw_r']:+.3f}  partial_r={wc['partial_r']:+.3f}  "
                f"survives={'YES' if wc['relationship_survives_wind'] else 'NO'}"
            )
        else:
            print(f"  {col:30s}  ERROR: {wc['error']}")
    results["test_7_wind_confound"] = wind_results

    # ── TEST 8: Multiple Comparison Correction ────────────────────────────
    print("\n--- TEST 8: BH FDR Multiple Comparison Correction ---")
    all_pvals = pd.concat([raw_corr, first_diff, detrended], ignore_index=True)
    fdr_results = test_multiple_comparison(all_pvals)
    if len(fdr_results) > 0:
        survivors = fdr_results[fdr_results["survives_fdr"]]
        print(f"  {len(survivors)} / {len(fdr_results)} tests survive FDR correction (q < {FDR_THRESHOLD})")
        for _, row in survivors.iterrows():
            print(f"    {row['test']:25s}  {row['feature']:25s}  q={row['bh_q']:.4f}")
    results["test_8_fdr"] = fdr_results.to_dict(orient="records") if len(fdr_results) > 0 else []

    # ── CONTROL NULL TEST ─────────────────────────────────────────────────
    print("\n--- CONTROL NULL TEST (grassland ROI — must show NO correlation) ---")
    if control_path.exists():
        try:
            control_weekly = pd.read_parquet(control_path)
            if not isinstance(control_weekly.index, pd.DatetimeIndex):
                control_weekly.index = pd.to_datetime(control_weekly.index)

            # Merge EIA with control
            if eia_path.exists():
                eia = pd.read_parquet(eia_path)
                if not isinstance(eia.index, pd.DatetimeIndex):
                    if "date" in eia.columns:
                        eia = eia.set_index("date")
                    eia.index = pd.to_datetime(eia.index)
                if eia.index.tz is not None:
                    eia.index = eia.index.tz_localize(None)
                if control_weekly.index.tz is not None:
                    control_weekly.index = control_weekly.index.tz_localize(None)

                # Simple reindex merge
                eia_col = "stocks_mbbl"
                if eia_col not in eia.columns:
                    raise KeyError(f"Expected EIA column '{eia_col}' in {eia_path}")

                control_weekly[eia_col] = eia[eia_col].reindex(
                    control_weekly.index,
                    method="nearest",
                    tolerance=pd.Timedelta("3D"),
                )

                ctrl_null = test_control_null(control_weekly, eia_col=eia_col)
                if len(ctrl_null) > 0:
                    any_sig = (ctrl_null["pearson_p"] < 0.05).any()
                    print(f"  Control has significant correlation with EIA: {any_sig} (should be False)")
                    for _, row in ctrl_null.iterrows():
                        print(f"    {row['test']:30s} {row['feature']:20s}  r={row['pearson_r']:+.3f}  p={row['pearson_p']:.4f}")
                results["control_null"] = ctrl_null.to_dict(orient="records") if len(ctrl_null) > 0 else []
            else:
                print("  EIA data not found for control merge; skipping.")
                results["control_null"] = []
        except Exception as exc:
            print(f"  Control null test failed: {exc}")
            results["control_null"] = {"error": str(exc)}
    else:
        print(f"  Control weekly data not found at {control_path}; skipping.")
        results["control_null"] = []

    # ── KILL CRITERION ────────────────────────────────────────────────────
    print("\n--- KILL CRITERION EVALUATION ---")
    kill = evaluate_kill_criterion(first_diff)
    results["kill_criterion"] = kill
    print(f"  KILL = {kill['kill']}")
    print(f"  Reason: {kill['reason']}")
    for feat, info in kill["features"].items():
        rho_str = f"{info['rho']:+.4f}" if info["rho"] is not None else "N/A"
        print(f"    {feat:35s}  rho={rho_str}  below_threshold={info['below_threshold']}")

    # ── VERDICT ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if kill["kill"]:
        verdict = "NO-GO"
        verdict_detail = (
            "SIGNAL IS DEAD. First-difference correlations are below kill threshold "
            f"({KILL_THRESHOLD}) for all critical features. The levels correlation is "
            "spurious — driven by shared trends, not by SAR measuring tank fill changes. "
            "This is a rigorous negative result."
        )
    else:
        # Check additional criteria
        has_positive_oot_r2 = len(oot) > 0 and oot["positive_r2"].any()
        has_fdr_survivors = len(fdr_results) > 0 and fdr_results["survives_fdr"].any()

        if has_positive_oot_r2 and has_fdr_survivors:
            verdict = "GO (with caveats)"
            verdict_detail = (
                "Signal shows promise: first-difference correlation above kill threshold, "
                "positive out-of-time R-squared, and features survive FDR correction. "
                "Proceed to modeling with caution."
            )
        else:
            verdict = "WEAK — PROCEED WITH CAUTION"
            verdict_detail = (
                "First-difference signal is above kill threshold but additional validation "
                "is mixed. Check out-of-time R-squared and FDR results carefully before "
                "proceeding to modeling."
            )

    results["verdict"] = verdict
    results["verdict_detail"] = verdict_detail

    print(f"  VERDICT: {verdict}")
    print(f"  {verdict_detail}")
    print("=" * 70 + "\n")

    # ── DIAGNOSTIC PLOTS ──────────────────────────────────────────────────
    logger.info("Generating diagnostic plots...")

    try:
        _plot_first_diff_scatter(df, figures_dir)
    except Exception as exc:
        logger.warning("First-diff scatter plot failed: %s", exc)

    try:
        _plot_year_stability(year_stability_df, figures_dir)
    except Exception as exc:
        logger.warning("Year stability plot failed: %s", exc)

    # Lead-lag for primary feature
    eia_col = "stocks_mbbl"
    if primary_sar and eia_col in df.columns:
        try:
            ll = lead_lag_crosscorrelation(df[primary_sar], df[eia_col])
            _plot_lead_lag(ll, primary_sar, figures_dir)
            results["lead_lag"] = ll.to_dict(orient="records") if len(ll) > 0 else []
        except Exception as exc:
            logger.warning("Lead-lag plot failed: %s", exc)

    try:
        _plot_wind_confound(wind_results, figures_dir)
    except Exception as exc:
        logger.warning("Wind confound plot failed: %s", exc)

    # ── SAVE RESULTS ──────────────────────────────────────────────────────
    results_path = output_dir / "statistical_tests.json"

    # Convert any remaining non-serializable types
    def _make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    def _recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: _recursive_convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_recursive_convert(v) for v in obj]
        return _make_serializable(obj)

    results_clean = _recursive_convert(results)

    with open(results_path, "w") as f:
        json.dump(results_clean, f, indent=2, default=str)

    report_paths = write_markdown_report(
        results_clean,
        output_dir=output_dir,
        analysis_ready_path=analysis_ready_path,
    )
    logger.info("Saved results to %s", results_path)
    for report_path in report_paths:
        logger.info("Saved report to %s", report_path)
    print(f"Results saved to: {results_path}")
    for report_path in report_paths:
        print(f"Report saved to: {report_path}")
    print(f"Plots saved to: {figures_dir}/")

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

def _fmt_number(value: object, precision: int = 3, signed: bool = False) -> str:
    """Format numeric values for markdown summaries."""
    if value is None:
        return "N/A"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(num):
        return "N/A"
    return f"{num:+.{precision}f}" if signed else f"{num:.{precision}f}"


def _fmt_p_value(value: object) -> str:
    """Format p-values compactly."""
    if value is None:
        return "N/A"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(num):
        return "N/A"
    if num < 1e-4:
        return "<1e-4"
    return f"{num:.4f}"


def _best_by_abs(records: list[dict], field: str) -> Optional[dict]:
    """Return the record with the largest absolute value for a metric."""
    valid = [row for row in records if row.get(field) is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: abs(float(row[field])))


def _best_by_value(records: list[dict], field: str) -> Optional[dict]:
    """Return the record with the largest direct value for a metric."""
    valid = [row for row in records if row.get(field) is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: float(row[field]))


def _best_by_min_value(records: list[dict], field: str) -> Optional[dict]:
    """Return the record with the smallest direct value for a metric."""
    valid = [row for row in records if row.get(field) is not None]
    if not valid:
        return None
    return min(valid, key=lambda row: float(row[field]))


def render_markdown_report(results: dict, analysis_ready_path: Path) -> str:
    """Render a reproducible markdown summary from one analysis result dict."""
    dataset = results.get("dataset_info", {})
    raw_corr = results.get("test_1_raw_correlation", [])
    first_diff = results.get("test_2_first_difference", [])
    oot = results.get("test_5_out_of_time", [])
    granger = results.get("test_6_granger", [])
    control_null = results.get("control_null", [])
    fdr = results.get("test_8_fdr", [])
    lead_lag = results.get("lead_lag", [])
    kill = results.get("kill_criterion", {})

    strongest_raw = _best_by_abs(raw_corr, "pearson_r")
    strongest_diff = _best_by_abs(first_diff, "pearson_r")
    best_oot = _best_by_value(oot, "test_r2")
    best_control = _best_by_abs(control_null, "pearson_r") if isinstance(control_null, list) else None
    best_lead_lag = _best_by_abs(lead_lag, "correlation") if isinstance(lead_lag, list) else None
    best_sar_granger = _best_by_min_value(granger, "sar_causes_eia_min_p") if isinstance(granger, list) else None
    best_eia_granger = _best_by_min_value(
        [row for row in granger if row.get("eia_causes_sar_min_p") is not None],
        "eia_causes_sar_min_p",
    ) if isinstance(granger, list) else None

    negative_oot = sum(not row.get("positive_r2", False) for row in oot)
    fdr_survivors = sum(bool(row.get("survives_fdr")) for row in fdr)
    control_significant = [
        row for row in control_null
        if isinstance(control_null, list) and row.get("pearson_p") is not None and float(row["pearson_p"]) < 0.05
    ] if isinstance(control_null, list) else []

    lines = [
        "# Cushing SAR Oil Storage Estimation — Analysis Report",
        "",
        f"Generated from `{analysis_ready_path.name}` by `src.analysis`.",
        "",
        f"## Verdict: {results.get('verdict', 'UNKNOWN')}",
        "",
        results.get("verdict_detail", "No verdict detail available."),
        "",
        "## Dataset",
        "",
        f"- Rows: {dataset.get('n_rows', 'N/A')}",
        f"- Date range: {dataset.get('date_range', ['N/A', 'N/A'])[0]} to {dataset.get('date_range', ['N/A', 'N/A'])[-1]}",
        f"- SAR features tested: {', '.join(dataset.get('sar_features_tested', [])) or 'N/A'}",
        f"- Delta features tested: {', '.join(dataset.get('delta_features_tested', [])) or 'N/A'}",
        "",
        "## Key Findings",
        "",
    ]

    if strongest_raw:
        lines.append(
            f"- Strongest raw level correlation: `{strongest_raw['feature']}` "
            f"(r={_fmt_number(strongest_raw.get('pearson_r'), signed=True)}, p={_fmt_p_value(strongest_raw.get('pearson_p'))})"
        )
    if strongest_diff:
        lines.append(
            f"- Strongest first-difference signal: `{strongest_diff['feature']}` "
            f"(r={_fmt_number(strongest_diff.get('pearson_r'), signed=True)}, p={_fmt_p_value(strongest_diff.get('pearson_p'))})"
        )
    if best_oot:
        lines.append(
            f"- Best out-of-time feature: `{best_oot['feature']}` "
            f"(train R²={_fmt_number(best_oot.get('train_r2'))}, test R²={_fmt_number(best_oot.get('test_r2'))})"
        )
    lines.append(
        f"- Out-of-time validation: {negative_oot} / {len(oot)} features have negative test R²"
        if oot else "- Out-of-time validation: no valid models"
    )
    lines.append(
        f"- FDR correction: {fdr_survivors} / {len(fdr)} tests survive q < {FDR_THRESHOLD}"
        if fdr else "- FDR correction: no valid tests"
    )
    if best_control:
        lines.append(
            f"- Control null strongest feature: `{best_control['feature']}` "
            f"(r={_fmt_number(best_control.get('pearson_r'), signed=True)}, p={_fmt_p_value(best_control.get('pearson_p'))})"
        )
    lines.append(
        f"- Control null significant results: {len(control_significant)}"
        if isinstance(control_null, list) else f"- Control null failed: {control_null}"
    )
    if best_sar_granger:
        lines.append(
            f"- Best SAR→EIA Granger p-value: `{best_sar_granger['feature']}` "
            f"(p={_fmt_p_value(best_sar_granger.get('sar_causes_eia_min_p'))})"
        )
    if best_eia_granger:
        lines.append(
            f"- Best EIA→SAR Granger p-value: `{best_eia_granger['feature']}` "
            f"(p={_fmt_p_value(best_eia_granger.get('eia_causes_sar_min_p'))})"
        )
    if best_lead_lag:
        lines.append(
            f"- Strongest lead-lag correlation: lag {best_lead_lag['lag_weeks']} weeks "
            f"(r={_fmt_number(best_lead_lag.get('correlation'), signed=True)}, p={_fmt_p_value(best_lead_lag.get('p_value'))})"
        )

    lines.extend([
        "",
        "## Kill Criterion",
        "",
        f"- KILL = {kill.get('kill', 'N/A')}",
        f"- Reason: {kill.get('reason', 'N/A')}",
        "",
        "## Interpretation",
        "",
    ])

    if best_oot and float(best_oot.get("test_r2", 0.0)) <= 0:
        lines.append("- The pipeline still fails out-of-time validation, so the apparent level correlations do not generalize.")
    if strongest_diff and abs(float(strongest_diff.get("pearson_r", 0.0))) < KILL_THRESHOLD:
        lines.append("- No first-difference feature survives the kill threshold; the level relationship is likely spurious.")
    elif strongest_diff:
        lines.append("- One first-difference feature barely clears the kill threshold, but it is marginal and should not be treated as production-grade signal.")
    if control_significant:
        lines.append("- The control ROI still correlates with EIA on some tests, which weakens the causal interpretation of the tank-farm features.")
    if best_sar_granger and float(best_sar_granger.get("sar_causes_eia_min_p", 1.0)) >= 0.05:
        lines.append("- SAR does not Granger-cause EIA in this run; the temporal direction remains unfavorable.")

    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `statistical_tests.json` is the machine-readable source of truth for this report.",
        "- Diagnostic plots are written alongside the JSON results in the run-specific figures directory.",
        "",
    ])

    return "\n".join(lines)


def write_markdown_report(results: dict, output_dir: Path, analysis_ready_path: Path) -> list[Path]:
    """Write the markdown analysis report for one run."""
    report_text = render_markdown_report(results, analysis_ready_path=analysis_ready_path)
    report_paths = [output_dir / "results_report.md"]

    if output_dir == Path(RESULTS_DIR):
        report_paths.append(output_dir / "negative_result_report.md")

    for report_path in report_paths:
        with open(report_path, "w") as f:
            f.write(report_text + "\n")

    return report_paths

def build_cli_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the statistical battery."""
    parser = argparse.ArgumentParser(description="Run the SAR statistical test battery.")
    parser.add_argument(
        "--analysis-ready-path",
        type=Path,
        default=Path(FEATURES_DIR) / "analysis_ready.parquet",
        help="Input analysis-ready parquet.",
    )
    parser.add_argument(
        "--control-path",
        type=Path,
        default=Path(FEATURES_DIR) / "control_weekly.parquet",
        help="Input control-weekly parquet.",
    )
    parser.add_argument(
        "--eia-path",
        type=Path,
        default=Path(FEATURES_DIR).parent / "eia" / "cushing_stocks.parquet",
        help="Input EIA stocks parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(RESULTS_DIR),
        help="Directory for JSON results.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory for diagnostic figures. Defaults to results/figures for the main run or <output-dir>/figures for custom runs.",
    )
    return parser

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = build_cli_parser().parse_args()

    results = run_full_battery(
        analysis_ready_path=args.analysis_ready_path,
        output_dir=args.output_dir,
        control_path=args.control_path,
        eia_path=args.eia_path,
        figures_dir=args.figures_dir,
    )
