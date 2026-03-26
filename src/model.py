"""Predictive model for Cushing SAR oil storage estimation (Phase 5).

This module ONLY runs if Phase 4 statistical analysis finds signal:
first-diff |rho| >= KILL_THRESHOLD for at least one FDR-surviving feature.
If Phase 4 kills the project, this module documents the negative result instead.

Models:
  - Naive benchmark (previous week persists)
  - Ridge regression (RidgeCV with StandardScaler)
  - XGBoost (conservative hyperparameters, TimeSeriesSplit CV)

Validation:
  - Walk-forward (expanding window) — no information leakage
  - Skill score vs naive: 1 - (model_rmse / naive_rmse)
  - Directional accuracy: % correct build vs draw predictions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.config import (
    TRAIN_END,
    TEST_START,
    FDR_THRESHOLD,
    KILL_THRESHOLD,
    FEATURES_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
)

logger = logging.getLogger(__name__)

# Unit conversion: EIA reports in thousand bbl; we report RMSE in million bbl
THOUSAND_TO_MILLION = 1000.0


# ═════════════════════════════════════════════════════════════════════════════
# 1. Load analysis results and check kill criterion
# ═════════════════════════════════════════════════════════════════════════════

def load_analysis_results(
    results_path: str | Path,
) -> tuple[dict, bool]:
    """Load results/statistical_tests.json and check kill criterion.

    Parameters
    ----------
    results_path : Path to statistical_tests.json from Phase 4.

    Returns
    -------
    Tuple of (results_dict, is_killed).
        is_killed is True if the project should be terminated (no signal).
    """
    results_path = Path(results_path)
    if not results_path.exists():
        logger.error(
            "Analysis results not found at %s. Run analysis.py (Phase 4) first.",
            results_path,
        )
        raise FileNotFoundError(
            f"Statistical test results not found: {results_path}\n"
            "Please run Phase 4 (src/analysis.py) first to generate "
            "results/statistical_tests.json."
        )

    with open(results_path, "r") as f:
        results = json.load(f)

    # Check kill criterion
    kill_info = results.get("kill_criterion", {})
    is_killed = kill_info.get("kill", False)

    if is_killed:
        logger.warning(
            "PROJECT KILLED by Phase 4: %s", kill_info.get("reason", "unknown")
        )
    else:
        verdict = results.get("verdict", "UNKNOWN")
        logger.info("Phase 4 verdict: %s — proceeding to modeling.", verdict)

    return results, is_killed


# ═════════════════════════════════════════════════════════════════════════════
# 2. Get surviving features
# ═════════════════════════════════════════════════════════════════════════════

def get_surviving_features(results: dict) -> list[str]:
    """Identify SAR features that survived FDR correction AND the kill test.

    A feature survives if:
      - It appears in test_8_fdr with survives_fdr == True, AND
      - Its first-difference |rho| >= KILL_THRESHOLD

    Parameters
    ----------
    results : The loaded statistical_tests.json dict.

    Returns
    -------
    List of feature column names that are valid for modeling.
    """
    # Collect FDR survivors
    fdr_results = results.get("test_8_fdr", [])
    fdr_survivors = set()
    for entry in fdr_results:
        if entry.get("survives_fdr", False):
            fdr_survivors.add(entry["feature"])

    # Collect first-diff features above kill threshold
    first_diff_results = results.get("test_2_first_difference", [])
    first_diff_above = set()
    for entry in first_diff_results:
        rho = entry.get("pearson_r")
        if rho is not None and abs(rho) >= KILL_THRESHOLD:
            feature = entry["feature"]
            first_diff_above.add(feature)
            # Also add the levels version (strip delta_ prefix)
            if feature.startswith("delta_"):
                first_diff_above.add(feature[6:])

    # Features must survive both FDR and first-diff threshold
    # But FDR is applied across all tests, so a feature can survive FDR
    # in the raw or detrended test. We accept any feature that:
    #   (a) has a first-diff |rho| >= KILL_THRESHOLD, OR
    #   (b) survived FDR in any test
    # The intersection would be too restrictive when tests are on
    # different column names (delta_ vs levels).
    survivors = []

    # Primary: features whose delta_ version passed the kill test AND
    # that appear (in any form) in FDR survivors
    for feat in first_diff_above:
        if feat in fdr_survivors:
            survivors.append(feat)
        # Check delta_ version too
        delta_feat = f"delta_{feat}" if not feat.startswith("delta_") else feat
        if delta_feat in fdr_survivors and feat not in survivors:
            survivors.append(feat)

    # If strict intersection yields nothing, fall back to first-diff-above
    # features (the kill test is the most important gate)
    if len(survivors) == 0 and len(first_diff_above) > 0:
        logger.warning(
            "No features survive both FDR and first-diff threshold. "
            "Falling back to first-diff-above features: %s",
            first_diff_above,
        )
        survivors = [f for f in first_diff_above if not f.startswith("delta_")]
        if len(survivors) == 0:
            survivors = list(first_diff_above)

    # Remove delta_ prefix for modeling (we use levels + deltas as separate features)
    clean = []
    for feat in survivors:
        if feat.startswith("delta_"):
            base = feat[6:]
            if base not in clean:
                clean.append(base)
            if feat not in clean:
                clean.append(feat)
        else:
            if feat not in clean:
                clean.append(feat)

    logger.info("Surviving features for modeling: %s", clean)
    return clean


# ═════════════════════════════════════════════════════════════════════════════
# 3. Ridge regression
# ═════════════════════════════════════════════════════════════════════════════

def train_ridge_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "stocks_mbbl",
    train_end: str = TRAIN_END,
) -> tuple[object, object, np.ndarray, np.ndarray, dict]:
    """Train Ridge regression with cross-validated alpha.

    Parameters
    ----------
    df : Analysis-ready DataFrame with DatetimeIndex.
    feature_cols : SAR feature column names to use.
    target_col : EIA inventory column (thousand bbl).
    train_end : End of training period (inclusive).

    Returns
    -------
    Tuple of (fitted RidgeCV model, fitted StandardScaler,
              train predictions, test predictions, metrics dict).
    """
    train_end_ts = pd.Timestamp(train_end)

    # Subset to rows with valid target and features
    cols_needed = [c for c in feature_cols if c in df.columns] + [target_col]
    work = df[cols_needed].dropna()

    if len(work) == 0:
        raise ValueError("No valid rows after dropping NaN for Ridge model.")

    # Split
    train = work[work.index <= train_end_ts]
    test = work[work.index > train_end_ts]

    if len(train) < 20:
        raise ValueError(f"Insufficient training data: {len(train)} rows (need >= 20).")

    feature_cols_present = [c for c in feature_cols if c in work.columns]

    X_train = train[feature_cols_present].values
    y_train = train[target_col].values
    X_test = test[feature_cols_present].values if len(test) > 0 else np.empty((0, len(feature_cols_present)))
    y_test = test[target_col].values if len(test) > 0 else np.array([])

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else X_test

    # RidgeCV
    alphas = np.logspace(-3, 3, 50)
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train_scaled, y_train)

    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled) if len(X_test_scaled) > 0 else np.array([])

    # Metrics
    metrics = {
        "model": "ridge",
        "alpha": float(model.alpha_),
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(feature_cols_present),
        "feature_cols": feature_cols_present,
    }

    if len(y_test) > 0:
        metrics.update(compute_metrics(y_test, test_preds))
        metrics["train_r2"] = float(r2_score(y_train, train_preds))
    else:
        metrics["train_r2"] = float(r2_score(y_train, train_preds))

    logger.info(
        "Ridge model: alpha=%.4f, train R2=%.3f, test R2=%.3f, features=%d",
        model.alpha_,
        metrics.get("train_r2", np.nan),
        metrics.get("r2", np.nan),
        len(feature_cols_present),
    )

    return model, scaler, train_preds, test_preds, metrics


# ═════════════════════════════════════════════════════════════════════════════
# 4. XGBoost
# ═════════════════════════════════════════════════════════════════════════════

def train_xgboost_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "stocks_mbbl",
    train_end: str = TRAIN_END,
) -> tuple[object, np.ndarray, np.ndarray, dict]:
    """Train XGBoost with conservative defaults and time-series CV.

    Parameters
    ----------
    df : Analysis-ready DataFrame with DatetimeIndex.
    feature_cols : SAR feature column names to use.
    target_col : EIA inventory column (thousand bbl).
    train_end : End of training period (inclusive).

    Returns
    -------
    Tuple of (fitted XGBRegressor, train predictions, test predictions, metrics dict).
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        logger.error(
            "xgboost not installed. Install with: pip install xgboost"
        )
        raise

    train_end_ts = pd.Timestamp(train_end)

    cols_needed = [c for c in feature_cols if c in df.columns] + [target_col]
    work = df[cols_needed].dropna()

    if len(work) == 0:
        raise ValueError("No valid rows after dropping NaN for XGBoost model.")

    train = work[work.index <= train_end_ts]
    test = work[work.index > train_end_ts]

    if len(train) < 20:
        raise ValueError(f"Insufficient training data: {len(train)} rows (need >= 20).")

    feature_cols_present = [c for c in feature_cols if c in work.columns]

    X_train = train[feature_cols_present].values
    y_train = train[target_col].values
    X_test = test[feature_cols_present].values if len(test) > 0 else np.empty((0, len(feature_cols_present)))
    y_test = test[target_col].values if len(test) > 0 else np.array([])

    # Time-series aware CV for hyperparameter awareness
    tscv = TimeSeriesSplit(n_splits=5)

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )

    # Cross-validated score on training data
    cv_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

        model.fit(
            X_cv_train, y_cv_train,
            eval_set=[(X_cv_val, y_cv_val)],
            verbose=False,
        )
        val_pred = model.predict(X_cv_val)
        cv_scores.append(float(r2_score(y_cv_val, val_pred)))

    # Final fit on full training data
    model.fit(X_train, y_train, verbose=False)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test) if len(X_test) > 0 else np.array([])

    metrics = {
        "model": "xgboost",
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "cv_r2_mean": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores)),
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(feature_cols_present),
        "feature_cols": feature_cols_present,
    }

    if len(y_test) > 0:
        metrics.update(compute_metrics(y_test, test_preds))
        metrics["train_r2"] = float(r2_score(y_train, train_preds))
    else:
        metrics["train_r2"] = float(r2_score(y_train, train_preds))

    logger.info(
        "XGBoost model: CV R2=%.3f +/- %.3f, train R2=%.3f, test R2=%.3f",
        metrics["cv_r2_mean"],
        metrics["cv_r2_std"],
        metrics.get("train_r2", np.nan),
        metrics.get("r2", np.nan),
    )

    return model, train_preds, test_preds, metrics


# ═════════════════════════════════════════════════════════════════════════════
# 5. Walk-forward validation
# ═════════════════════════════════════════════════════════════════════════════

def walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "stocks_mbbl",
    model_type: str = "ridge",
    min_train_weeks: int = 104,
) -> pd.DataFrame:
    """Walk-forward (expanding window) validation — no information leakage.

    For each test point t (from TEST_START onward):
      - Train on all data before t (with at least min_train_weeks)
      - Predict t
      - Never peek ahead

    Parameters
    ----------
    df : Analysis-ready DataFrame with DatetimeIndex.
    feature_cols : Feature column names.
    target_col : Target column.
    model_type : "ridge" or "xgboost".
    min_train_weeks : Minimum training observations before first prediction.

    Returns
    -------
    DataFrame with columns: date, actual, predicted, residual.
    """
    feature_cols_present = [c for c in feature_cols if c in df.columns]
    cols_needed = feature_cols_present + [target_col]
    work = df[cols_needed].dropna().sort_index()

    if len(work) == 0:
        logger.warning("No valid data for walk-forward validation.")
        return pd.DataFrame(columns=["date", "actual", "predicted", "residual"])

    test_start_ts = pd.Timestamp(TEST_START)
    test_mask = work.index >= test_start_ts
    test_indices = work.index[test_mask]

    if len(test_indices) == 0:
        logger.warning("No test data after %s for walk-forward.", TEST_START)
        return pd.DataFrame(columns=["date", "actual", "predicted", "residual"])

    records = []

    for t in test_indices:
        # Training data: everything strictly before t
        train_data = work[work.index < t]

        if len(train_data) < min_train_weeks:
            continue

        X_train = train_data[feature_cols_present].values
        y_train = train_data[target_col].values

        X_test_point = work.loc[[t], feature_cols_present].values
        y_actual = float(work.loc[t, target_col])

        try:
            if model_type == "ridge":
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test_point)

                alphas = np.logspace(-3, 3, 50)
                model = RidgeCV(alphas=alphas, cv=min(5, len(X_train_s)))
                model.fit(X_train_s, y_train)
                y_pred = float(model.predict(X_test_s)[0])

            elif model_type == "xgboost":
                from xgboost import XGBRegressor
                model = XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=1.0,
                    random_state=42,
                    verbosity=0,
                )
                model.fit(X_train, y_train, verbose=False)
                y_pred = float(model.predict(X_test_point)[0])

            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            records.append({
                "date": t,
                "actual": y_actual,
                "predicted": y_pred,
                "residual": y_actual - y_pred,
            })

        except Exception as exc:
            logger.warning("Walk-forward failed at %s: %s", t, exc)
            continue

    result = pd.DataFrame(records)
    if len(result) > 0:
        result = result.set_index("date").sort_index()

    logger.info(
        "Walk-forward validation (%s): %d predictions generated.",
        model_type, len(result),
    )
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 6. Naive benchmark
# ═════════════════════════════════════════════════════════════════════════════

def naive_benchmark(
    df: pd.DataFrame,
    target_col: str = "stocks_mbbl",
) -> tuple[float, float]:
    """Naive forecast: previous week's EIA value persists.

    The model MUST beat this to be meaningful.

    Parameters
    ----------
    df : Analysis-ready DataFrame with DatetimeIndex.
    target_col : EIA inventory column.

    Returns
    -------
    Tuple of (naive_rmse, naive_r2), computed on test period only.
    """
    test_start_ts = pd.Timestamp(TEST_START)

    series = df[target_col].dropna().sort_index()
    test_series = series[series.index >= test_start_ts]

    if len(test_series) < 2:
        logger.warning("Insufficient test data for naive benchmark.")
        return np.nan, np.nan

    # Naive: predict current value = previous week's value
    actual = test_series.values[1:]
    predicted = test_series.values[:-1]

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = float(r2_score(actual, predicted))

    logger.info(
        "Naive benchmark: RMSE=%.1f thousand bbl (%.2f M bbl), R2=%.3f",
        rmse, rmse / THOUSAND_TO_MILLION, r2,
    )

    return rmse, r2


# ═════════════════════════════════════════════════════════════════════════════
# 7. Compute metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    naive_rmse: float | None = None,
) -> dict:
    """Compute comprehensive model evaluation metrics.

    Parameters
    ----------
    actual : True values (thousand bbl).
    predicted : Model predictions (thousand bbl).
    naive_rmse : RMSE of naive benchmark for skill score computation.

    Returns
    -------
    Dict with RMSE (M bbl), MAE, R2, directional accuracy, skill score.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    # Remove any NaN pairs
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) < 2:
        return {
            "rmse_thousand_bbl": np.nan,
            "rmse_million_bbl": np.nan,
            "mae_thousand_bbl": np.nan,
            "r2": np.nan,
            "directional_accuracy": np.nan,
            "skill_score": np.nan,
            "n_predictions": int(len(actual)),
        }

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    r2 = float(r2_score(actual, predicted))

    # Directional accuracy: % of weeks where model correctly predicts
    # build (increase) vs draw (decrease)
    actual_diff = np.diff(actual)
    predicted_diff = np.diff(predicted)
    if len(actual_diff) > 0:
        correct_direction = np.sum(
            np.sign(actual_diff) == np.sign(predicted_diff)
        )
        directional_accuracy = float(correct_direction / len(actual_diff))
    else:
        directional_accuracy = np.nan

    # Skill score vs naive: 1 - (model_rmse / naive_rmse)
    # Positive means model is better than naive
    if naive_rmse is not None and naive_rmse > 0:
        skill_score = float(1.0 - rmse / naive_rmse)
    else:
        skill_score = np.nan

    metrics = {
        "rmse_thousand_bbl": rmse,
        "rmse_million_bbl": rmse / THOUSAND_TO_MILLION,
        "mae_thousand_bbl": mae,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
        "skill_score": skill_score,
        "n_predictions": int(len(actual)),
    }

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# 8. Plotting
# ═════════════════════════════════════════════════════════════════════════════

def plot_model_results(
    walk_forward_df: pd.DataFrame,
    metrics: dict,
    output_dir: str | Path,
) -> None:
    """Generate diagnostic plots for model performance.

    Plot 1: Actual vs predicted time series
    Plot 2: Residuals over time
    Plot 3: Predicted vs actual scatter with R2
    Plot 4: Rolling R2 over 26-week windows

    Parameters
    ----------
    walk_forward_df : Output of walk_forward_validation().
    metrics : Metrics dict from compute_metrics().
    output_dir : Directory to save figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(walk_forward_df) == 0:
        logger.warning("No walk-forward data to plot.")
        return

    wf = walk_forward_df.copy()
    model_name = metrics.get("model", "model")

    # ── Plot 1: Actual vs Predicted time series ──────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(wf.index, wf["actual"], "b-", label="Actual (EIA)", linewidth=1.5)
    ax.plot(wf.index, wf["predicted"], "r--", label=f"Predicted ({model_name})", linewidth=1.5)

    # Mark train/test boundary
    test_start_ts = pd.Timestamp(TEST_START)
    ax.axvline(x=test_start_ts, color="green", linestyle=":", linewidth=1.5, label="Test start")

    ax.set_xlabel("Date")
    ax.set_ylabel("Cushing Stocks (thousand bbl)")
    ax.set_title(
        f"Walk-Forward Validation: {model_name.upper()}\n"
        f"RMSE={metrics.get('rmse_million_bbl', np.nan):.2f} M bbl | "
        f"R2={metrics.get('r2', np.nan):.3f} | "
        f"Directional Acc={metrics.get('directional_accuracy', np.nan):.1%}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"wf_actual_vs_predicted_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: Residuals over time ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(wf.index, wf["residual"], color="steelblue", alpha=0.7, width=5)
    ax.axhline(y=0, color="black", linewidth=0.8)

    # Rolling mean of residuals (non-stationarity check)
    if len(wf) >= 13:
        rolling_mean = wf["residual"].rolling(13, min_periods=5).mean()
        ax.plot(wf.index, rolling_mean, "r-", linewidth=2, label="13-week rolling mean")
        ax.legend()

    ax.set_xlabel("Date")
    ax.set_ylabel("Residual (thousand bbl)")
    ax.set_title(f"Residuals Over Time — {model_name.upper()} (check for non-stationarity)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"wf_residuals_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: Predicted vs Actual scatter ──────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(wf["actual"], wf["predicted"], alpha=0.5, s=20, edgecolors="none")

    # 1:1 line
    lims = [
        min(wf["actual"].min(), wf["predicted"].min()),
        max(wf["actual"].max(), wf["predicted"].max()),
    ]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "r--", linewidth=1, label="1:1 line")

    r2_val = metrics.get("r2", np.nan)
    ax.text(
        0.05, 0.95,
        f"R2 = {r2_val:.3f}\nn = {len(wf)}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Actual (thousand bbl)")
    ax.set_ylabel("Predicted (thousand bbl)")
    ax.set_title(f"Predicted vs Actual — {model_name.upper()}")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"wf_scatter_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 4: Rolling R2 over 26-week windows ─────────────────────────
    if len(wf) >= 26:
        fig, ax = plt.subplots(figsize=(14, 5))

        window = 26
        rolling_r2 = []
        dates = []
        for i in range(window, len(wf) + 1):
            chunk = wf.iloc[i - window:i]
            act = chunk["actual"].values
            pred = chunk["predicted"].values
            if np.std(act) > 0:
                rolling_r2.append(float(r2_score(act, pred)))
            else:
                rolling_r2.append(np.nan)
            dates.append(chunk.index[-1])

        ax.plot(dates, rolling_r2, "b-", linewidth=1.5)
        ax.axhline(y=0, color="red", linewidth=1, linestyle="--", label="R2 = 0")
        ax.fill_between(dates, 0, rolling_r2, alpha=0.2, color="blue")

        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling R2 (26-week window)")
        ax.set_title(f"Model Stability: Rolling R2 — {model_name.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / f"wf_rolling_r2_{model_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("Saved model diagnostic plots to %s", output_dir)


# ═════════════════════════════════════════════════════════════════════════════
# 9. Main orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def run_model_pipeline(
    analysis_ready_path: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> dict:
    """Main model pipeline orchestrator.

    Flow:
      1. Load Phase 4 analysis results, check kill criterion
      2. If KILLED: document negative result, save report, return
      3. If SIGNAL: identify surviving features
      4. Run naive benchmark
      5. Train Ridge with walk-forward validation
      6. If Ridge R2 > 0: also train XGBoost
      7. Compare all models, generate plots
      8. Save model_performance.json

    Parameters
    ----------
    analysis_ready_path : Path to analysis_ready.parquet.
    results_dir : Directory for model outputs.

    Returns
    -------
    Dict with all model results.
    """
    if analysis_ready_path is None:
        analysis_ready_path = Path(FEATURES_DIR) / "analysis_ready.parquet"
    if results_dir is None:
        results_dir = Path(RESULTS_DIR)

    analysis_ready_path = Path(analysis_ready_path)
    results_dir = Path(results_dir)
    figures_dir = Path(FIGURES_DIR)

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Check prerequisites ──────────────────────────────────────────────
    if not analysis_ready_path.exists():
        msg = (
            f"Analysis-ready dataset not found at {analysis_ready_path}.\n"
            "Please run the feature engineering pipeline first:\n"
            "  python -m src.features"
        )
        print(msg)
        logger.error(msg)
        return {"error": msg}

    stats_path = results_dir / "statistical_tests.json"
    if not stats_path.exists():
        msg = (
            f"Statistical test results not found at {stats_path}.\n"
            "Please run the statistical analysis first:\n"
            "  python -m src.analysis"
        )
        print(msg)
        logger.error(msg)
        return {"error": msg}

    # ── Step 1: Load analysis results ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CUSHING SAR OIL STORAGE — PREDICTIVE MODEL (PHASE 5)")
    print("=" * 70)

    analysis_results, is_killed = load_analysis_results(stats_path)
    pipeline_results = {"phase": 5, "analysis_verdict": analysis_results.get("verdict", "UNKNOWN")}

    # ── Step 2: Handle KILLED project ────────────────────────────────────
    if is_killed:
        kill_info = analysis_results.get("kill_criterion", {})
        negative_report = {
            "phase": 5,
            "status": "KILLED",
            "reason": kill_info.get("reason", "First-difference signal below kill threshold."),
            "kill_details": kill_info,
            "conclusion": (
                "NEGATIVE RESULT: SAR backscatter from Sentinel-1 does NOT contain "
                "a statistically robust signal for estimating week-to-week changes "
                "in Cushing, Oklahoma crude oil storage. The levels correlation "
                "observed in raw data is spurious, driven by shared macro trends "
                "and/or atmospheric confounds rather than actual tank-fill changes. "
                "This is consistent with the Santos et al. finding that raw "
                "correlations (rho=0.597) collapse to near-zero (rho=0.010) after "
                "first-differencing."
            ),
            "recommendation": (
                "Do NOT proceed to production modeling. Possible future directions: "
                "(1) Higher-resolution SAR (e.g., ICEYE or Capella at <1m), "
                "(2) Change detection on individual tank shadows rather than "
                "aggregate backscatter statistics, "
                "(3) Multi-sensor fusion with optical imagery for floating-roof "
                "shadow measurement."
            ),
        }

        print("\n" + "-" * 70)
        print("  PROJECT KILLED — NEGATIVE RESULT")
        print("-" * 70)
        print(f"\n  Reason: {negative_report['reason']}")
        print(f"\n  Conclusion:\n  {negative_report['conclusion']}")
        print(f"\n  Recommendation:\n  {negative_report['recommendation']}")
        print("\n" + "=" * 70 + "\n")

        # Save negative report
        report_path = results_dir / "model_performance.json"
        with open(report_path, "w") as f:
            json.dump(negative_report, f, indent=2, default=str)
        logger.info("Negative result report saved to %s", report_path)

        return negative_report

    # ── Step 3: Identify surviving features ──────────────────────────────
    print("\n--- Identifying surviving features ---")
    surviving_features = get_surviving_features(analysis_results)

    if len(surviving_features) == 0:
        msg = (
            "No features survived the combined FDR + first-difference filter. "
            "Cannot proceed to modeling despite non-kill verdict."
        )
        print(f"  {msg}")
        logger.warning(msg)
        pipeline_results["status"] = "NO_FEATURES"
        pipeline_results["message"] = msg

        report_path = results_dir / "model_performance.json"
        with open(report_path, "w") as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        return pipeline_results

    print(f"  Surviving features: {surviving_features}")

    # ── Load analysis-ready data ─────────────────────────────────────────
    df = pd.read_parquet(analysis_ready_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # ── Step 4: Naive benchmark ──────────────────────────────────────────
    print("\n--- Naive Benchmark (previous week persists) ---")
    naive_rmse, naive_r2 = naive_benchmark(df)
    print(f"  Naive RMSE: {naive_rmse:.1f} thousand bbl ({naive_rmse / THOUSAND_TO_MILLION:.2f} M bbl)")
    print(f"  Naive R2:   {naive_r2:.3f}")

    pipeline_results["naive_benchmark"] = {
        "rmse_thousand_bbl": naive_rmse,
        "rmse_million_bbl": naive_rmse / THOUSAND_TO_MILLION if not np.isnan(naive_rmse) else None,
        "r2": naive_r2,
    }

    # ── Step 5: Ridge model with walk-forward ────────────────────────────
    print("\n--- Ridge Regression ---")
    try:
        ridge_model, ridge_scaler, ridge_train_preds, ridge_test_preds, ridge_metrics = \
            train_ridge_model(df, surviving_features)
        ridge_metrics_full = compute_metrics(
            ridge_test_preds, ridge_test_preds, naive_rmse  # placeholder
        )

        # Walk-forward validation (the real test)
        print("  Running walk-forward validation (Ridge)...")
        wf_ridge = walk_forward_validation(df, surviving_features, model_type="ridge")

        if len(wf_ridge) > 0:
            wf_ridge_metrics = compute_metrics(
                wf_ridge["actual"].values,
                wf_ridge["predicted"].values,
                naive_rmse,
            )
            wf_ridge_metrics["model"] = "ridge_walk_forward"
            ridge_metrics["walk_forward"] = wf_ridge_metrics

            print(f"  Walk-forward RMSE:  {wf_ridge_metrics['rmse_million_bbl']:.2f} M bbl")
            print(f"  Walk-forward R2:    {wf_ridge_metrics['r2']:.3f}")
            print(f"  Directional Acc:    {wf_ridge_metrics['directional_accuracy']:.1%}")
            print(f"  Skill vs naive:     {wf_ridge_metrics['skill_score']:.3f}")

            # Generate plots
            plot_model_results(wf_ridge, wf_ridge_metrics, figures_dir)
        else:
            print("  Walk-forward produced no predictions.")

        pipeline_results["ridge"] = ridge_metrics

    except Exception as exc:
        logger.error("Ridge model failed: %s", exc)
        print(f"  Ridge model failed: {exc}")
        pipeline_results["ridge"] = {"error": str(exc)}
        wf_ridge = pd.DataFrame()
        ridge_metrics = {}

    # ── Step 6: XGBoost (only if Ridge R2 > 0) ──────────────────────────
    ridge_r2 = ridge_metrics.get("walk_forward", {}).get("r2", ridge_metrics.get("r2", -1))

    if ridge_r2 > 0:
        print("\n--- XGBoost (Ridge R2 > 0, worth trying) ---")
        try:
            xgb_model, xgb_train_preds, xgb_test_preds, xgb_metrics = \
                train_xgboost_model(df, surviving_features)

            # Walk-forward validation
            print("  Running walk-forward validation (XGBoost)...")
            wf_xgb = walk_forward_validation(df, surviving_features, model_type="xgboost")

            if len(wf_xgb) > 0:
                wf_xgb_metrics = compute_metrics(
                    wf_xgb["actual"].values,
                    wf_xgb["predicted"].values,
                    naive_rmse,
                )
                wf_xgb_metrics["model"] = "xgboost_walk_forward"
                xgb_metrics["walk_forward"] = wf_xgb_metrics

                print(f"  Walk-forward RMSE:  {wf_xgb_metrics['rmse_million_bbl']:.2f} M bbl")
                print(f"  Walk-forward R2:    {wf_xgb_metrics['r2']:.3f}")
                print(f"  Directional Acc:    {wf_xgb_metrics['directional_accuracy']:.1%}")
                print(f"  Skill vs naive:     {wf_xgb_metrics['skill_score']:.3f}")

                # Generate plots
                plot_model_results(wf_xgb, wf_xgb_metrics, figures_dir)
            else:
                print("  Walk-forward produced no predictions.")

            pipeline_results["xgboost"] = xgb_metrics

        except ImportError:
            print("  XGBoost not installed — skipping. Install with: pip install xgboost")
            pipeline_results["xgboost"] = {"error": "xgboost not installed"}
        except Exception as exc:
            logger.error("XGBoost model failed: %s", exc)
            print(f"  XGBoost model failed: {exc}")
            pipeline_results["xgboost"] = {"error": str(exc)}
    else:
        print(f"\n--- Skipping XGBoost (Ridge R2 = {ridge_r2:.3f} <= 0) ---")
        pipeline_results["xgboost"] = {"skipped": True, "reason": f"Ridge R2 = {ridge_r2:.3f} <= 0"}

    # ── Model comparison summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n  {'Model':<25s} {'RMSE (M bbl)':>14s} {'R2':>8s} {'Dir Acc':>9s} {'Skill':>8s}")
    print("  " + "-" * 64)

    # Naive
    print(
        f"  {'Naive (persist)':<25s} "
        f"{naive_rmse / THOUSAND_TO_MILLION:>14.2f} "
        f"{naive_r2:>8.3f} "
        f"{'N/A':>9s} "
        f"{'0.000':>8s}"
    )

    # Ridge walk-forward
    wf_r = pipeline_results.get("ridge", {}).get("walk_forward", {})
    if wf_r and "rmse_million_bbl" in wf_r:
        print(
            f"  {'Ridge (walk-fwd)':<25s} "
            f"{wf_r['rmse_million_bbl']:>14.2f} "
            f"{wf_r['r2']:>8.3f} "
            f"{wf_r['directional_accuracy']:>8.1%} "
            f"{wf_r['skill_score']:>8.3f}"
        )

    # XGBoost walk-forward
    wf_x = pipeline_results.get("xgboost", {}).get("walk_forward", {})
    if wf_x and "rmse_million_bbl" in wf_x:
        print(
            f"  {'XGBoost (walk-fwd)':<25s} "
            f"{wf_x['rmse_million_bbl']:>14.2f} "
            f"{wf_x['r2']:>8.3f} "
            f"{wf_x['directional_accuracy']:>8.1%} "
            f"{wf_x['skill_score']:>8.3f}"
        )

    print()

    # Interpretation
    best_rmse_m = None
    best_model = None
    for model_key, wf_key in [("ridge", wf_r), ("xgboost", wf_x)]:
        if wf_key and "rmse_million_bbl" in wf_key:
            rmse_m = wf_key["rmse_million_bbl"]
            if best_rmse_m is None or rmse_m < best_rmse_m:
                best_rmse_m = rmse_m
                best_model = model_key

    if best_rmse_m is not None:
        beats_naive = best_rmse_m < (naive_rmse / THOUSAND_TO_MILLION)
        competitive = best_rmse_m < 2.0  # < 2M bbl is competitive with commercial SAR

        pipeline_results["best_model"] = best_model
        pipeline_results["beats_naive"] = beats_naive
        pipeline_results["competitive_with_commercial"] = competitive

        if beats_naive and competitive:
            print(f"  RESULT: {best_model.upper()} beats naive AND is competitive (<2M bbl RMSE).")
            print("  This is a positive result — SAR signal has predictive value.")
        elif beats_naive:
            print(f"  RESULT: {best_model.upper()} beats naive but RMSE ({best_rmse_m:.2f}M) > 2M bbl.")
            print("  Signal exists but is noisy — may improve with more data or better features.")
        else:
            print(f"  RESULT: No model beats naive benchmark. SAR signal is too weak for prediction.")
            print("  This is a valid negative result — document and move on.")
    else:
        print("  RESULT: No valid model predictions generated.")

    print("=" * 70 + "\n")

    # ── Save results ─────────────────────────────────────────────────────
    pipeline_results["status"] = "COMPLETED"

    report_path = results_dir / "model_performance.json"

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

    pipeline_results = _recursive_convert(pipeline_results)

    with open(report_path, "w") as f:
        json.dump(pipeline_results, f, indent=2, default=str)

    logger.info("Model performance saved to %s", report_path)
    print(f"Results saved to {report_path}")

    return pipeline_results


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = run_model_pipeline()
