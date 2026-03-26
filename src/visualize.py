"""Visualization module + Price Impact Analysis for Cushing SAR oil storage estimation.

Phase 6: All diagnostic plots and conditional trading simulation.

Generates publication-quality figures for every stage of the analysis pipeline,
from raw backscatter time series through kill-criterion dashboards and
(if signal survives) price-impact assessment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    TRAIN_END,
    TEST_START,
    KILL_THRESHOLD,
    FDR_THRESHOLD,
    FEATURES_DIR,
    EIA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    WTI_FUTURES_TICKER,
)

logger = logging.getLogger(__name__)

# ── Style setup ──────────────────────────────────────────────────────────────

# Try seaborn-v0_8 style, fall back to seaborn if unavailable
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")

# Colorblind-friendly palette (Wong 2011)
CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_RED = "#D55E00"
CB_PURPLE = "#CC79A7"
CB_CYAN = "#56B4E9"
CB_YELLOW = "#F0E442"
CB_BLACK = "#000000"

DEFAULT_DPI = 150


# ═════════════════════════════════════════════════════════════════════════════
# 1. Backscatter Time Series
# ═════════════════════════════════════════════════════════════════════════════

def plot_backscatter_timeseries(
    weekly_df: pd.DataFrame,
    output_dir: str | Path,
    control_df: pd.DataFrame | None = None,
) -> Path:
    """Tank farm mean_db over time, with optional control grassland overlay.

    Parameters
    ----------
    weekly_df : Weekly tank-farm features with DatetimeIndex and ``mean_db``.
    output_dir : Directory to save the figure.
    control_df : Optional weekly control-grassland features.

    Returns
    -------
    Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        weekly_df.index, weekly_df["mean_db"],
        color=CB_BLUE, linewidth=1.2, label="Tank Farm mean dB",
    )

    if control_df is not None and "mean_db" in control_df.columns:
        ax.plot(
            control_df.index, control_df["mean_db"],
            color=CB_ORANGE, linewidth=1.0, alpha=0.7, label="Control Grassland mean dB",
        )

    # Mark train/test split
    split_date = pd.Timestamp(TRAIN_END)
    ax.axvline(split_date, color=CB_RED, linestyle="--", linewidth=1.0, label=f"Train/Test split ({TRAIN_END})")

    ax.set_xlabel("Date")
    ax.set_ylabel("Backscatter (dB)")
    ax.set_title("Cushing Tank Farm — Weekly Mean Backscatter")
    ax.legend(fontsize=9, loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()

    out_path = output_dir / "backscatter_timeseries.png"
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved backscatter timeseries: %s", out_path)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# 2. EIA Stocks Time Series
# ═════════════════════════════════════════════════════════════════════════════

def plot_eia_stocks_timeseries(
    eia_df: pd.DataFrame,
    output_dir: str | Path,
    stocks_col: str = "value",
) -> Path:
    """EIA Cushing stocks over time with key event annotations.

    Parameters
    ----------
    eia_df : EIA stocks DataFrame with DatetimeIndex.
    output_dir : Directory to save the figure.
    stocks_col : Column name for stock levels.

    Returns
    -------
    Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(eia_df.index, eia_df[stocks_col], color=CB_BLUE, linewidth=1.2, label="Cushing Crude Stocks")

    # Mark train/test split
    split_date = pd.Timestamp(TRAIN_END)
    ax.axvline(split_date, color=CB_RED, linestyle="--", linewidth=1.0, label=f"Train/Test split ({TRAIN_END})")

    # Annotate key events
    events = {
        "2020-04-20": "COVID crash /\nnegative WTI",
        "2020-04-01": None,  # approximate storage peak
        "2022-03-01": "Russia-Ukraine\ninvasion",
    }
    for date_str, label in events.items():
        try:
            evt_date = pd.Timestamp(date_str)
            if evt_date < eia_df.index.min() or evt_date > eia_df.index.max():
                continue
            if label is not None:
                # Find nearest stock value for annotation placement
                nearest_idx = eia_df.index.get_indexer([evt_date], method="nearest")[0]
                y_val = eia_df.iloc[nearest_idx][stocks_col]
                ax.annotate(
                    label,
                    xy=(evt_date, y_val),
                    xytext=(30, 30),
                    textcoords="offset points",
                    fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=CB_BLACK, lw=0.8),
                    ha="left",
                    color=CB_RED,
                    fontweight="bold",
                )
        except Exception:
            pass

    ax.set_xlabel("Date")
    ax.set_ylabel("Crude Stocks (thousand bbl)")
    ax.set_title("EIA Cushing Oklahoma Crude Oil Stocks")
    ax.legend(fontsize=9, loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()

    out_path = output_dir / "eia_stocks_timeseries.png"
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved EIA stocks timeseries: %s", out_path)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# 3. First-Difference Scatter (THE key diagnostic)
# ═════════════════════════════════════════════════════════════════════════════

def plot_first_diff_scatter(
    sar_diff: np.ndarray | pd.Series,
    eia_diff: np.ndarray | pd.Series,
    rho: float,
    p_value: float,
    output_dir: str | Path,
    feature_name: str = "delta_mean_db",
) -> Path:
    """Scatter of delta(SAR) vs delta(EIA) — the critical diagnostic plot.

    Parameters
    ----------
    sar_diff : First-differenced SAR values.
    eia_diff : First-differenced EIA values.
    rho : Spearman rho (or Pearson r) to annotate.
    p_value : Associated p-value.
    output_dir : Directory to save the figure.
    feature_name : Label for the SAR feature being plotted.

    Returns
    -------
    Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sar_arr = np.asarray(sar_diff, dtype=float)
    eia_arr = np.asarray(eia_diff, dtype=float)

    # Drop NaNs
    mask = np.isfinite(sar_arr) & np.isfinite(eia_arr)
    sar_clean = sar_arr[mask]
    eia_clean = eia_arr[mask]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(sar_clean, eia_clean, alpha=0.45, s=25, color=CB_BLUE, edgecolors="none")

    # Regression line
    if len(sar_clean) > 2:
        z = np.polyfit(sar_clean, eia_clean, 1)
        x_line = np.linspace(sar_clean.min(), sar_clean.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), color=CB_RED, linewidth=1.8, label="OLS fit")

    # Annotate with rho and p-value
    kill_status = "BELOW" if abs(rho) < KILL_THRESHOLD else "ABOVE"
    color = CB_RED if abs(rho) < KILL_THRESHOLD else CB_GREEN
    annotation = (
        f"Spearman rho = {rho:+.4f}\n"
        f"p-value = {p_value:.4f}\n"
        f"|rho| {kill_status} kill threshold ({KILL_THRESHOLD})"
    )
    ax.text(
        0.05, 0.95, annotation,
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, alpha=0.9),
    )

    ax.set_xlabel(f"{feature_name} (week-over-week change)", fontsize=11)
    ax.set_ylabel("delta EIA Stocks (thousand bbl)", fontsize=11)
    ax.set_title(
        "First-Difference Scatter — THE KILL TEST",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)

    out_path = output_dir / "first_diff_scatter.png"
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved first-diff scatter: %s", out_path)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# 4. Correlation Heatmap
# ═════════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(
    results_dict: dict,
    output_dir: str | Path,
) -> Path:
    """Heatmap of correlation (rho) for each SAR feature x test type.

    Parameters
    ----------
    results_dict : Full results dict from ``run_full_battery``, containing keys
        ``test_1_raw_correlation``, ``test_2_first_difference``, ``test_3_detrended``.
    output_dir : Directory to save the figure.

    Returns
    -------
    Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect test results into a matrix: rows=features, columns=test types
    test_keys = [
        ("test_1_raw_correlation", "Raw Levels"),
        ("test_2_first_difference", "First Diff"),
        ("test_3_detrended", "Detrended"),
    ]

    all_features = set()
    for key, _ in test_keys:
        records = results_dict.get(key, [])
        for rec in records:
            all_features.add(rec.get("feature", ""))

    all_features = sorted(all_features)
    if not all_features:
        logger.warning("No features found for correlation heatmap")
        return output_dir / "correlation_heatmap.png"

    # Build matrices
    rho_matrix = np.full((len(all_features), len(test_keys)), np.nan)
    p_matrix = np.full((len(all_features), len(test_keys)), np.nan)

    for j, (key, _) in enumerate(test_keys):
        records = results_dict.get(key, [])
        for rec in records:
            feat = rec.get("feature", "")
            if feat in all_features:
                i = all_features.index(feat)
                rho_matrix[i, j] = rec.get("pearson_r", np.nan)
                p_matrix[i, j] = rec.get("pearson_p", np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(test_keys) * 2.5), max(6, len(all_features) * 0.6)))

    # Color map: diverging blue-red centered at 0
    vmax = max(0.3, np.nanmax(np.abs(rho_matrix)))
    im = ax.imshow(rho_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Annotate cells with rho value and * for significance
    for i in range(len(all_features)):
        for j in range(len(test_keys)):
            val = rho_matrix[i, j]
            p_val = p_matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "--", ha="center", va="center", fontsize=9, color="gray")
            else:
                sig = "*" if (not np.isnan(p_val) and p_val < FDR_THRESHOLD) else ""
                text_color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:+.3f}{sig}", ha="center", va="center",
                        fontsize=9, fontweight="bold" if sig else "normal", color=text_color)

    ax.set_xticks(range(len(test_keys)))
    ax.set_xticklabels([label for _, label in test_keys], fontsize=10)
    ax.set_yticks(range(len(all_features)))
    ax.set_yticklabels(all_features, fontsize=9)
    ax.set_title("Correlation Heatmap: SAR Features x Test Types\n(* = p < {})".format(FDR_THRESHOLD),
                 fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    out_path = output_dir / "correlation_heatmap.png"
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation heatmap: %s", out_path)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# 5. Year Stability
# ═════════════════════════════════════════════════════════════════════════════

def plot_year_stability(
    yearly_results: pd.DataFrame | list[dict],
    output_dir: str | Path,
) -> Path:
    """Bar chart of correlation per year (2017-2025), colored by sign.

    Parameters
    ----------
    yearly_results : DataFrame or list of dicts from ``test_within_year_stability``.
    output_dir : Directory to save the figure.

    Returns
    -------
    Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(yearly_results, list):
        yr_df = pd.DataFrame(yearly_results)
    else:
        yr_df = yearly_results.copy()

    if len(yr_df) == 0 or "year" not in yr_df.columns:
        logger.warning("No year-stability data to plot")
        return output_dir / "year_stability.png"

    fig, ax = plt.subplots(figsize=(10, 6))

    features = yr_df["feature"].unique()
    n_features = len(features)
    bar_width = 0.8 / max(n_features, 1)

    for idx, feat in enumerate(features):
        sub = yr_df[yr_df["feature"] == feat].sort_values("year")
        years = sub["year"].values
        rhos = sub["pearson_r"].values
        offsets = np.arange(len(years)) + idx * bar_width

        # Color by sign
        colors = [CB_BLUE if r >= 0 else CB_RED for r in rhos]
        ax.bar(offsets, rhos, width=bar_width, color=colors, alpha=0.75, label=feat, edgecolor="white")

    ax.axhline(y=0, color=CB_BLACK, linewidth=1.0, linestyle="-")

    # X-axis labels: use years from the first feature
    first_feat = features[0]
    first_sub = yr_df[yr_df["feature"] == first_feat].sort_values("year")
    tick_positions = np.arange(len(first_sub)) + (n_features - 1) * bar_width / 2
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(first_sub["year"].values.astype(int), fontsize=10)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Pearson r (SAR vs EIA)", fontsize=11)
    ax.set_title("Within-Year Correlation Stability (2017-2025)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")

    out_path = output_dir / "year_stability.png"
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved year stability: %s", out_path)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# 6. Lead-Lag Cross-Correlation
# ═════════════════════════════════════════════════════════════════════════════

def plot_lead_lag(
    lead_lag_df: pd.DataFrame,
    output_dir: str | Path,
    feature_name: str = "mean_db",
) -> Path:
    """Cross-correlation at +/-12 week lags, with optimal lag marked.

    Parameters
    ----------
    lead_lag_df : DataFrame from ``lead_lag_crosscorrelation`` with columns
        ``lag_weeks``, ``correlation``, ``is_best``.
    output_dir : Directory to save the figure.
    feature_name : Name of the SAR feature for the title.

    Returns
    -------
    Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(lead_lag_df) == 0:
        logger.warning("No lead-lag data to plot")
        return output_dir / "lead_lag.png"

    fig, ax = plt.subplots(figsize=(10, 6))

    lags = lead_lag_df["lag_weeks"].values
    corrs = lead_lag_df["correlation"].values

    # Normal bars
    bar_colors = [CB_CYAN] * len(lags)

    # Highlight best lag
    if "is_best" in lead_lag_df.columns:
        best_mask = lead_lag_df["is_best"].values.astype(bool)
        for i, is_best in enumerate(best_mask):
            if is_best:
                bar_colors[i] = CB_RED

    ax.bar(lags, corrs, color=bar_colors, alpha=0.8, edgecolor="white", linewidth=0.5)

    # Vertical line at lag=0
    ax.axvline(x=0, color=CB_BLACK, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axhline(y=0, color=CB_BLACK, linewidth=0.8)

    # Annotate best lag
    if "is_best" in lead_lag_df.columns:
        best_row = lead_lag_df[lead_lag_df["is_best"]]
        if len(best_row) > 0:
            best_lag = best_row["lag_weeks"].iloc[0]
            best_corr = best_row["correlation"].iloc[0]
            ax.annotate(
                f"Best: lag={best_lag}w\nr={best_corr:.3f}",
                xy=(best_lag, best_corr),
                xytext=(20, 20),
                textcoords="offset points",
                fontsize=10, fontweight="bold",
                color=CB_RED,
                arrowprops=dict(arrowstyle="->", color=CB_RED, lw=1.2),
            )

    ax.set_xlabel("Lag (weeks)  [+ = SAR leads, - = EIA leads]", fontsize=11)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_title(f"Lead-Lag Cross-Correlation: {feature_name}", fontsize=13, fontweight="bold")

    out_path = output_dir / "lead_lag.png"
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved lead-lag plot: %s", out_path)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# 7. Control Comparison
# ═════════════════════════════════════════════════════════════════════════════

def plot_control_comparison(
    tank_stats: pd.DataFrame,
    control_stats: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Side-by-side histograms and time series comparison of tank vs control.

    Parameters
    ----------
    tank_stats : Weekly tank-farm features with ``mean_db``.
    control_stats : Weekly control-grassland features with ``mean_db``.
    output_dir : Directory to save the figure.

    Returns
    -------
    Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Histograms
    ax1 = axes[0]
    tank_vals = tank_stats["mean_db"].dropna().values
    ctrl_vals = control_stats["mean_db"].dropna().values

    bins = np.linspace(
        min(tank_vals.min(), ctrl_vals.min()) - 0.5,
        max(tank_vals.max(), ctrl_vals.max()) + 0.5,
        40,
    )
    ax1.hist(tank_vals, bins=bins, alpha=0.6, color=CB_BLUE, label="Tank Farm", density=True, edgecolor="white")
    ax1.hist(ctrl_vals, bins=bins, alpha=0.6, color=CB_ORANGE, label="Control Grassland", density=True, edgecolor="white")
    ax1.set_xlabel("Mean Backscatter (dB)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Backscatter Distribution", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)

    # Panel 2: Time series comparison
    ax2 = axes[1]
    ax2.plot(tank_stats.index, tank_stats["mean_db"], color=CB_BLUE, linewidth=1.0, label="Tank Farm")
    ax2.plot(control_stats.index, control_stats["mean_db"], color=CB_ORANGE, linewidth=1.0, label="Control Grassland")
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Mean Backscatter (dB)", fontsize=11)
    ax2.set_title("Time Series Comparison", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    fig.suptitle("Tank Farm vs Control Grassland", fontsize=14, fontweight="bold", y=1.02)
    fig.autofmt_xdate()
    plt.tight_layout()

    out_path = output_dir / "control_comparison.png"
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved control comparison: %s", out_path)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# 8. Kill Criterion Dashboard
# ═════════════════════════════════════════════════════════════════════════════

def plot_kill_criterion_summary(
    results: dict,
    output_dir: str | Path,
) -> Path:
    """Visual go/no-go dashboard with traffic-light indicators.

    Parameters
    ----------
    results : Full results dict from ``run_full_battery`` containing
        ``kill_criterion``, ``test_2_first_difference``, ``test_5_out_of_time``, etc.
    output_dir : Directory to save the figure.

    Returns
    -------
    Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Kill Criterion Dashboard — GO / NO-GO", fontsize=14, fontweight="bold")

    # ── Panel 1: First-diff rho per kill feature ──
    ax1 = axes[0, 0]
    kill_info = results.get("kill_criterion", {})
    kill_features = kill_info.get("features", {})
    threshold = kill_info.get("threshold", KILL_THRESHOLD)

    if kill_features:
        feat_names = list(kill_features.keys())
        rho_vals = [kill_features[f].get("rho") or 0.0 for f in feat_names]
        below = [kill_features[f].get("below_threshold", True) for f in feat_names]

        colors = [CB_RED if b else CB_GREEN for b in below]
        bars = ax1.barh(range(len(feat_names)), rho_vals, color=colors, alpha=0.8, edgecolor="white")

        ax1.axvline(x=threshold, color=CB_RED, linestyle="--", linewidth=1.0, label=f"+threshold ({threshold})")
        ax1.axvline(x=-threshold, color=CB_RED, linestyle="--", linewidth=1.0, label=f"-threshold")
        ax1.axvline(x=0, color=CB_BLACK, linewidth=0.8)

        ax1.set_yticks(range(len(feat_names)))
        ax1.set_yticklabels([f.replace("delta_", "d_") for f in feat_names], fontsize=8)
        ax1.set_xlabel("First-Diff Pearson r", fontsize=9)
        ax1.set_title("Kill Test: First-Diff Correlations", fontsize=10, fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "No kill features\navailable", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=12, color="gray")
        ax1.set_title("Kill Test: First-Diff Correlations", fontsize=10, fontweight="bold")

    # ── Panel 2: Traffic light verdict ──
    ax2 = axes[0, 1]
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    is_kill = kill_info.get("kill", None)
    if is_kill is True:
        light_color = CB_RED
        verdict_text = "NO-GO\nSignal Killed"
    elif is_kill is False:
        # Check for marginal vs strong
        max_rho = max(
            (abs(kill_features[f].get("rho") or 0.0) for f in kill_features),
            default=0.0,
        )
        if max_rho > 0.3:
            light_color = CB_GREEN
            verdict_text = "GO\nSignal Detected"
        else:
            light_color = CB_YELLOW
            verdict_text = "MARGINAL\nWeak Signal"
    else:
        light_color = "gray"
        verdict_text = "UNKNOWN\nInsufficient Data"

    circle = plt.Circle((0.5, 0.6), 0.2, color=light_color, ec=CB_BLACK, linewidth=2)
    ax2.add_patch(circle)
    ax2.text(0.5, 0.6, verdict_text, ha="center", va="center",
             fontsize=14, fontweight="bold", color="white" if light_color != CB_YELLOW else CB_BLACK)
    ax2.set_title("Verdict", fontsize=10, fontweight="bold")

    # ── Panel 3: Test pass/fail summary ──
    ax3 = axes[1, 0]
    ax3.axis("off")

    test_checks = []
    # First diff
    fd_records = results.get("test_2_first_difference", [])
    if fd_records:
        max_fd_rho = max(abs(r.get("pearson_r", 0) or 0) for r in fd_records)
        test_checks.append(("First-Diff |rho| > 0.1", max_fd_rho >= KILL_THRESHOLD))
    # OOT R2
    oot_records = results.get("test_5_out_of_time", [])
    if oot_records:
        any_pos_r2 = any(r.get("positive_r2", False) for r in oot_records)
        test_checks.append(("Positive OOS R-squared", any_pos_r2))
    # FDR survivors
    fdr_records = results.get("test_8_fdr", [])
    if fdr_records:
        any_fdr = any(r.get("survives_fdr", False) for r in fdr_records)
        test_checks.append(("Survives BH FDR", any_fdr))
    # Control null
    ctrl_records = results.get("control_null", [])
    if isinstance(ctrl_records, list) and ctrl_records:
        ctrl_sig = any(r.get("pearson_p", 1.0) < 0.05 for r in ctrl_records)
        test_checks.append(("Control shows NO signal", not ctrl_sig))

    if test_checks:
        y_pos = 0.9
        for label, passed in test_checks:
            marker = "PASS" if passed else "FAIL"
            color = CB_GREEN if passed else CB_RED
            ax3.text(0.05, y_pos, f"[{marker}]  {label}",
                     transform=ax3.transAxes, fontsize=10, fontweight="bold",
                     color=color, fontfamily="monospace")
            y_pos -= 0.15
    else:
        ax3.text(0.5, 0.5, "No test results\navailable", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=12, color="gray")
    ax3.set_title("Test Summary", fontsize=10, fontweight="bold")

    # ── Panel 4: Reason text ──
    ax4 = axes[1, 1]
    ax4.axis("off")
    reason = kill_info.get("reason", "No kill criterion evaluated.")
    ax4.text(
        0.05, 0.8, reason,
        transform=ax4.transAxes, fontsize=9,
        wrap=True, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray"),
    )
    ax4.set_title("Rationale", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = output_dir / "kill_criterion.png"
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved kill criterion dashboard: %s", out_path)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# 9. SAR Surprise Computation (Phase 6 — Price Impact)
# ═════════════════════════════════════════════════════════════════════════════

def compute_sar_surprise(
    sar_estimates: pd.DataFrame,
    eia_releases: pd.DataFrame,
    sar_col: str = "wind_corrected",
    eia_col: str = "value",
) -> pd.DataFrame:
    """Compute SAR surprise and EIA surprise relative to last known release.

    SAR_surprise = SAR_estimate_at_time_t - last_known_EIA_release
    EIA_surprise = next_EIA_release - last_known_EIA_release

    Parameters
    ----------
    sar_estimates : DataFrame with DatetimeIndex and SAR-based estimate column.
    eia_releases : DataFrame with DatetimeIndex and EIA stock levels.
    sar_col : Column in sar_estimates to use as SAR-based estimate.
    eia_col : Column in eia_releases for stock level.

    Returns
    -------
    DataFrame with ``sar_surprise``, ``eia_surprise``, ``sar_date``,
    ``last_eia_date``, ``next_eia_date`` columns.
    """
    # Ensure tz-naive for alignment
    sar_idx = sar_estimates.index.tz_localize(None) if sar_estimates.index.tz else sar_estimates.index
    eia_idx = eia_releases.index.tz_localize(None) if eia_releases.index.tz else eia_releases.index

    sar_work = sar_estimates.copy()
    sar_work.index = sar_idx
    eia_work = eia_releases[[eia_col]].copy()
    eia_work.index = eia_idx
    eia_work = eia_work.sort_index().dropna()

    records = []
    for sar_date in sar_work.index:
        # Last known EIA release (on or before SAR date)
        prior_eia = eia_work.loc[:sar_date]
        if len(prior_eia) == 0:
            continue

        last_eia_date = prior_eia.index[-1]
        last_eia_val = prior_eia[eia_col].iloc[-1]

        # Next EIA release (after SAR date)
        future_eia = eia_work.loc[sar_date:]
        # Skip the same date if it exists — we want the *next* release
        future_eia = future_eia[future_eia.index > sar_date]
        if len(future_eia) == 0:
            continue

        next_eia_date = future_eia.index[0]
        next_eia_val = future_eia[eia_col].iloc[0]

        sar_val = sar_work.loc[sar_date, sar_col] if sar_col in sar_work.columns else np.nan

        records.append({
            "sar_date": sar_date,
            "last_eia_date": last_eia_date,
            "next_eia_date": next_eia_date,
            "sar_value": sar_val,
            "last_eia_value": last_eia_val,
            "next_eia_value": next_eia_val,
            "sar_surprise": sar_val - last_eia_val,
            "eia_surprise": next_eia_val - last_eia_val,
        })

    result = pd.DataFrame(records)
    if len(result) > 0:
        result = result.set_index("sar_date")
    logger.info("Computed SAR surprise: %d observations", len(result))
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 10. Trading Simulation
# ═════════════════════════════════════════════════════════════════════════════

def simulate_trading(
    sar_surprise: pd.DataFrame,
    wti_prices: pd.DataFrame,
    threshold: float = 0.5,
    price_col: str = "Close",
) -> tuple[pd.DataFrame, dict]:
    """Simulate directional WTI trades based on SAR surprise signal.

    On each SAR observation day:
      - if SAR_surprise > threshold: go long WTI
      - if SAR_surprise < -threshold: go short WTI
      - Hold until next EIA release (Wednesday)

    Parameters
    ----------
    sar_surprise : DataFrame from ``compute_sar_surprise`` with ``sar_surprise``
        and ``next_eia_date`` columns.
    wti_prices : Daily WTI prices with DatetimeIndex and a price column.
    threshold : Minimum |SAR_surprise| to enter a trade (in units of SAR).
    price_col : Column name for WTI price.

    Returns
    -------
    Tuple of (trade_log DataFrame, summary stats dict).
    """
    if len(sar_surprise) == 0 or len(wti_prices) == 0:
        logger.warning("Insufficient data for trading simulation")
        return pd.DataFrame(), {"error": "insufficient_data"}

    # Ensure tz-naive
    wti_idx = wti_prices.index.tz_localize(None) if wti_prices.index.tz else wti_prices.index
    wti = wti_prices.copy()
    wti.index = wti_idx
    wti = wti.sort_index()

    trades = []
    for sar_date, row in sar_surprise.iterrows():
        surprise = row.get("sar_surprise", np.nan)
        if np.isnan(surprise) or abs(surprise) < threshold:
            continue

        direction = 1 if surprise > threshold else -1
        next_eia = row.get("next_eia_date")
        if pd.isna(next_eia):
            continue

        # Get entry price (SAR observation day or next trading day)
        entry_candidates = wti.loc[sar_date:]
        if len(entry_candidates) == 0:
            continue
        entry_date = entry_candidates.index[0]
        entry_price = entry_candidates[price_col].iloc[0]

        # Get exit price (next EIA release day or next trading day)
        exit_candidates = wti.loc[next_eia:]
        if len(exit_candidates) == 0:
            continue
        exit_date = exit_candidates.index[0]
        exit_price = exit_candidates[price_col].iloc[0]

        if entry_date >= exit_date:
            continue

        pnl = direction * (exit_price - entry_price)
        pnl_pct = direction * (exit_price - entry_price) / entry_price * 100

        trades.append({
            "sar_date": sar_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "direction": "LONG" if direction == 1 else "SHORT",
            "sar_surprise": surprise,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_usd": pnl,
            "pnl_pct": pnl_pct,
            "hold_days": (exit_date - entry_date).days,
        })

    trade_log = pd.DataFrame(trades)

    if len(trade_log) == 0:
        return trade_log, {"n_trades": 0, "error": "no_trades_generated"}

    # Summary stats
    trade_log["cumulative_pnl"] = trade_log["pnl_usd"].cumsum()

    n_trades = len(trade_log)
    win_rate = (trade_log["pnl_usd"] > 0).mean()
    total_pnl = trade_log["pnl_usd"].sum()
    mean_pnl = trade_log["pnl_usd"].mean()
    std_pnl = trade_log["pnl_usd"].std()
    sharpe = mean_pnl / std_pnl * np.sqrt(52) if std_pnl > 0 else 0.0

    # Bootstrap: random entry timing comparison
    n_bootstrap = 1000
    rng = np.random.default_rng(42)
    bootstrap_pnls = []
    for _ in range(n_bootstrap):
        random_pnls = []
        for _, trade in trade_log.iterrows():
            hold_days = trade["hold_days"]
            # Random entry within +/- 3 days of actual
            valid_dates = wti.index[(wti.index >= wti.index.min()) & (wti.index <= wti.index.max())]
            if len(valid_dates) < hold_days + 1:
                continue
            rand_idx = rng.integers(0, max(1, len(valid_dates) - hold_days - 1))
            rand_entry = wti[price_col].iloc[rand_idx]
            rand_exit = wti[price_col].iloc[min(rand_idx + hold_days, len(wti) - 1)]
            direction = 1 if trade["direction"] == "LONG" else -1
            random_pnls.append(direction * (rand_exit - rand_entry))
        bootstrap_pnls.append(sum(random_pnls))

    bootstrap_pnls = np.array(bootstrap_pnls)
    p_value_vs_random = float(np.mean(bootstrap_pnls >= total_pnl))

    summary = {
        "n_trades": n_trades,
        "win_rate": float(win_rate),
        "total_pnl_usd": float(total_pnl),
        "mean_pnl_usd": float(mean_pnl),
        "std_pnl_usd": float(std_pnl),
        "annualized_sharpe": float(sharpe),
        "threshold": threshold,
        "p_value_vs_random": p_value_vs_random,
        "bootstrap_mean_pnl": float(bootstrap_pnls.mean()),
        "bootstrap_std_pnl": float(bootstrap_pnls.std()),
    }

    logger.info(
        "Trading simulation: %d trades, win_rate=%.1f%%, total PnL=$%.2f, Sharpe=%.2f, p_vs_random=%.3f",
        n_trades, win_rate * 100, total_pnl, sharpe, p_value_vs_random,
    )

    return trade_log, summary


# ═════════════════════════════════════════════════════════════════════════════
# 11. Price Impact Plot
# ═════════════════════════════════════════════════════════════════════════════

def plot_price_impact(
    trade_log: pd.DataFrame,
    cumulative_pnl: pd.Series | None = None,
    output_dir: str | Path = "",
) -> Path:
    """Cumulative P&L and SAR surprise vs WTI price change scatter.

    Parameters
    ----------
    trade_log : DataFrame from ``simulate_trading`` with trade details.
    cumulative_pnl : Optional pre-computed cumulative P&L series. If None,
        uses ``trade_log["cumulative_pnl"]``.
    output_dir : Directory to save the figure.

    Returns
    -------
    Path to the saved figure.
    """
    output_dir = Path(output_dir) if output_dir else Path(FIGURES_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(trade_log) == 0:
        logger.warning("No trades to plot for price impact")
        return output_dir / "price_impact.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Cumulative P&L
    ax1 = axes[0]
    if cumulative_pnl is not None:
        ax1.plot(cumulative_pnl.index, cumulative_pnl.values, color=CB_BLUE, linewidth=1.5)
    elif "cumulative_pnl" in trade_log.columns:
        dates = trade_log.get("exit_date", trade_log.index)
        ax1.plot(dates, trade_log["cumulative_pnl"].values, color=CB_BLUE, linewidth=1.5, marker="o", markersize=3)

    ax1.axhline(y=0, color=CB_BLACK, linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Cumulative P&L ($)", fontsize=11)
    ax1.set_title("SAR-Based Trading: Cumulative P&L", fontsize=12, fontweight="bold")
    ax1.fill_between(
        trade_log.get("exit_date", trade_log.index),
        trade_log["cumulative_pnl"].values,
        0,
        alpha=0.15,
        color=CB_BLUE,
    )

    # Panel 2: SAR surprise vs WTI price change
    ax2 = axes[1]
    if "sar_surprise" in trade_log.columns and "pnl_pct" in trade_log.columns:
        colors = [CB_GREEN if p > 0 else CB_RED for p in trade_log["pnl_pct"]]
        ax2.scatter(
            trade_log["sar_surprise"], trade_log["pnl_pct"],
            c=colors, alpha=0.6, s=30, edgecolors="none",
        )

        # Regression line
        mask = trade_log["sar_surprise"].notna() & trade_log["pnl_pct"].notna()
        if mask.sum() > 2:
            x = trade_log.loc[mask, "sar_surprise"].values
            y = trade_log.loc[mask, "pnl_pct"].values
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax2.plot(x_line, np.polyval(z, x_line), color=CB_RED, linewidth=1.5)

            r, p = stats.pearsonr(x, y)
            ax2.text(
                0.05, 0.95, f"r = {r:.3f}\np = {p:.3f}",
                transform=ax2.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
            )

    ax2.axhline(y=0, color=CB_BLACK, linewidth=0.8, linestyle="--")
    ax2.axvline(x=0, color=CB_BLACK, linewidth=0.8, linestyle="--")
    ax2.set_xlabel("SAR Surprise", fontsize=11)
    ax2.set_ylabel("WTI Price Change (%)", fontsize=11)
    ax2.set_title("SAR Surprise vs Subsequent WTI Move", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = output_dir / "price_impact.png"
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved price impact plot: %s", out_path)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# 12. Price Impact Analysis Orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def run_price_impact_analysis(
    model_results_path: str | Path | None = None,
    wti_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Orchestrator: run price impact analysis if model has positive OOS R-squared.

    Steps:
    1. Load model predictions and WTI prices
    2. Compute SAR surprise
    3. Test correlation with EIA surprise
    4. Run trading simulation
    5. Generate plots
    6. Save results to results/price_impact.json

    Parameters
    ----------
    model_results_path : Path to analysis results JSON with OOS R-squared info.
    wti_path : Path to WTI price data (CSV or Parquet).
    output_dir : Output directory for results and figures.

    Returns
    -------
    Dict with price impact analysis results.
    """
    if model_results_path is None:
        model_results_path = Path(RESULTS_DIR) / "analysis_results.json"
    if output_dir is None:
        output_dir = Path(RESULTS_DIR)

    model_results_path = Path(model_results_path)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 0: Check if model has positive OOS R-squared ──
    results = {"phase": "price_impact"}

    if model_results_path.exists():
        with open(model_results_path) as f:
            model_results = json.load(f)

        # Check OOS R2 from test_5_out_of_time
        oot = model_results.get("test_5_out_of_time", [])
        has_positive_r2 = any(r.get("positive_r2", False) for r in oot)

        if not has_positive_r2:
            results["skipped"] = True
            results["reason"] = "No feature has positive out-of-sample R-squared. Price impact analysis requires positive OOS R2."
            logger.info("Price impact analysis SKIPPED: no positive OOS R2")

            # Save
            out_json = output_dir / "price_impact.json"
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2, default=str)
            return results
    else:
        logger.warning("Model results not found at %s; attempting price impact anyway.", model_results_path)
        model_results = {}

    # ── Step 1: Load SAR estimates and WTI prices ──
    logger.info("Step 1: Loading data for price impact analysis")

    analysis_ready_path = Path(FEATURES_DIR) / "analysis_ready.parquet"
    if not analysis_ready_path.exists():
        results["error"] = f"Analysis-ready data not found at {analysis_ready_path}"
        return results

    sar_df = pd.read_parquet(analysis_ready_path)
    if not isinstance(sar_df.index, pd.DatetimeIndex):
        if "timestamp" in sar_df.columns:
            sar_df = sar_df.set_index("timestamp")
        sar_df.index = pd.to_datetime(sar_df.index)

    # Load EIA
    eia_path = Path(EIA_DIR) / "cushing_stocks.parquet"
    if not eia_path.exists():
        results["error"] = f"EIA data not found at {eia_path}"
        return results

    eia_df = pd.read_parquet(eia_path)
    if "date" in eia_df.columns:
        eia_df = eia_df.set_index("date")
    eia_df.index = pd.to_datetime(eia_df.index)
    if eia_df.index.tz is not None:
        eia_df.index = eia_df.index.tz_localize(None)

    # Load WTI prices
    wti_df = None
    if wti_path is not None:
        wti_path = Path(wti_path)
    else:
        # Try common locations
        candidates = [
            Path(EIA_DIR) / "wti_prices.parquet",
            Path(EIA_DIR) / "wti_prices.csv",
            Path(FEATURES_DIR) / "wti_prices.parquet",
        ]
        for candidate in candidates:
            if candidate.exists():
                wti_path = candidate
                break

    if wti_path is not None and wti_path.exists():
        if wti_path.suffix == ".csv":
            wti_df = pd.read_csv(wti_path, parse_dates=True, index_col=0)
        else:
            wti_df = pd.read_parquet(wti_path)
        if not isinstance(wti_df.index, pd.DatetimeIndex):
            wti_df.index = pd.to_datetime(wti_df.index)
        if wti_df.index.tz is not None:
            wti_df.index = wti_df.index.tz_localize(None)
        logger.info("Loaded WTI prices: %d rows", len(wti_df))
    else:
        # Attempt download via yfinance
        try:
            import yfinance as yf
            logger.info("Downloading WTI prices from Yahoo Finance...")
            wti_ticker = yf.Ticker(WTI_FUTURES_TICKER)
            wti_df = wti_ticker.history(start="2017-01-01", end="2026-03-01")
            if wti_df.index.tz is not None:
                wti_df.index = wti_df.index.tz_localize(None)
            logger.info("Downloaded %d WTI price rows", len(wti_df))
        except Exception as exc:
            results["error"] = f"WTI price data not found and yfinance download failed: {exc}"
            logger.error(results["error"])
            return results

    if wti_df is None or len(wti_df) == 0:
        results["error"] = "No WTI price data available"
        return results

    # Determine price column
    price_col = "Close"
    if price_col not in wti_df.columns:
        # Try alternatives
        for alt in ["close", "Adj Close", "adj_close", "price"]:
            if alt in wti_df.columns:
                price_col = alt
                break
        else:
            # Use first numeric column
            numeric_cols = wti_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
            else:
                results["error"] = "No numeric price column found in WTI data"
                return results

    # ── Step 2: Compute SAR surprise ──
    logger.info("Step 2: Computing SAR surprise")

    # Use wind_corrected if available, else mean_db
    sar_col = "wind_corrected" if "wind_corrected" in sar_df.columns else "mean_db"
    surprise_df = compute_sar_surprise(sar_df, eia_df, sar_col=sar_col)

    if len(surprise_df) == 0:
        results["error"] = "Could not compute SAR surprise (no aligned observations)"
        return results

    # ── Step 3: Test correlation between SAR surprise and EIA surprise ──
    logger.info("Step 3: Testing SAR surprise vs EIA surprise correlation")

    mask = surprise_df["sar_surprise"].notna() & surprise_df["eia_surprise"].notna()
    clean = surprise_df[mask]
    if len(clean) >= 10:
        r_surprise, p_surprise = stats.pearsonr(clean["sar_surprise"].values, clean["eia_surprise"].values)
        rho_surprise, rho_p = stats.spearmanr(clean["sar_surprise"].values, clean["eia_surprise"].values)
        results["surprise_correlation"] = {
            "pearson_r": float(r_surprise),
            "pearson_p": float(p_surprise),
            "spearman_rho": float(rho_surprise),
            "spearman_p": float(rho_p),
            "n_obs": int(len(clean)),
        }
        logger.info(
            "SAR surprise vs EIA surprise: r=%.3f (p=%.4f), rho=%.3f (p=%.4f)",
            r_surprise, p_surprise, rho_surprise, rho_p,
        )
    else:
        results["surprise_correlation"] = {"error": "insufficient_data", "n_obs": int(len(clean))}

    # ── Step 4: Run trading simulation ──
    logger.info("Step 4: Running trading simulation")

    trade_log, trade_summary = simulate_trading(
        surprise_df, wti_df, threshold=0.5, price_col=price_col,
    )
    results["trading_simulation"] = trade_summary

    # ── Step 5: Generate plots ──
    logger.info("Step 5: Generating price impact plots")

    if len(trade_log) > 0:
        plot_price_impact(trade_log, output_dir=figures_dir)
        results["figures"] = [str(figures_dir / "price_impact.png")]
    else:
        results["figures"] = []

    # ── Step 6: Save results ──
    out_json = output_dir / "price_impact.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Price impact results saved to %s", out_json)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 13. Master Pipeline — Generate All Plots
# ═════════════════════════════════════════════════════════════════════════════

def generate_all_plots(
    analysis_ready_path: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> list[Path]:
    """Run all visualization functions using whatever data is available.

    Handles missing files gracefully — each plot is attempted independently.

    Parameters
    ----------
    analysis_ready_path : Path to ``analysis_ready.parquet``.
    results_dir : Path to results directory containing ``analysis_results.json``.

    Returns
    -------
    List of paths to generated figures.
    """
    if analysis_ready_path is None:
        analysis_ready_path = Path(FEATURES_DIR) / "analysis_ready.parquet"
    if results_dir is None:
        results_dir = Path(RESULTS_DIR)

    analysis_ready_path = Path(analysis_ready_path)
    results_dir = Path(results_dir)
    figures_dir = Path(FIGURES_DIR)
    figures_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    # ── Load analysis-ready data ──
    df = None
    if analysis_ready_path.exists():
        try:
            df = pd.read_parquet(analysis_ready_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            logger.info("Loaded analysis-ready data: %d rows", len(df))
        except Exception as exc:
            logger.warning("Failed to load analysis-ready data: %s", exc)
    else:
        logger.warning("Analysis-ready data not found at %s", analysis_ready_path)

    # ── Load analysis results JSON ──
    results = {}
    results_json = results_dir / "analysis_results.json"
    if results_json.exists():
        try:
            with open(results_json) as f:
                results = json.load(f)
            logger.info("Loaded analysis results from %s", results_json)
        except Exception as exc:
            logger.warning("Failed to load analysis results: %s", exc)

    # ── Load control data ──
    control_df = None
    control_path = Path(FEATURES_DIR) / "control_weekly.parquet"
    if control_path.exists():
        try:
            control_df = pd.read_parquet(control_path)
            if not isinstance(control_df.index, pd.DatetimeIndex):
                control_df.index = pd.to_datetime(control_df.index)
            control_df = control_df.sort_index()
        except Exception as exc:
            logger.warning("Failed to load control data: %s", exc)

    # ── Load EIA data ──
    eia_df = None
    eia_path = Path(EIA_DIR) / "cushing_stocks.parquet"
    if eia_path.exists():
        try:
            eia_df = pd.read_parquet(eia_path)
            if "date" in eia_df.columns:
                eia_df = eia_df.set_index("date")
            eia_df.index = pd.to_datetime(eia_df.index)
            if eia_df.index.tz is not None:
                eia_df.index = eia_df.index.tz_localize(None)
            eia_df = eia_df.sort_index()
        except Exception as exc:
            logger.warning("Failed to load EIA data: %s", exc)

    # ── 1. Backscatter time series ──
    if df is not None and "mean_db" in df.columns:
        try:
            path = plot_backscatter_timeseries(df, figures_dir, control_df=control_df)
            generated.append(path)
        except Exception as exc:
            logger.warning("Failed to plot backscatter timeseries: %s", exc)

    # ── 2. EIA stocks time series ──
    if eia_df is not None and "value" in eia_df.columns:
        try:
            path = plot_eia_stocks_timeseries(eia_df, figures_dir)
            generated.append(path)
        except Exception as exc:
            logger.warning("Failed to plot EIA stocks timeseries: %s", exc)

    # ── 3. First-diff scatter ──
    if df is not None and "delta_mean_db" in df.columns and "delta_value" in df.columns:
        try:
            mask = df["delta_mean_db"].notna() & df["delta_value"].notna()
            sar_vals = df.loc[mask, "delta_mean_db"].values
            eia_vals = df.loc[mask, "delta_value"].values
            if len(sar_vals) >= 10:
                rho, p_val = stats.spearmanr(sar_vals, eia_vals)
                path = plot_first_diff_scatter(sar_vals, eia_vals, rho, p_val, figures_dir)
                generated.append(path)
        except Exception as exc:
            logger.warning("Failed to plot first-diff scatter: %s", exc)

    # ── 4. Correlation heatmap ──
    if results:
        try:
            path = plot_correlation_heatmap(results, figures_dir)
            generated.append(path)
        except Exception as exc:
            logger.warning("Failed to plot correlation heatmap: %s", exc)

    # ── 5. Year stability ──
    year_data = results.get("test_4_year_stability", [])
    if year_data:
        try:
            path = plot_year_stability(year_data, figures_dir)
            generated.append(path)
        except Exception as exc:
            logger.warning("Failed to plot year stability: %s", exc)

    # ── 6. Lead-lag ──
    if df is not None and "mean_db" in df.columns and "value" in df.columns:
        try:
            from src.analysis import lead_lag_crosscorrelation
            ll = lead_lag_crosscorrelation(df["mean_db"], df["value"], max_lag_weeks=12)
            if len(ll) > 0:
                path = plot_lead_lag(ll, figures_dir, feature_name="mean_db")
                generated.append(path)
        except Exception as exc:
            logger.warning("Failed to plot lead-lag: %s", exc)

    # ── 7. Control comparison ──
    if df is not None and control_df is not None and "mean_db" in df.columns and "mean_db" in control_df.columns:
        try:
            path = plot_control_comparison(df, control_df, figures_dir)
            generated.append(path)
        except Exception as exc:
            logger.warning("Failed to plot control comparison: %s", exc)

    # ── 8. Kill criterion dashboard ──
    if results and "kill_criterion" in results:
        try:
            path = plot_kill_criterion_summary(results, figures_dir)
            generated.append(path)
        except Exception as exc:
            logger.warning("Failed to plot kill criterion: %s", exc)

    # ── Price impact (Phase 6 — conditional) ──
    kill_info = results.get("kill_criterion", {})
    if not kill_info.get("kill", True):
        try:
            price_results = run_price_impact_analysis(output_dir=results_dir)
            if "figures" in price_results:
                generated.extend([Path(p) for p in price_results["figures"]])
        except Exception as exc:
            logger.warning("Failed to run price impact analysis: %s", exc)

    logger.info("Generated %d figures total", len(generated))
    return generated


# ═════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n" + "=" * 70)
    print("  CUSHING SAR — VISUALIZATION + PRICE IMPACT ANALYSIS")
    print("=" * 70)

    generated = generate_all_plots()

    print(f"\nGenerated {len(generated)} figures:")
    for p in generated:
        print(f"  - {p}")

    if not generated:
        print("  (No figures generated — check that data files exist)")
        print(f"  Expected analysis-ready at: {Path(FEATURES_DIR) / 'analysis_ready.parquet'}")
        print(f"  Expected results at: {Path(RESULTS_DIR) / 'analysis_results.json'}")
