"""SAR preprocessing and ROI backscatter extraction pipeline.

Converts Sentinel-1 RTC (linear power) to dB, applies speckle filtering,
and extracts per-ROI backscatter statistics for downstream modelling.

Supports both:
- Planetary Computer RTC data (EPSG:326xx UTM, gamma-naught linear power)
- ASF GRD data (may be EPSG:4326, amplitude DN)
Handles CRS mismatch between raster and ROI polygons automatically.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from scipy.ndimage import uniform_filter
from scipy.stats import skew, kurtosis
from shapely.geometry import shape, mapping

from src.config import (
    DB_FLOOR,
    SPECKLE_FILTER_SIZE,
    BRIGHT_SCATTERER_THRESHOLD,
    CROPPED_DIR,
    RAW_DIR,
    ROI_DIR,
    FEATURES_DIR,
)

logger = logging.getLogger(__name__)

# Minimum pixel count to consider an ROI extraction valid
MIN_PIXEL_COUNT = 50

SCENE_KIND_SUFFIXES = {
    "rtc_vv": "_rtc_vv",
    "vv_crop": "_vv_crop",
}


def infer_scene_kind(scene_name: str | Path) -> str:
    """Infer scene product family from the filename stem."""
    stem = Path(scene_name).stem
    for scene_kind, suffix in SCENE_KIND_SUFFIXES.items():
        if stem.endswith(suffix):
            return scene_kind
    return "unknown"


def parse_scene_kind_filter(scene_kind: str | None) -> set[str] | None:
    """Parse a scene-kind filter string into a set, or None for no filtering."""
    if scene_kind is None:
        return {"rtc_vv"}

    normalized = scene_kind.strip().lower()
    if normalized in {"", "all"}:
        return None

    scene_kinds = {part.strip().lower() for part in normalized.split(",") if part.strip()}
    if not scene_kinds:
        return {"rtc_vv"}

    invalid = scene_kinds - set(SCENE_KIND_SUFFIXES)
    if invalid:
        valid = ", ".join(sorted(SCENE_KIND_SUFFIXES))
        raise ValueError(f"Unsupported scene kinds: {sorted(invalid)}. Valid values: {valid}, all")

    return scene_kinds


# ---------------------------------------------------------------------------
# 1. Calibration
# ---------------------------------------------------------------------------

def dn_to_sigma0_db(data: np.ndarray) -> np.ndarray:
    """Convert linear power DN values to sigma-naught in dB.

    sigma0_db = 10 * log10(linear)
    with a floor at DB_FLOOR to avoid log(0).
    """
    linear = data.astype(np.float64)
    linear = np.where(linear > 0, linear, 1e-10)
    db = 10.0 * np.log10(linear)
    db = np.clip(db, DB_FLOOR, None)
    return db


# ---------------------------------------------------------------------------
# 2. Speckle filter
# ---------------------------------------------------------------------------

def lee_speckle_filter(
    data: np.ndarray,
    valid_mask: np.ndarray | None = None,
    size: int = SPECKLE_FILTER_SIZE,
) -> np.ndarray:
    """Apply a Lee-style speckle filter to linear power data.

    The filter operates in the linear domain and estimates local mean/variance
    while ignoring nodata pixels outside the ROI mask.
    """
    linear = data.astype(np.float64)
    if valid_mask is None:
        valid_mask = linear > 0
    else:
        valid_mask = valid_mask.astype(bool)

    if valid_mask.sum() == 0:
        return np.zeros_like(linear, dtype=np.float64)

    size = max(int(size), 1)
    if size == 1:
        return np.where(valid_mask, linear, 0.0)

    window_area = float(size * size)
    data_valid = np.where(valid_mask, linear, 0.0)
    local_count = uniform_filter(valid_mask.astype(np.float64), size=size, mode="reflect") * window_area
    local_count = np.where(local_count > 0, local_count, np.nan)

    local_sum = uniform_filter(data_valid, size=size, mode="reflect") * window_area
    local_sum_sq = uniform_filter(data_valid ** 2, size=size, mode="reflect") * window_area

    local_mean = local_sum / local_count
    local_mean_sq = local_sum_sq / local_count
    local_var = np.maximum(local_mean_sq - local_mean ** 2, 0.0)

    valid_local_var = local_var[valid_mask & np.isfinite(local_var)]
    if valid_local_var.size == 0:
        return np.where(valid_mask, linear, 0.0)

    noise_var = float(np.median(valid_local_var))
    if not np.isfinite(noise_var) or noise_var < 0:
        noise_var = 0.0

    denominator = local_var + noise_var
    weights = np.where(denominator > 0, local_var / denominator, 0.0)
    filtered = local_mean + weights * (linear - local_mean)
    filtered = np.where(valid_mask, np.maximum(filtered, 0.0), 0.0)
    return filtered


# ---------------------------------------------------------------------------
# 3. ROI statistics extraction
# ---------------------------------------------------------------------------

def extract_roi_stats(
    scene_path: str | Path,
    roi_geojson_path: str | Path,
) -> list[dict]:
    """Extract per-ROI backscatter statistics from one cropped VV GeoTIFF.

    Steps:
        1. Load scene, calibrate DN -> sigma0 dB
        2. Apply Lee speckle filter
        3. For each ROI polygon: mask, compute stats

    Returns a list of dicts (one per ROI).
    """
    scene_path = Path(scene_path)
    roi_geojson_path = Path(roi_geojson_path)

    # Load ROI polygons and reproject to match raster CRS
    roi_gdf = gpd.read_file(roi_geojson_path)

    with rasterio.open(scene_path) as src:
        scene_crs = src.crs

        # Reproject ROIs to raster CRS if needed (e.g. EPSG:4326 → UTM)
        if scene_crs and roi_gdf.crs and roi_gdf.crs != scene_crs:
            roi_gdf = roi_gdf.to_crs(scene_crs)

        # Check for all-zero data
        sample = src.read(1)
        if sample.max() == 0:
            logger.warning("All-zero data in %s — skipping", scene_path.name)
            return []

        results = []
        for _, roi_row in roi_gdf.iterrows():
            roi_name = roi_row["name"]
            geom = roi_row.geometry
            geom_dict = mapping(geom)

            try:
                masked, masked_transform = rasterio.mask.mask(
                    src, [geom_dict], crop=True, nodata=0, filled=True,
                )
            except (ValueError, rasterio.errors.WindowError):
                logger.warning(
                    "ROI '%s' does not overlap scene %s — skipping",
                    roi_name, scene_path.name,
                )
                continue

            # Filter in linear power, then convert to dB for statistics
            roi_raw = masked[0].astype(np.float64)
            valid_mask = roi_raw > 0
            pixel_count = int(valid_mask.sum())

            if pixel_count < MIN_PIXEL_COUNT:
                logger.warning(
                    "ROI '%s' has only %d valid pixels in %s — skipping",
                    roi_name, pixel_count, scene_path.name,
                )
                continue

            roi_filtered_linear = lee_speckle_filter(roi_raw, valid_mask=valid_mask)
            roi_filtered = dn_to_sigma0_db(roi_filtered_linear)

            # Extract valid pixels only
            vals = roi_filtered[valid_mask]

            stats = {
                "roi_name": roi_name,
                "mean_db": float(np.mean(vals)),
                "median_db": float(np.median(vals)),
                "std_db": float(np.std(vals)),
                "p10_db": float(np.percentile(vals, 10)),
                "p25_db": float(np.percentile(vals, 25)),
                "p75_db": float(np.percentile(vals, 75)),
                "p90_db": float(np.percentile(vals, 90)),
                "skewness": float(skew(vals)),
                "kurtosis": float(kurtosis(vals)),
                "bright_pixel_ratio": float(
                    (vals > BRIGHT_SCATTERER_THRESHOLD).sum() / len(vals)
                ),
                "pixel_count": pixel_count,
            }
            results.append(stats)

    return results


# ---------------------------------------------------------------------------
# 4. Metadata loader
# ---------------------------------------------------------------------------

def load_metadata(scene_filename: str, cropped_dir: str | Path) -> dict:
    """Load metadata from a .meta.json sidecar file matching a scene.

    Supports both formats:
    - Planetary Computer sidecar (next to the .tif in cropped_dir):
        S1A_IW_GRDH_..._rtc_vv.meta.json  with keys: datetime, orbit_direction, rel_orbit, platform
    - ASF sidecar (in raw_dir):
        <granuleID>.meta.json with keys: startTime, flightDirection, pathNumber, platform
    """
    cropped_dir = Path(cropped_dir)
    stem = Path(scene_filename).stem  # e.g. S1A_IW_..._rtc_vv

    defaults = {
        "startTime": None,
        "flightDirection": "UNKNOWN",
        "pathNumber": -1,
        "platform": "UNKNOWN",
    }

    # Look for sidecar next to the TIF (Planetary Computer format)
    meta_path = cropped_dir / f"{stem}.meta.json"
    if not meta_path.exists():
        # Try raw_dir for ASF format
        raw_dir = Path(RAW_DIR)
        meta_files = list(raw_dir.glob("*.meta.json"))
        meta_path = None
        for mf in meta_files:
            if stem.split("_rtc_")[0] in mf.name or stem.split("_vv")[0] in mf.name:
                meta_path = mf
                break

    if meta_path is None or not meta_path.exists():
        # Try parsing timestamp from filename as fallback
        # Format: S1A_IW_GRDH_1SDV_20170101T002818_...
        parts = scene_filename.split("_")
        for part in parts:
            if len(part) == 15 and "T" in part:
                try:
                    defaults["startTime"] = datetime.strptime(part, "%Y%m%dT%H%M%S")
                    break
                except ValueError:
                    pass
        # Extract platform from filename
        if scene_filename.startswith("S1A"):
            defaults["platform"] = "SENTINEL-1A"
        elif scene_filename.startswith("S1B"):
            defaults["platform"] = "SENTINEL-1B"
        elif scene_filename.startswith("S1C"):
            defaults["platform"] = "SENTINEL-1C"
        return defaults

    try:
        with open(meta_path) as f:
            meta = json.load(f)

        # Planetary Computer format
        if "datetime" in meta:
            dt_str = meta["datetime"]
            try:
                start_time = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            except ValueError:
                start_time = None
            return {
                "startTime": start_time,
                "flightDirection": meta.get("orbit_direction", "UNKNOWN").upper(),
                "pathNumber": meta.get("rel_orbit", -1),
                "platform": meta.get("platform", "UNKNOWN").upper(),
            }
        # ASF format
        elif "startTime" in meta:
            return {
                "startTime": datetime.fromisoformat(
                    meta["startTime"].replace("Z", "+00:00")
                ),
                "flightDirection": meta.get("flightDirection", "UNKNOWN"),
                "pathNumber": meta.get("relativeOrbit", meta.get("pathNumber", -1)),
                "platform": meta.get("platform", "UNKNOWN"),
            }
        else:
            return defaults

    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Error parsing meta file %s: %s", meta_path, exc)
        return defaults


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------

def process_all_scenes(
    cropped_dir: str | Path | None = None,
    roi_geojson_path: str | Path | None = None,
    raw_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    scene_kind: str | None = "rtc_vv",
) -> pd.DataFrame:
    """Process all cropped scenes and extract ROI backscatter statistics.

    Returns a DataFrame and saves it as parquet.
    """
    cropped_dir = Path(cropped_dir or CROPPED_DIR)
    roi_geojson_path = Path(roi_geojson_path or Path(ROI_DIR) / "cushing_tank_farms.geojson")
    raw_dir = Path(raw_dir or RAW_DIR)
    output_path = Path(output_path or Path(FEATURES_DIR) / "roi_backscatter.parquet")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_tif_files = sorted(glob.glob(str(cropped_dir / "*.tif")))
    allowed_scene_kinds = parse_scene_kind_filter(scene_kind)
    if allowed_scene_kinds is None:
        tif_files = all_tif_files
        filter_label = "all"
    else:
        tif_files = [
            tif_path for tif_path in all_tif_files
            if infer_scene_kind(tif_path) in allowed_scene_kinds
        ]
        filter_label = ",".join(sorted(allowed_scene_kinds))

    n_scenes = len(tif_files)
    logger.info(
        "Found %d scenes in %s after scene_kind filter '%s' (%d total .tif files)",
        n_scenes,
        cropped_dir,
        filter_label,
        len(all_tif_files),
    )

    if n_scenes == 0:
        print(f"No matching .tif files found in {cropped_dir} for scene_kind={filter_label}")
        return pd.DataFrame()

    rows = []
    for i, tif_path in enumerate(tif_files):
        scene_filename = Path(tif_path).name
        scene_id = Path(tif_path).stem
        current_scene_kind = infer_scene_kind(scene_filename)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"Processing scene {i + 1}/{n_scenes}: {scene_filename}")

        # Extract ROI stats
        try:
            roi_stats = extract_roi_stats(tif_path, roi_geojson_path)
        except Exception as exc:
            logger.error("Failed to process %s: %s", scene_filename, exc)
            continue

        # Load metadata (check cropped_dir first for PC sidecars, then raw_dir)
        meta = load_metadata(scene_filename, cropped_dir)

        for stats in roi_stats:
            row = {
                "scene_id": scene_id,
                "scene_kind": current_scene_kind,
                "timestamp": meta["startTime"],
                "flight_direction": meta["flightDirection"],
                "path_number": meta["pathNumber"],
                "platform": meta["platform"],
                **stats,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) > 0:
        # Ensure column order
        col_order = [
            "scene_id", "scene_kind", "timestamp", "flight_direction", "path_number",
            "platform", "roi_name", "mean_db", "median_db", "std_db",
            "p10_db", "p25_db", "p75_db", "p90_db", "skewness", "kurtosis",
            "bright_pixel_ratio", "pixel_count",
        ]
        df = df[[c for c in col_order if c in df.columns]]
        df.to_parquet(output_path, index=False)
        print(f"\nSaved {len(df)} rows to {output_path}")
    else:
        print("No valid extractions — output not saved.")

    return df


# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------

def validate_extraction(df: pd.DataFrame) -> None:
    """Run QA checks on the extracted backscatter DataFrame."""
    if df.empty:
        print("DataFrame is empty — nothing to validate.")
        return

    print("=" * 60)
    print("EXTRACTION VALIDATION")
    print("=" * 60)

    if "scene_kind" in df.columns:
        scene_kind_counts = df["scene_kind"].value_counts(dropna=False)
        print("\nScene kinds:")
        for scene_kind, count in scene_kind_counts.items():
            print(f"  {scene_kind}: {count} rows")
        non_unknown_kinds = [kind for kind in scene_kind_counts.index if kind != "unknown"]
        if len(non_unknown_kinds) > 1:
            print("  WARNING: multiple scene kinds present; do not use this output for one time series")

    for roi_name, grp in df.groupby("roi_name"):
        print(f"\n--- ROI: {roi_name} ({len(grp)} scenes) ---")

        # dB range check
        # NOTE: range depends on calibration. For raw GRD amplitude DN
        # (uint16, values ~50-500), 10*log10(DN) yields ~17-27 dB.
        # For radiometrically calibrated sigma0, expect [-25, 0] dB.
        mean_vals = grp["mean_db"]
        db_lo, db_hi = -30.0, 35.0
        out_of_range = ((mean_vals < db_lo) | (mean_vals > db_hi)).sum()
        if out_of_range > 0:
            print(f"  WARNING: {out_of_range} scenes with mean_db outside [{db_lo}, {db_hi}]")
        else:
            print(f"  OK: All mean_db values in [{db_lo}, {db_hi}]")

        # Pixel count stability
        pc = grp["pixel_count"]
        pc_std = pc.std()
        pc_mean = pc.mean()
        if pc_mean > 0 and pc_std / pc_mean > 0.10:
            print(
                f"  WARNING: pixel_count unstable "
                f"(mean={pc_mean:.0f}, std={pc_std:.0f}, "
                f"cv={pc_std / pc_mean:.2%})"
            )
        else:
            print(f"  OK: pixel_count stable (mean={pc_mean:.0f}, std={pc_std:.0f})")

        # Summary stats
        print(f"  mean_db:  {mean_vals.mean():.2f} +/- {mean_vals.std():.2f}")
        print(f"  median_db range: [{grp['median_db'].min():.2f}, {grp['median_db'].max():.2f}]")
        print(f"  bright_pixel_ratio: {grp['bright_pixel_ratio'].mean():.4f} (mean)")
        print(f"  pixel_count range: [{pc.min()}, {pc.max()}]")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_cli_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for extraction."""
    parser = argparse.ArgumentParser(description="Extract per-ROI SAR backscatter statistics.")
    parser.add_argument(
        "--cropped-dir",
        type=Path,
        default=Path(CROPPED_DIR),
        help="Directory containing cropped GeoTIFF scenes.",
    )
    parser.add_argument(
        "--roi-file",
        type=Path,
        default=Path(ROI_DIR) / "cushing_tank_farms.geojson",
        help="ROI GeoJSON or vector file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(RAW_DIR),
        help="Directory containing raw downloads and ASF sidecar metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(FEATURES_DIR) / "roi_backscatter.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--scene-kind",
        default="rtc_vv",
        help="Scene kind to include: rtc_vv, vv_crop, comma-separated list, or all. Default: rtc_vv.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the summary validation report after extraction.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = build_cli_parser().parse_args()

    print("Running Cushing SAR backscatter extraction pipeline...\n")
    df = process_all_scenes(
        cropped_dir=args.cropped_dir,
        roi_geojson_path=args.roi_file,
        raw_dir=args.raw_dir,
        output_path=args.output,
        scene_kind=args.scene_kind,
    )

    if not df.empty and not args.skip_validation:
        validate_extraction(df)
        print(f"\nPreview:\n{df.head(8).to_string()}")
