#!/usr/bin/env python3
"""
Sentinel-1 RTC downloader via Microsoft Planetary Computer.

Downloads only the cropped VV pixels within the AOI bbox directly from
cloud-optimized GeoTIFFs — no zip files, no full scenes. ~2 MB per scene
instead of ~1 GB.

Planetary Computer's sentinel-1-rtc collection provides radiometrically
terrain-corrected (RTC) Sentinel-1 data, already calibrated to gamma-naught
in linear power. No SAFE extraction or manual calibration needed.

No authentication required — Planetary Computer is free and open.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import planetary_computer
import rasterio
from pystac_client import Client
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely.geometry import box as shapely_box, shape as shapely_shape

# Project imports
try:
    from src.config import (
        CUSHING_BBOX, CUSHING_BBOX_SEARCH, CROPPED_DIR,
        DATA_START, DATA_END,
    )
except ImportError:
    CUSHING_BBOX = (-96.800, 35.892, -96.681, 36.053)
    CUSHING_BBOX_SEARCH = (-97.05, 35.90, -96.65, 36.10)
    CROPPED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cropped")
    DATA_START = "2017-01-01"
    DATA_END = "2026-03-01"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Planetary Computer STAC
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"

# Minimum overlap and pixel thresholds
MIN_OVERLAP_FRACTION = 0.05
MIN_PIXELS = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-1 RTC crops from Planetary Computer"
    )
    parser.add_argument("--bbox", type=float, nargs=4,
                        metavar=("W", "S", "E", "N"),
                        default=list(CUSHING_BBOX),
                        help="Crop bounding box (default: tight Cushing bbox)")
    parser.add_argument("--search-bbox", type=float, nargs=4,
                        metavar=("W", "S", "E", "N"),
                        default=list(CUSHING_BBOX_SEARCH),
                        help="Search bounding box (default: broad Cushing bbox)")
    parser.add_argument("--start", type=str, default=DATA_START,
                        help=f"Start date YYYY-MM-DD (default: {DATA_START})")
    parser.add_argument("--end", type=str, default=DATA_END,
                        help=f"End date YYYY-MM-DD (default: {DATA_END})")
    parser.add_argument("--output", type=str, default=CROPPED_DIR,
                        help=f"Output directory (default: {CROPPED_DIR})")
    parser.add_argument("--orbit", type=str, default=None,
                        help="Filter by orbit direction: ascending or descending")
    parser.add_argument("--rel-orbit", type=int, default=None,
                        help="Filter by relative orbit number")
    parser.add_argument("--search-only", action="store_true",
                        help="Only search, don't download")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of scenes to download")
    parser.add_argument("--asset", type=str, default="vv",
                        help="Asset key to download: vv, vh, or both (default: vv)")
    return parser.parse_args()


def search_scenes(search_bbox, start, end):
    """Search Planetary Computer STAC for Sentinel-1 RTC scenes."""
    logger.info("Connecting to Planetary Computer STAC...")
    client = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

    date_range = f"{start}/{end}"
    logger.info(f"Searching: bbox={search_bbox}, dates={date_range}")

    search = client.search(
        collections=[COLLECTION],
        bbox=list(search_bbox),
        datetime=date_range,
    )

    items = []
    for item in search.items():
        items.append(item)

    logger.info(f"Found {len(items)} scenes")
    return items


def get_item_metadata(item):
    """Extract metadata from a STAC item."""
    props = item.properties
    return {
        "id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime else props.get("datetime"),
        "platform": props.get("platform", ""),
        "orbit_direction": props.get("sat:orbit_state", props.get("s1:orbit_source", "")),
        "rel_orbit": props.get("sat:relative_orbit", props.get("s1:resolution", "")),
        "polarization": props.get("s1:polarization", ""),
        "resolution": props.get("s1:resolution", ""),
        "orbit_number": props.get("sat:absolute_orbit", ""),
    }


def filter_items(items, orbit_direction=None, rel_orbit=None):
    """Filter STAC items by orbit direction and relative orbit."""
    filtered = items

    if orbit_direction:
        filtered = [
            i for i in filtered
            if i.properties.get("sat:orbit_state", "").lower() == orbit_direction.lower()
        ]
        logger.info(f"After orbit direction filter ({orbit_direction}): {len(filtered)} scenes")

    if rel_orbit is not None:
        filtered = [
            i for i in filtered
            if i.properties.get("sat:relative_orbit") == rel_orbit
        ]
        logger.info(f"After relative orbit filter ({rel_orbit}): {len(filtered)} scenes")

    return filtered


def print_orbit_summary(items):
    """Print available orbits with scene counts."""
    orbit_counts = {}
    for item in items:
        props = item.properties
        direction = props.get("sat:orbit_state", "unknown")
        rel_orbit = props.get("sat:relative_orbit", "unknown")
        key = (rel_orbit, direction)
        orbit_counts[key] = orbit_counts.get(key, 0) + 1

    print(f"\n=== ORBIT SUMMARY ({len(items)} total scenes) ===")
    for (orb, direction), count in sorted(orbit_counts.items()):
        print(f"  Orbit {orb} ({direction}): {count} scenes")

    # Platform breakdown
    platforms = {}
    for item in items:
        p = item.properties.get("platform", "unknown")
        platforms[p] = platforms.get(p, 0) + 1
    print(f"\nPlatforms:")
    for p, count in sorted(platforms.items()):
        print(f"  {p}: {count} scenes")

    # Date range
    dates = sorted([
        item.datetime.strftime("%Y-%m-%d") if item.datetime else "unknown"
        for item in items
    ])
    if dates:
        print(f"\nDate range: {dates[0]} to {dates[-1]}")


def download_crop(item, asset_key, crop_bbox, output_dir):
    """Download a single asset cropped to bbox via windowed COG read.

    Returns (output_path, metadata_dict) or (None, None) on failure.
    """
    if asset_key not in item.assets:
        logger.warning(f"Asset '{asset_key}' not in {item.id}. Available: {list(item.assets.keys())}")
        return None, None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output filename
    dt_str = item.datetime.strftime("%Y%m%dT%H%M%S") if item.datetime else "unknown"
    out_name = f"{item.id}_{asset_key}.tif"
    out_path = output_dir / out_name

    # Skip if already downloaded
    if out_path.exists():
        logger.debug(f"Already exists: {out_path.name}")
        return out_path, get_item_metadata(item)

    # Check overlap
    site_box = shapely_box(*crop_bbox)
    try:
        scene_geom = shapely_shape(item.geometry)
        overlap = site_box.intersection(scene_geom).area / site_box.area
    except Exception:
        overlap = 0.0

    if overlap < MIN_OVERLAP_FRACTION:
        logger.info(f"Skipping {item.id}: only {overlap*100:.1f}% overlap")
        return None, None

    # Signed href (Planetary Computer requires token signing)
    href = item.assets[asset_key].href

    try:
        west, south, east, north = crop_bbox

        with rasterio.open(href) as src:
            # Reproject bbox to raster CRS if needed
            raster_crs = src.crs
            if raster_crs and raster_crs.to_epsg() != 4326:
                transformer = Transformer.from_crs(
                    "EPSG:4326", raster_crs, always_xy=True
                )
                west_t, south_t = transformer.transform(west, south)
                east_t, north_t = transformer.transform(east, north)
            else:
                west_t, south_t, east_t, north_t = west, south, east, north

            # Compute read window
            window = from_bounds(west_t, south_t, east_t, north_t, src.transform)
            window = window.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )

            if window.width < MIN_PIXELS or window.height < MIN_PIXELS:
                logger.info(f"Skipping {item.id}: crop too small ({int(window.width)}x{int(window.height)} px)")
                return None, None

            # Read only the windowed pixels
            data = src.read(1, window=window)
            win_transform = src.window_transform(window)

            profile = src.profile.copy()
            profile.update(
                width=data.shape[1],
                height=data.shape[0],
                count=1,
                transform=win_transform,
                driver="GTiff",
                compress="deflate",
            )

            with rasterio.open(str(out_path), "w", **profile) as dst:
                dst.write(data, 1)

        size_mb = out_path.stat().st_size / 1e6
        logger.info(f"Saved {out_path.name} ({data.shape[1]}x{data.shape[0]} px, {size_mb:.1f} MB)")

        # Save metadata sidecar
        meta = get_item_metadata(item)
        meta["crop_bbox"] = list(crop_bbox)
        meta["pixel_size"] = [data.shape[1], data.shape[0]]
        meta["file_size_mb"] = round(size_mb, 2)
        meta_path = out_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return out_path, meta

    except Exception as e:
        logger.error(f"Failed to download {item.id}/{asset_key}: {e}")
        if out_path.exists():
            out_path.unlink()
        return None, None


def download_all(items, asset_keys, crop_bbox, output_dir):
    """Download all scenes, cropped to bbox."""
    total = len(items)
    downloaded = 0
    skipped = 0
    failed = 0

    print(f"\n=== DOWNLOADING {total} SCENES ===")
    print(f"  Crop bbox: {crop_bbox}")
    print(f"  Assets: {asset_keys}")
    print(f"  Output: {output_dir}")
    print()

    for i, item in enumerate(items):
        for key in asset_keys:
            path, meta = download_crop(item, key, crop_bbox, output_dir)
            if path is not None:
                if path.stat().st_mtime < time.time() - 5:
                    skipped += 1  # already existed
                else:
                    downloaded += 1
            else:
                failed += 1

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{total} scenes processed")

    print(f"\n=== COMPLETE ===")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (exist): {skipped}")
    print(f"  Failed/skipped: {failed}")
    print(f"  Output: {output_dir}")
    return downloaded


def main():
    args = parse_args()

    crop_bbox = tuple(args.bbox)
    search_bbox = tuple(args.search_bbox)
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    # Search
    items = search_scenes(search_bbox, args.start, args.end)

    if not items:
        print("No scenes found.")
        sys.exit(1)

    # Print orbit summary
    print_orbit_summary(items)

    # Filter
    items = filter_items(items, args.orbit, args.rel_orbit)

    # Sort by date
    items.sort(key=lambda i: i.datetime or datetime.min)

    if args.search_only:
        print(f"\n(--search-only mode, {len(items)} scenes matched)")
        return

    if args.limit:
        items = items[:args.limit]
        print(f"\nLimited to {args.limit} scenes")

    # Download
    asset_keys = ["vv", "vh"] if args.asset == "both" else [args.asset]
    download_all(items, asset_keys, crop_bbox, output_dir)


if __name__ == "__main__":
    main()
