# Cushing Oklahoma SAR Oil Storage Estimation

Determine whether Sentinel-1 C-band SAR (20m resolution) can estimate crude oil inventory at Cushing, Oklahoma (~91M bbl capacity, ~431 floating-roof tanks) using aggregate backscatter statistics. A rigorous negative result is a valid outcome.

## Quick Reference

**Python:** `.venv/bin/python` (Python 3.12 venv)

**SAR Download:** Use the original downloader on the external SSD:
```bash
cd /Users/davidbass/Documents/FinancelModels
python sentinel1_downloader.py --bbox -97.05 35.90 -96.65 36.10 \
    --start 2017-01-01 --end 2026-03-01 --max-results 1000 \
    --output /Volumes/Extreme\ Pro/Sentinel1
```
Note: `src/download.py` is an adapted copy with crop support, but the original downloader at `/Users/davidbass/Documents/FinancelModels/sentinel1_downloader.py` is what's actually used for downloads. Raw zips go to `/Volumes/Extreme Pro/Sentinel1`.

## Pipeline Phases

### Phase 1: Data Acquisition
```bash
# Fetch EIA Cushing stocks + WTI prices (cached to data/eia/)
.venv/bin/python -m src.ground_truth

# Fetch ERA5 weather via Open-Meteo (cached to data/era5/)
.venv/bin/python -m src.fetch_weather

# Crop existing SAR zips to Cushing AOI (after downloading to external SSD)
.venv/bin/python -m src.download --crop-only \
    --output /Volumes/Extreme\ Pro/Sentinel1 \
    --crop-output data/cropped
```

### Phase 2: Preprocessing
```bash
# Extract per-ROI backscatter stats from cropped scenes
# Output: data/features/roi_backscatter.parquet
.venv/bin/python -m src.preprocess
```

### Phase 3: Feature Engineering
```bash
# Build analysis-ready dataset (weekly aggregation, wind correction, etc.)
# Output: data/features/analysis_ready.parquet
.venv/bin/python -m src.features
```

### Phase 4: Statistical Analysis (GO/NO-GO)
```bash
# Run all 8 tests + control null test
# Output: results/statistical_tests.json, results/figures/
.venv/bin/python -m src.analysis
```

## File Structure

```
OILOKLOHAMA/
├── src/
│   ├── config.py          # All constants: bboxes, thresholds, paths, EIA series IDs
│   ├── download.py        # ASF search + /vsizip/ crop (adapted from sentinel1_downloader.py)
│   ├── preprocess.py      # DN->dB calibration, Lee speckle filter, ROI stat extraction
│   ├── features.py        # Weekly aggregation, first-diffs, wind RLM, control differencing
│   ├── analysis.py        # 8-test battery: raw corr, first-diff, STL, Granger, FDR, etc.
│   ├── ground_truth.py    # EIA API v2 stocks + yfinance WTI prices
│   └── fetch_weather.py   # ERA5 hourly weather from Open-Meteo (wind, precip, soil moisture)
├── data/
│   ├── raw/               # Raw SAFE zips (or on external SSD)
│   ├── cropped/           # Cropped VV GeoTIFFs (AOI subset)
│   ├── roi/               # cushing_tank_farms.geojson (tank farm + control polygons)
│   ├── eia/               # cushing_stocks.parquet, wti_prices.parquet
│   ├── era5/              # cushing_weather_hourly.parquet, cushing_weather_daily.parquet
│   └── features/          # roi_backscatter.parquet, analysis_ready.parquet
├── results/
│   ├── figures/           # Diagnostic plots
│   └── statistical_tests.json
├── notebooks/
├── requirements.txt
└── CLAUDE.md
```

## Important Constants (src/config.py)

| Constant | Value | Purpose |
|----------|-------|---------|
| `CUSHING_BBOX` | (-96.800, 35.892, -96.681, 36.053) | Tight crop box around tank farms |
| `CUSHING_BBOX_SEARCH` | (-97.05, 35.90, -96.65, 36.10) | Broad box for ASF scene search |
| `DB_FLOOR` | -30.0 dB | Backscatter floor for calibration |
| `SPECKLE_FILTER_SIZE` | 3 | Lee filter window |
| `BRIGHT_SCATTERER_THRESHOLD` | -5.0 dB | Bright pixel ratio cutoff |
| `FDR_THRESHOLD` | 0.05 | Benjamini-Hochberg q-value |
| `KILL_THRESHOLD` | 0.1 | Min first-difference abs(rho) to proceed |
| `MIN_OBSERVATIONS` | 26 | Min weeks for correlation |
| `TRAIN_END` | 2023-12-31 | Train/test split |
| `DATA_START` / `DATA_END` | 2017-01-01 / 2026-03-01 | Full study period |

## Kill Criterion

The project is killed if ALL of these first-difference correlations have |rho| < 0.1:
- `delta_mean_db` vs `delta_stocks`
- `delta_control_diff` vs `delta_stocks`
- `delta_wind_corrected` vs `delta_stocks`
- `delta_seasonal_residual` vs `delta_stocks`

This prevents reporting spurious level correlations (lesson from Santos: rho=0.597 level, rho=0.010 first-diff).

## Key Lessons from Prior Projects

- **Santos**: Level correlation was 0.597 but first-diff was 0.010 (spurious). Always test first-differences first.
- **Rotterdam**: Wind explained 43% of vessel count variance. Always correct for wind.

## Dependencies

See `requirements.txt`. Key packages: asf_search, rasterio, shapely, pandas, scipy, statsmodels, xgboost, yfinance.

## External Data Paths

- SAR downloads: `/Volumes/Extreme Pro/Sentinel1` (external SSD)
- Original downloader: `/Users/davidbass/Documents/FinancelModels/sentinel1_downloader.py`
