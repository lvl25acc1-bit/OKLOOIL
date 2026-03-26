# Cushing, Oklahoma — SAR-Based Oil Storage Estimation from Sentinel-1

## TL;DR

Free Sentinel-1 C-band SAR at 20 m resolution **cannot** reliably estimate crude oil inventory at Cushing, Oklahoma. Raw level correlations (r = -0.53) are entirely spurious — first-differencing eliminates the signal (rho < 0.02 for all features except one marginal case at -0.10). This is a rigorous negative result: 371 SAR scenes, 8 statistical tests, and a pre-registered kill criterion confirm that 20 m resolution is below the floor needed for aggregate backscatter-based storage estimation. Commercial providers achieve r = 0.98 using 1-3 m SAR; this project quantifies why free data cannot replicate that.

## Background

Cushing, Oklahoma is the physical delivery point for WTI crude oil futures and the largest commercial crude storage hub in the United States (~91 million barrels capacity, ~431 floating-roof tanks). Weekly inventory reports from the EIA move global oil prices.

Commercial satellite analytics firms (Ursa Space, Kayrros) monitor Cushing tank levels using 1-3 m SAR (ICEYE, COSMO-SkyMed), achieving correlations of r = 0.98 with EIA data by measuring individual floating-roof positions. This project tests whether the same can be accomplished with **free** Sentinel-1 data at 20 m resolution, using aggregate backscatter statistics rather than individual tank measurement.

## Method

### Data
- **SAR**: 371 Sentinel-1A ascending scenes (orbit 34), 2017-01-01 to 2026-02-19, sourced as pre-calibrated RTC gamma-naught from Microsoft Planetary Computer (free, no authentication required)
- **Ground truth**: EIA weekly Cushing crude stocks (478 observations, range 20,038-69,420 thousand barrels)
- **Weather**: ERA5 hourly reanalysis via Open-Meteo (wind, precipitation, temperature, soil moisture) for confound correction

### ROI Masks
Manual polygons traced over two tank farm clusters (north and south Cushing) plus a control grassland area, stored as GeoJSON.

### Statistical Battery (8 tests)
1. **Raw Pearson/Spearman correlation** — baseline (expected to show signal due to co-trending)
2. **First-difference correlation** — the kill test (do *changes* in backscatter track *changes* in inventory?)
3. **STL-detrended correlation** — seasonal/trend decomposition
4. **Within-year stability** — does the sign of correlation hold across all years?
5. **Out-of-time validation** — train on 2017-2023, predict 2024-2025
6. **Granger causality** — does SAR predict future EIA reports (and vice versa)?
7. **Wind confound test** — partial correlation controlling for wind speed
8. **Benjamini-Hochberg FDR** — multiple comparison correction across all tests

### Kill Criterion
Inspired by the Santos soybean lesson (level r = 0.60, first-diff r = 0.01 — entirely spurious): the project is killed if all first-difference correlations have |rho| < 0.1.

## Results

### Key Test Results

| Test | Feature | Value | Interpretation |
|------|---------|-------|----------------|
| Raw correlation | std_db | r = -0.53, p < 1e-25 | Looks promising but is spurious |
| First-diff (KILL) | delta_mean_db | rho = -0.02, p = 0.69 | No signal |
| First-diff (KILL) | delta_control_diff | rho = -0.10, p = 0.06 | Marginal, not significant |
| First-diff (KILL) | delta_wind_corrected | rho = -0.02, p = 0.74 | No signal |
| First-diff (KILL) | delta_seasonal_residual | rho = -0.02, p = 0.74 | No signal |
| Out-of-time R-squared | control_diff | R2 = -11.0 | Catastrophically worse than mean |
| Out-of-time R-squared | std_db | R2 = -23.6 | Catastrophically worse than mean |
| Granger (SAR->EIA) | mean_db | min p = 0.26 | No predictive power |
| Granger (SAR->EIA) | std_db | min p = 0.62 | No predictive power |

### First-Difference Scatter

![First-difference scatter](results/figures/first_diff_scatter.png)

The scatter plot of delta_mean_db vs. delta_stocks shows a shapeless cloud with no discernible relationship — the hallmark of a spurious level correlation killed by differencing.

### Verdict

**NEGATIVE RESULT.** The raw correlation (r = -0.53) is entirely explained by shared seasonal patterns and multi-year trends between SAR backscatter and oil inventory. First-differencing reveals no observation-to-observation relationship. The one marginal exception (delta_control_diff, rho = -0.10, p = 0.06) is too weak and not statistically significant.

## Key Lessons

1. **Santos lesson confirmed.** Level correlations between SAR backscatter and commodity stocks can be strongly spurious. First-differencing is mandatory before drawing conclusions. This project reproduced the exact pattern: strong level correlation, near-zero first-difference correlation.

2. **Resolution floor quantified.** At 20 m, individual tanks (30-80 m diameter) occupy only 1-4 pixels. Floating-roof height changes produce sub-pixel geometric effects that are below the noise floor. Commercial providers need 1-3 m resolution (30-80 pixels per tank) for direct roof-edge measurement.

3. **Methodology is reusable.** The 8-test statistical battery with a pre-registered kill criterion correctly identified the spurious correlation and can be applied to other SAR-commodity studies.

4. **Negative results have scientific value.** This is the first systematic quantification (to our knowledge) of the resolution floor for SAR-based oil storage estimation, filling a gap in published literature.

## Quick Start

```bash
# Set up environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download SAR data from Planetary Computer (free, no auth needed)
python src/download_pc.py --rel-orbit 34

# Run the pipeline
PYTHONPATH=. python src/preprocess.py
PYTHONPATH=. python src/features.py
PYTHONPATH=. python src/analysis.py
```

## Project Structure

```
OILOKLOHAMA/
├── src/
│   ├── config.py           # Constants: bounding boxes, thresholds, paths
│   ├── download_pc.py      # Sentinel-1 RTC download from Planetary Computer
│   ├── preprocess.py       # Speckle filtering, ROI backscatter extraction
│   ├── features.py         # Weekly aggregation, first-diffs, wind correction
│   ├── analysis.py         # 8-test statistical battery + kill criterion
│   ├── model.py            # ML models (Ridge, XGBoost) — confirms negative result
│   ├── visualize.py        # Diagnostic plots and figure generation
│   ├── ground_truth.py     # EIA API stocks + yfinance WTI prices
│   └── fetch_weather.py    # ERA5 weather from Open-Meteo
├── data/
│   ├── roi/                # Manual tank farm + control masks (GeoJSON)
│   ├── cropped/            # Cropped SAR GeoTIFFs (generated, not tracked)
│   ├── eia/                # EIA stocks parquet (generated)
│   ├── era5/               # ERA5 weather parquet (generated)
│   └── features/           # Analysis-ready parquet (generated)
├── results/
│   ├── statistical_tests.json    # Full test battery output
│   ├── negative_result_report.md # Detailed findings
│   └── figures/                  # Diagnostic plots (generated)
├── Manual_mask.gpkg        # QGIS-format tank farm masks
├── CLAUDE.md               # Development notes
├── requirements.txt
└── README.md
```

## Data Sources

| Source | What | Access |
|--------|------|--------|
| [Planetary Computer](https://planetarycomputer.microsoft.com/) | Sentinel-1 RTC (gamma-naught, VV) | Free, no authentication |
| [EIA API v2](https://api.eia.gov/) | Weekly Cushing crude stocks | Free (optional API key) |
| [Open-Meteo](https://open-meteo.com/) | ERA5 hourly weather reanalysis | Free, no authentication |

## License

MIT
