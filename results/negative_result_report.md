# Cushing SAR Oil Storage Estimation — Results Report

## Verdict: NEGATIVE (with one marginal exception)

Free Sentinel-1 C-band SAR at 20m resolution **cannot reliably estimate crude oil inventory levels at Cushing, Oklahoma**. The aggregate backscatter approach fails the rigorous statistical test battery designed to detect spurious correlations.

## Key Findings

### What worked
- Raw level correlations exist (Pearson r = -0.31 to -0.53, all p < 0.001)
- Tank farms show 2-4 dB higher backscatter than control grassland (physically expected: metal infrastructure reflects radar)
- Wind is NOT a confound (partial correlations survive wind removal)
- Planetary Computer RTC data provides clean, pre-calibrated gamma-naught without authentication hassles

### What failed (and why it matters)

| Test | Result | Interpretation |
|------|--------|----------------|
| First-difference (KILL TEST) | All |rho| < 0.09 except control_diff at 0.10 | **Changes** in backscatter don't track **changes** in inventory |
| Year stability | 2-4 sign flips across 9 years | Relationship is not consistent |
| Out-of-time R-squared | -12 to -129 | Model is catastrophically worse than predicting the mean |
| Granger causality | All p > 0.26 (SAR→EIA) | SAR has no predictive power for future EIA reports |

### The one marginal signal
`delta_control_diff` (tank backscatter change minus grassland backscatter change) shows rho = -0.102 (p = 0.062). This is:
- Just barely above the 0.1 kill threshold
- Not statistically significant at p < 0.05
- The correct physical quantity to test (atmospheric cancellation)
- Too weak and noisy for any practical use

## Why the raw correlation is spurious

The Santos soybean lesson applies perfectly here. Both SAR backscatter (std_db in particular) and EIA inventory have:
- **Shared seasonal patterns**: Oklahoma weather seasonality affects both SAR backscatter (through soil moisture, vegetation phenology) and oil storage patterns (seasonal demand cycles)
- **Shared trends**: Multi-year trends in both series create mechanical correlation

First-differencing removes these shared trends and reveals that there is no observation-to-observation relationship between SAR changes and inventory changes.

## Resolution analysis

At 20m Sentinel-1 resolution:
- Individual tanks (30-80m diameter) occupy 1-4 pixels
- The tank farm ROI contains ~175,000 pixels (south) and ~23,000 pixels (north)
- Floating-roof height changes of 10-15m produce sub-pixel geometric changes
- The backscatter signature change from roof displacement is below the noise floor at this resolution

Commercial providers (Ursa Space, Kayrros) achieve 0.98 correlation with EIA using 1-3m SAR (ICEYE, COSMO-SkyMed) because individual tanks are 30-80 pixels across, allowing direct roof-edge measurement.

## Data summary

- **SAR scenes**: 371 Sentinel-1A ascending orbit-34 scenes (2017-01-01 to 2026-02-19)
- **Source**: Microsoft Planetary Computer Sentinel-1 RTC (pre-calibrated gamma-naught)
- **EIA stocks**: 478 weekly observations (range: 20,038 - 69,420 thousand barrels)
- **ERA5 weather**: 80,328 hourly observations (wind, temp, precipitation, soil moisture)
- **ROI**: Manual tank farm masks traced from satellite imagery + control grassland

## Methodology

8-test statistical battery applied (all tests run, none skipped):
1. Raw Pearson/Spearman correlation (baseline)
2. First-difference correlation (kill test)
3. STL-detrended correlation
4. Within-year stability analysis
5. Out-of-time validation (train 2017-2023, test 2024-2025)
6. Granger causality (both directions)
7. Wind confound partial correlation
8. Benjamini-Hochberg FDR multiple comparison correction

Plus control ROI null test on agricultural land.

## Recommendations

1. **Do not pursue aggregate backscatter at 20m resolution** for oil storage estimation at Cushing
2. **Commercial SAR (1-3m) is necessary** for individual tank fill measurement
3. The code and methodology are reusable for other SAR-commodity studies — the statistical battery correctly identified the spurious correlation
4. The marginal control-difference signal (rho=-0.10) could be investigated further with Sentinel-1C (launched Dec 2024) providing 6-day revisit, but expectations should be low

## Lessons confirmed

- **Santos lesson validated**: Level correlations (rho = -0.53) were entirely spurious. First-differencing killed the signal.
- **Rotterdam lesson validated**: Wind correction was applied but was not the issue here — the resolution floor is the fundamental limitation.
- **Negative results have value**: This quantifies the resolution floor for SAR-based oil storage estimation, which was previously uncharacterized in published literature at 20m.
