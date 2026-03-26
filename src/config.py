"""Project constants for Cushing SAR oil storage estimation."""

# Cushing AOI
CUSHING_CENTER = (35.985, -96.77)  # lat, lon
CUSHING_BBOX_SEARCH = (-97.05, 35.90, -96.65, 36.10)  # broad bbox for ASF scene search
CUSHING_BBOX = (-96.800, 35.892, -96.681, 36.053)  # tight crop bbox around tank farms
CROP_BUFFER = 0.03  # degrees buffer around manual mask bounds

# EIA API
EIA_CUSHING_STOCKS = "W_EPC0_SAX_YCUOK_MBBL"  # weekly Cushing crude stocks (thousand bbl)
EIA_WTI_SPOT = "PET.RWTC.D"  # daily WTI spot price
WTI_FUTURES_TICKER = "CL=F"  # Yahoo Finance front-month WTI

# Time range
DATA_START = "2017-01-01"
DATA_END = "2026-03-01"

# Train/test split
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"

# Analysis thresholds
FDR_THRESHOLD = 0.05  # Benjamini-Hochberg q-value threshold
KILL_THRESHOLD = 0.1  # minimum first-difference |rho| to proceed
MIN_OBSERVATIONS = 26  # minimum weeks for correlation analysis

# SAR preprocessing
DB_FLOOR = -30.0  # backscatter floor in dB
SPECKLE_FILTER_SIZE = 3  # Lee filter window size
BRIGHT_SCATTERER_THRESHOLD = -5.0  # dB threshold for bright pixel ratio

# Paths
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CROPPED_DIR = os.path.join(DATA_DIR, "cropped")
EIA_DIR = os.path.join(DATA_DIR, "eia")
ERA5_DIR = os.path.join(DATA_DIR, "era5")
ROI_DIR = os.path.join(DATA_DIR, "roi")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
