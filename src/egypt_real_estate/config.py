"""Project paths and global settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILES = {
    "olx": "Egypt_Houses_Price.csv",
    "bayut": "real_estate_data_bayut_full.csv",
    "cairo": "properties.csv",
}

PROCESSED_FILE = PROCESSED_DIR / "listings_clean.csv"

RANDOM_STATE = 42

# Plausibility limits used during cleaning (sale listings, EGP).
MIN_AREA_SQM = 25
MAX_AREA_SQM = 2_000
MIN_PRICE_PER_SQM = 3_000
MAX_PRICE_PER_SQM = 400_000
MAX_ROOMS = 10
