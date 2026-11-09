"""Rebuild the processed dataset from the raw CSVs:  python -m egypt_real_estate"""

from . import config
from .cleaning import build_dataset


def main() -> None:
    listings, log = build_dataset()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    listings.to_csv(config.PROCESSED_FILE, index=False)
    log.to_frame().to_csv(config.REPORTS_DIR / "cleaning_log.csv", index=False)
    print(log.to_frame()[["source", "step", "removed", "rows_after"]].to_string(index=False))
    print(f"\nSaved {len(listings):,} listings to {config.PROCESSED_FILE}")


if __name__ == "__main__":
    main()
