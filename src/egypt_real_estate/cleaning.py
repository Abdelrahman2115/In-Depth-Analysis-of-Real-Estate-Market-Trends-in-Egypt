"""Load, clean and merge the three listing sources into one table.

Every step is logged so the notebooks can show exactly how many rows each
rule removed (see :class:`CleaningLog`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .locations import compound_from_hierarchy, normalize_compound, standardize_locations

COLUMNS = [
    "source", "type", "price", "area_sqm", "bedrooms", "bathrooms",
    "location_raw", "district", "region", "compound", "price_per_sqm",
]

TYPE_MAP = {
    "apartment": "Apartment",
    "apartement": "Apartment",
    "flat": "Apartment",
    "hotel apartment": "Apartment",
    "studio": "Apartment",
    "duplex": "Duplex",
    "penthouse": "Penthouse",
    "chalet": "Chalet",
    "cabin": "Chalet",
    "villa": "Villa",
    "ivilla": "Villa",
    "standalone villa": "Villa",
    "stand alone villa": "Villa",
    "town house": "Townhouse",
    "townhouse": "Townhouse",
    "twin house": "Twin House",
    "twinhouse": "Twin House",
}
# Anything not in TYPE_MAP (rooms, land, roofs, whole compounds, "Unknown", ...)
# is not a comparable residential unit and is dropped.


@dataclass
class CleaningLog:
    """Row counts before/after each cleaning rule."""

    steps: list[dict] = field(default_factory=list)

    def record(self, source: str, step: str, before: int, after: int) -> None:
        self.steps.append(
            {"source": source, "step": step, "rows_before": before,
             "rows_after": after, "removed": before - after}
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def _to_number(series: pd.Series) -> pd.Series:
    """Parse strings like "7,800,000", "155 Sq. M." or "10+" into floats."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _map_type(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True).map(TYPE_MAP)


def _read(name: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / config.RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the datasets listed in README.md into {raw_dir}."
        )
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Source-specific loaders: each returns the common schema.
# --------------------------------------------------------------------------- #

def load_olx(raw_dir: Path, log: CleaningLog) -> pd.DataFrame:
    """OLX Egypt (Kaggle: mohammedaltet/egypt-houses-price)."""
    raw = _read("olx", raw_dir)
    n = len(raw)
    df = pd.DataFrame({
        "source": "OLX",
        "type": _map_type(raw["Type"]),
        "price": _to_number(raw["Price"]),
        "area_sqm": _to_number(raw["Area"]),
        # "10+" and "Unknown" are not exact counts, so they become NaN and are dropped.
        "bedrooms": pd.to_numeric(raw["Bedrooms"], errors="coerce"),
        "bathrooms": pd.to_numeric(raw["Bathrooms"], errors="coerce"),
        "location_raw": raw["City"].astype(str).str.strip(),
    })
    # Rows where the scraper captured "(View phone number)" instead of a city.
    bad_city = df["location_raw"].str.contains("phone number", case=False, na=False)
    df = df[~bad_city]
    log.record("OLX", "drop rows with a phone-number placeholder as location", n, len(df))
    compound = raw.loc[df.index, "Compound"].map(normalize_compound)
    return standardize_locations(df, "location_raw", compound)


def load_bayut(raw_dir: Path, log: CleaningLog) -> pd.DataFrame:
    """Bayut Egypt (Kaggle: ameribrahim/real-estate-bayot-egypt)."""
    raw = _read("bayut", raw_dir)
    n = len(raw)
    raw = raw.drop_duplicates(subset="link")
    log.record("Bayut", "drop repeated listing URLs", n, len(raw))

    n = len(raw)
    currency = raw["currency"].astype(str).str.strip().str.upper()
    raw = raw[currency == "EGP"]
    log.record("Bayut", "keep EGP prices only (drop USD/other currencies)", n, len(raw))

    df = pd.DataFrame({
        "source": "Bayut",
        "type": _map_type(raw["type"]),
        "price": _to_number(raw["price"]),
        "area_sqm": _to_number(raw["area"]),
        "bedrooms": pd.to_numeric(raw["beds"], errors="coerce"),
        "bathrooms": pd.to_numeric(raw["baths"], errors="coerce"),
        "location_raw": raw["location"].astype(str).str.strip(),
    })
    # "Compound, Area, Governorate" -> the first part is a compound when there are 3+ parts.
    compound = raw["location"].map(lambda s: compound_from_hierarchy(s, min_parts=3))
    return standardize_locations(df, "location_raw", compound)


def load_cairo(raw_dir: Path, log: CleaningLog) -> pd.DataFrame:
    """Cairo listings (Kaggle: iyadelwy/egypt-housing-prices)."""
    raw = _read("cairo", raw_dir)
    n = len(raw)
    raw = raw[raw["price"].astype(str).str.strip().str.lower() != "ask"]
    log.record("Cairo", "drop 'Ask' (no price) listings", n, len(raw))

    bedrooms = raw["bedroom"].astype(str).str.strip().str.lower().replace({"studio": "1"})
    df = pd.DataFrame({
        "source": "Cairo",
        "type": _map_type(raw["type"]),
        "price": _to_number(raw["price"]),
        "area_sqm": _to_number(raw["size_sqm"]),
        "bedrooms": pd.to_numeric(bedrooms, errors="coerce"),
        "bathrooms": pd.to_numeric(raw["bathroom"], errors="coerce"),
        "location_raw": raw["location"].astype(str).str.strip(),
    })
    # "Compound, Sub-area, District, City, Governorate" -> first part is a compound at 4+ parts.
    compound = raw["location"].map(lambda s: compound_from_hierarchy(s, min_parts=4))
    return standardize_locations(df, "location_raw", compound)


# --------------------------------------------------------------------------- #
# Shared rules applied to the combined table.
# --------------------------------------------------------------------------- #

def apply_quality_rules(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Drop incomplete, non-residential and physically implausible rows."""
    steps = [
        ("drop non-residential / unknown property types", df["type"].notna()),
        ("drop rows missing price, area, bedrooms or bathrooms",
         df[["price", "area_sqm", "bedrooms", "bathrooms"]].notna().all(axis=1)),
    ]
    for name, mask in steps:
        before = len(df)
        df = df[mask.reindex(df.index, fill_value=False)]
        log.record("All", name, before, len(df))

    df = df.copy()
    df["price_per_sqm"] = df["price"] / df["area_sqm"]

    rules = [
        (f"area between {config.MIN_AREA_SQM} and {config.MAX_AREA_SQM} m²",
         df["area_sqm"].between(config.MIN_AREA_SQM, config.MAX_AREA_SQM)),
        (f"1-{config.MAX_ROOMS} bedrooms and bathrooms",
         df["bedrooms"].between(1, config.MAX_ROOMS) & df["bathrooms"].between(1, config.MAX_ROOMS)),
        (f"price per m² between {config.MIN_PRICE_PER_SQM:,} and {config.MAX_PRICE_PER_SQM:,} EGP "
         "(removes rentals, typos and unconverted foreign-currency prices)",
         df["price_per_sqm"].between(config.MIN_PRICE_PER_SQM, config.MAX_PRICE_PER_SQM)),
    ]
    for name, mask in rules:
        before = len(df)
        df = df[mask.reindex(df.index, fill_value=False)]
        log.record("All", name, before, len(df))
    return df


def remove_type_outliers(df: pd.DataFrame, log: CleaningLog, k: float = 2.5) -> pd.DataFrame:
    """Drop extreme log(price per m²) values within each (source, property type) group.

    Grouping by source matters because the sources were scraped at different
    times and their price levels differ (see notebook 02). A fence on price per m²
    removes data-entry errors, such as a down payment entered as the price,
    without cutting off genuinely expensive areas, unlike a 1.5 × IQR cut on raw price.
    """
    before = len(df)
    log_ppsqm = np.log(df["price_per_sqm"])
    grouped = log_ppsqm.groupby([df["source"], df["type"]])
    q1, q3 = grouped.transform("quantile", 0.25), grouped.transform("quantile", 0.75)
    iqr = q3 - q1
    keep = log_ppsqm.between(q1 - k * iqr, q3 + k * iqr)
    df = df[keep]
    log.record("All", f"outliers in log(price per m²) per source and type ({k:g}×IQR fence)", before, len(df))
    return df


def deduplicate(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Remove the same listing posted twice, on one site or across sites.

    Two rows are treated as the same listing when they share district, type,
    bedrooms, bathrooms, area (to the m²) and price (to the nearest EGP 10,000).
    Rows with a known compound are preferred as the copy to keep.
    """
    before = len(df)
    df = df.assign(
        _area_key=df["area_sqm"].round(0),
        _price_key=(df["price"] / 10_000).round(0),
        _has_compound=df["compound"].notna(),
    )
    key = ["district", "type", "bedrooms", "bathrooms", "_area_key", "_price_key"]
    dup_any = df.duplicated(subset=key, keep=False)
    multi_source = df[dup_any].groupby(key)["source"].transform("nunique") > 1
    cross_site = int(multi_source.sum())

    df = (
        df.sort_values("_has_compound", ascending=False)
        .drop_duplicates(subset=key, keep="first")
        .drop(columns=["_area_key", "_price_key", "_has_compound"])
        .sort_index()
    )
    log.record("All", f"remove duplicate listings ({cross_site:,} rows were in a group spanning 2+ sites)",
               before, len(df))
    return df


def build_dataset(raw_dir: Path | None = None) -> tuple[pd.DataFrame, CleaningLog]:
    """Run the full cleaning pipeline and return ``(listings, log)``."""
    raw_dir = Path(raw_dir) if raw_dir else config.RAW_DIR
    log = CleaningLog()
    frames = [load_olx(raw_dir, log), load_bayut(raw_dir, log), load_cairo(raw_dir, log)]
    combined = pd.concat(frames, ignore_index=True)
    log.record("All", "combine the three sources", sum(len(f) for f in frames), len(combined))

    df = apply_quality_rules(combined, log)
    df = remove_type_outliers(df, log)
    df = deduplicate(df, log)

    df = df[COLUMNS].reset_index(drop=True)
    df[["bedrooms", "bathrooms"]] = df[["bedrooms", "bathrooms"]].astype(int)
    df["price"] = df["price"].round(0).astype("int64")
    df["price_per_sqm"] = df["price_per_sqm"].round(0)
    return df, log


def load_clean(path: Path | None = None) -> pd.DataFrame:
    """Read the processed dataset written by :func:`build_dataset`."""
    path = Path(path) if path else config.PROCESSED_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run notebooks/01_data_cleaning.ipynb first.")
    return pd.read_csv(path)
