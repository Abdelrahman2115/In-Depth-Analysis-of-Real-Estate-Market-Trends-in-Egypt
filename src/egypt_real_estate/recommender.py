"""Rule-based property recommender.

1. **Hard rules** filter listings to the buyer's constraints (budget, type,
   location, bedrooms, size).
2. **Ranking** puts the best-value listings first. Value is the gap between the
   listing's asking price and a *fair price* estimated by the price model using
   out-of-fold predictions, so a listing is never judged by a model that was
   trained on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BuyerProfile:
    max_budget: float
    min_budget: float = 0
    property_type: str | list[str] | None = None
    district: str | list[str] | None = None
    region: str | None = None
    min_bedrooms: int | None = None
    min_area_sqm: float | None = None
    source: str | list[str] | None = None


def _as_list(value):
    if value is None:
        return None
    return [value] if isinstance(value, str) else list(value)


class PropertyRecommender:
    """Filter listings by hard rules, then rank them by value.

    Two independent signals measure value:

    * ``value_gap_pct``: how far the asking price is below the model's fair price
    * ``ppsqm_discount_pct``: how far the price per m² is below the median for the
      same source, district and property type

    The ranking score is their average, so a listing ranks high only when the
    model *and* the simple market comparison agree that it is cheap. Listings
    where either signal is extreme are flagged as likely data errors and excluded.
    """

    def __init__(self, listings: pd.DataFrame, fair_price: np.ndarray,
                 max_value_gap_pct: float = 35.0, max_ppsqm_discount_pct: float = 50.0):
        df = listings.copy()
        df["fair_price"] = np.asarray(fair_price, float)
        df["value_gap_pct"] = 100 * (df["fair_price"] - df["price"]) / df["fair_price"]
        segment_median = df.groupby(["source", "district", "type"])["price_per_sqm"].transform("median")
        df["ppsqm_discount_pct"] = 100 * (1 - df["price_per_sqm"] / segment_median)
        df["suspicious"] = (df["value_gap_pct"] > max_value_gap_pct) | (
            df["ppsqm_discount_pct"] > max_ppsqm_discount_pct)
        df["value_score"] = (df["value_gap_pct"] + df["ppsqm_discount_pct"]) / 2
        self.listings = df

    def filter(self, profile: BuyerProfile) -> pd.DataFrame:
        df = self.listings
        mask = df["price"].between(profile.min_budget, profile.max_budget)
        if (types := _as_list(profile.property_type)) is not None:
            mask &= df["type"].isin(types)
        if (districts := _as_list(profile.district)) is not None:
            mask &= df["district"].isin(districts)
        if profile.region is not None:
            mask &= df["region"].eq(profile.region)
        if profile.min_bedrooms is not None:
            mask &= df["bedrooms"].ge(profile.min_bedrooms)
        if (sources := _as_list(profile.source)) is not None:
            mask &= df["source"].isin(sources)
        if profile.min_area_sqm is not None:
            mask &= df["area_sqm"].ge(profile.min_area_sqm)
        return df[mask]

    def recommend(self, profile: BuyerProfile, top_n: int = 10) -> pd.DataFrame:
        """Return the ``top_n`` best-value, non-suspicious listings matching ``profile``."""
        candidates = self.filter(profile)
        candidates = candidates[~candidates["suspicious"]]
        ranked = candidates.sort_values("value_score", ascending=False)
        result = ranked.head(top_n).copy()
        result["why"] = result.apply(self._explain, axis=1)
        cols = ["type", "district", "compound", "bedrooms", "bathrooms", "area_sqm",
                "price", "fair_price", "value_score", "source", "why"]
        return result[cols].round({"fair_price": -3, "value_score": 1})

    @staticmethod
    def _explain(row: pd.Series) -> str:
        gap, disc = row["value_gap_pct"], row["ppsqm_discount_pct"]
        return (f"{abs(gap):.0f}% {'below' if gap >= 0 else 'above'} estimated fair price; "
                f"price/m² {abs(disc):.0f}% {'below' if disc >= 0 else 'above'} the "
                f"{row['district']} {row['type'].lower()} median")
