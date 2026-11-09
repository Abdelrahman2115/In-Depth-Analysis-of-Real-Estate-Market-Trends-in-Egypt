"""Market analysis helpers: district segmentation and association rules."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config


def district_summary(df: pd.DataFrame, min_listings: int = 30) -> pd.DataFrame:
    """Median price, size and price per m² for every district with enough listings."""
    summary = (
        df.groupby(["district", "region"])
        .agg(listings=("price", "size"),
             median_price=("price", "median"),
             median_area=("area_sqm", "median"),
             median_price_per_sqm=("price_per_sqm", "median"),
             share_villas=("type", lambda s: s.isin(["Villa", "Townhouse", "Twin House"]).mean()))
        .reset_index()
    )
    return summary[summary["listings"] >= min_listings].sort_values("median_price_per_sqm", ascending=False)


def segment_districts(summary: pd.DataFrame, k_range=range(2, 7)) -> tuple[pd.DataFrame, pd.Series]:
    """Cluster districts on log price/m², log size and villa share.

    ``k`` is chosen by silhouette score.
    """
    features = np.column_stack([
        np.log(summary["median_price_per_sqm"]),
        np.log(summary["median_area"]),
        summary["share_villas"],
    ])
    X = StandardScaler().fit_transform(features)
    scores = {}
    for k in k_range:
        if k >= len(summary):
            break
        labels = KMeans(n_clusters=k, n_init=20, random_state=config.RANDOM_STATE).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    best_k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_k, n_init=20, random_state=config.RANDOM_STATE).fit_predict(X)

    out = summary.copy()
    out["cluster"] = labels
    # Number segments from highest to lowest median price per m². The clusters
    # differ in character (size, villa share) as much as in price, so they are
    # numbered rather than given price labels such as "luxury" or "budget".
    order = out.groupby("cluster")["median_price_per_sqm"].median().sort_values(ascending=False).index
    out["segment"] = out["cluster"].map({c: f"Segment {i + 1}" for i, c in enumerate(order)})
    return out.drop(columns="cluster"), pd.Series(scores, name="silhouette")


def price_bands(price_per_sqm: pd.Series) -> pd.Series:
    """Split price per m² into Low / Medium / High terciles."""
    return pd.qcut(price_per_sqm, q=3, labels=["Low", "Medium", "High"]).astype(str)


def association_rules(
    df: pd.DataFrame,
    antecedent_cols=("district", "type", "bedrooms"),
    consequent_col: str = "price_band",
    min_support: float = 0.005,
    min_confidence: float = 0.6,
    max_antecedent_len: int = 2,
) -> pd.DataFrame:
    """Rules of the form ``{district=X, type=Y} -> price_band=High``.

    Because every listing has exactly one value per column, itemset support can
    be counted with ``groupby``; this is equivalent to Apriori on one-hot data
    restricted to rules with a price-band consequent.
    """
    n = len(df)
    data = df[list(antecedent_cols) + [consequent_col]].astype(str)
    consequent_support = data[consequent_col].value_counts(normalize=True)
    rules = []
    for length in range(1, max_antecedent_len + 1):
        for cols in combinations(antecedent_cols, length):
            cols = list(cols)
            ante = data.groupby(cols).size().rename("ante_count")
            joint = data.groupby(cols + [consequent_col]).size().rename("count").reset_index()
            joint = joint.join(ante, on=cols)
            joint["support"] = joint["count"] / n
            joint["confidence"] = joint["count"] / joint["ante_count"]
            joint["lift"] = joint["confidence"] / joint[consequent_col].map(consequent_support)
            joint["antecedent"] = joint[cols].apply(
                lambda r: ", ".join(f"{c}={v}" for c, v in zip(cols, r)), axis=1)
            joint["consequent"] = consequent_col + "=" + joint[consequent_col]
            rules.append(joint[["antecedent", "consequent", "count", "support", "confidence", "lift"]])
    out = pd.concat(rules, ignore_index=True)
    out = out[(out["support"] >= min_support) & (out["confidence"] >= min_confidence)]
    return out.sort_values(["lift", "support"], ascending=False).reset_index(drop=True)
