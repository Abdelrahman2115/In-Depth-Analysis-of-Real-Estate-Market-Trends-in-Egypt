import numpy as np
import pandas as pd

from egypt_real_estate import modeling
from egypt_real_estate.recommender import BuyerProfile, PropertyRecommender


def _listings(n=400, seed=0):
    rng = np.random.default_rng(seed)
    district = rng.choice(["New Cairo", "Sheikh Zayed", "Nasr City"], n)
    source = rng.choice(["Bayut", "OLX"], n)
    ptype = rng.choice(["Apartment", "Villa"], n)
    area = rng.uniform(80, 400, n).round()
    level = pd.Series(district).map({"New Cairo": 60_000, "Sheikh Zayed": 55_000, "Nasr City": 25_000}).to_numpy()
    level = level * np.where(source == "Bayut", 3.0, 1.0) * np.where(ptype == "Villa", 1.3, 1.0)
    price = (area * level * rng.lognormal(0, 0.1, n)).round()
    return pd.DataFrame({
        "source": source, "type": ptype, "price": price, "area_sqm": area,
        "bedrooms": np.clip((area / 60).round(), 1, 6).astype(int),
        "bathrooms": np.clip((area / 80).round(), 1, 5).astype(int),
        "district": district, "region": "Greater Cairo",
        "compound": rng.choice(["A", "B", None], n), "price_per_sqm": price / area,
    })


def test_baseline_falls_back_for_unseen_combinations():
    df = _listings()
    X, y = modeling.prepare_features(df)
    base = modeling.SegmentMedianBaseline().fit(X, y)
    unseen = X.head(3).assign(district="Unseen District")
    pred = base.predict(unseen)
    assert np.isfinite(pred).all() and (pred > 0).all()


def test_models_learn_signal_and_beat_global_median():
    df = _listings()
    X_tr, X_te, y_tr, y_te = modeling.split(df)
    models = {k: v for k, v in modeling.build_models().items() if "Random forest" not in k}
    table, _ = modeling.compare_models(X_tr, y_tr, X_te, y_te, models=models, cv_folds=3)
    assert table.loc["Gradient boosting (HistGB)", "Test Median APE (%)"] < \
        table.loc["Global median (baseline)", "Test Median APE (%)"] / 2


def test_recommender_respects_hard_rules():
    df = _listings()
    rec = PropertyRecommender(df, fair_price=df["price"] * 1.1)
    profile = BuyerProfile(max_budget=20e6, property_type="Apartment", district="New Cairo", min_bedrooms=2)
    out = rec.recommend(profile, top_n=5)
    assert len(out) > 0
    assert (out["price"] <= 20e6).all()
    assert set(out["type"]) == {"Apartment"} and set(out["district"]) == {"New Cairo"}
    assert (out["bedrooms"] >= 2).all()
    assert out["value_score"].is_monotonic_decreasing
