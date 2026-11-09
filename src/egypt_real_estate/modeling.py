"""Price-prediction models.

All preprocessing (encoding, scaling, target encoding) lives *inside*
scikit-learn pipelines, so it is fit on training folds only and nothing from
the test set leaks into training.

The target is ``log(price)``; metrics are reported back in EGP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

from . import config

CATEGORICAL = ["type", "district", "source"]
HIGH_CARDINALITY = ["compound"]
NUMERIC = ["area_sqm", "bedrooms", "bathrooms"]
FEATURES = CATEGORICAL + HIGH_CARDINALITY + NUMERIC
TARGET = "price"


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES].copy()
    X["compound"] = X["compound"].fillna("Unknown")
    return X, df[TARGET].astype(float)


INTERACTIONS = {
    # The sources price and rank districts differently (notebook 02), so the model
    # gets each district and compound *per source*, plus district × type.
    "source_district": ("source", "district"),
    "source_compound": ("source", "compound"),
    "district_type": ("district", "type"),
}


def add_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """Add combined categorical columns. Stateless, so it is safe inside a pipeline."""
    X = X.copy()
    for name, (a, b) in INTERACTIONS.items():
        X[name] = X[a].astype(str) + " | " + X[b].astype(str)
    return X


def _target_encoder() -> TargetEncoder:
    # Internal cross-fitting: each training row is encoded by folds that exclude it.
    return TargetEncoder(cv=KFold(5, shuffle=True, random_state=config.RANDOM_STATE))


class SegmentMedianBaseline(BaseEstimator, RegressorMixin):
    """Naive appraiser: area × median price per m² for the same source, district and type.

    Falls back to (source, type), then to the global median, for combinations not
    seen in training. Any useful model has to beat this.
    """

    levels = (["source", "district", "type"], ["source", "type"])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        ppsqm = np.asarray(y) / X["area_sqm"].to_numpy()
        frame = X.assign(ppsqm=ppsqm)
        self.medians_ = [frame.groupby(cols)["ppsqm"].median() for cols in self.levels]
        self.global_ = float(np.median(ppsqm))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ppsqm = np.full(len(X), np.nan)
        for cols, medians in zip(self.levels, self.medians_):
            found = medians.reindex(pd.MultiIndex.from_frame(X[cols])).to_numpy()
            ppsqm = np.where(np.isnan(ppsqm), found, ppsqm)
        ppsqm = np.where(np.isnan(ppsqm), self.global_, ppsqm)
        return ppsqm * X["area_sqm"].to_numpy()


def _log_target(regressor) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


_interactions = ("interactions", FunctionTransformer(add_interactions))
TARGET_ENCODED = HIGH_CARDINALITY + list(INTERACTIONS)


def _linear_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CATEGORICAL),
        ("target", _target_encoder(), TARGET_ENCODED),
        ("numeric", Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
    ])


def _tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False),
         ["type", "source"]),
        ("target", _target_encoder(), ["district"] + TARGET_ENCODED),
        ("numeric", "passthrough", NUMERIC),
    ], verbose_feature_names_out=False)


def build_models() -> dict[str, object]:
    """Candidate models, from trivial baselines to gradient boosting."""
    return {
        "Global median (baseline)": DummyRegressor(strategy="median"),
        "Source × district × type median price/m² (baseline)": SegmentMedianBaseline(),
        "Ridge regression": _log_target(Pipeline([
            _interactions,
            ("prep", _linear_preprocessor()),
            ("model", Ridge(alpha=1.0)),
        ])),
        "Random forest": _log_target(Pipeline([
            _interactions,
            ("prep", _tree_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, max_features=0.5,
                n_jobs=-1, random_state=config.RANDOM_STATE)),
        ])),
        "Gradient boosting (HistGB)": _log_target(Pipeline([
            _interactions,
            ("prep", _tree_preprocessor()),
            ("model", HistGradientBoostingRegressor(
                max_iter=1500, learning_rate=0.04, max_leaf_nodes=63, min_samples_leaf=20,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                random_state=config.RANDOM_STATE)),
        ])),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "MAE (EGP)": mean_absolute_error(y_true, y_pred),
        "RMSE (EGP)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE (%)": 100 * ape.mean(),
        "Median APE (%)": 100 * np.median(ape),
        "Within 20% (%)": 100 * (ape <= 0.20).mean(),
        "R² (EGP)": r2_score(y_true, y_pred),
        "R² (log price)": r2_score(np.log(y_true), np.log(np.clip(y_pred, 1, None))),
    }


def split(df: pd.DataFrame, test_size: float = 0.2):
    X, y = prepare_features(df)
    strata = df["type"]  # keep the property-type mix the same in train and test
    return train_test_split(X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=strata)


def compare_models(X_train, y_train, X_test, y_test, models=None, cv_folds: int = 5) -> tuple[pd.DataFrame, dict]:
    """Cross-validate every model on the training set, then score it once on the test set.

    Returns a results table and the fitted models.
    """
    models = models or build_models()
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    rows, fitted = [], {}
    for name, model in models.items():
        oof = cross_val_predict(model, X_train, y_train, cv=kfold, n_jobs=1)
        cv = regression_metrics(y_train, oof)
        model.fit(X_train, y_train)
        test = regression_metrics(y_test, model.predict(X_test))
        rows.append({
            "Model": name,
            "CV Median APE (%)": cv["Median APE (%)"],
            "CV R² (log price)": cv["R² (log price)"],
            **{f"Test {k}": v for k, v in test.items()},
        })
        fitted[name] = model
    table = pd.DataFrame(rows).set_index("Model")
    return table, fitted


def out_of_fold_predictions(model, df: pd.DataFrame, cv_folds: int = 5) -> np.ndarray:
    """Predict every listing with a model that never saw it (for the recommender)."""
    X, y = prepare_features(df)
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    return cross_val_predict(model, X, y, cv=kfold)


def feature_importance(model, X_test, y_test, n_repeats: int = 5) -> pd.Series:
    """Permutation importance on the test set (drop in R² of log price)."""
    from sklearn.inspection import permutation_importance

    def scorer(est, X, y):
        return r2_score(np.log(y), np.log(np.clip(est.predict(X), 1, None)))

    result = permutation_importance(model, X_test, y_test, scoring=scorer, n_repeats=n_repeats,
                                    random_state=config.RANDOM_STATE, n_jobs=1)
    return pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
