"""Supervised fraud scoring models: Logistic Regression baseline & GBDT (LightGBM / HistGBDT)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def build_baseline_model(random_state: int = 42) -> Pipeline:
    """Baseline model: StandardScaler + class-balanced Logistic Regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=0.5,
                random_state=random_state,
            ),
        ),
    ])


class FraudGBDTModel:
    """Wrapper around LightGBM (or scikit-learn HistGradientBoostingClassifier fallback).

    Handles severe class imbalance via balanced class weights and exposes a uniform scikit-learn API.
    """

    def __init__(self, params: dict[str, Any] | None = None, prefer_lightgbm: bool = True):
        self.params = dict(params or {})
        self.prefer_lightgbm = prefer_lightgbm and HAS_LIGHTGBM
        self.backend_name = "lightgbm" if self.prefer_lightgbm else "hist_gbdt"
        self.feature_names_: list[str] = []
        self.model = self._init_estimator()

    def _init_estimator(self) -> Any:
        if self.prefer_lightgbm:
            lgb_params = {
                "n_estimators": self.params.get("n_estimators", 250),
                "learning_rate": self.params.get("learning_rate", 0.05),
                "max_depth": self.params.get("max_depth", 6),
                "num_leaves": self.params.get("num_leaves", 31),
                "min_child_samples": self.params.get("min_child_samples", 30),
                "subsample": self.params.get("subsample", 0.8),
                "colsample_bytree": self.params.get("colsample_bytree", 0.8),
                "class_weight": self.params.get("class_weight", "balanced"),
                "random_state": self.params.get("random_state", 42),
                "verbose": -1,
                "n_jobs": -1,
            }
            return lgb.LGBMClassifier(**lgb_params)

        return HistGradientBoostingClassifier(
            max_iter=self.params.get("n_estimators", 250),
            learning_rate=self.params.get("learning_rate", 0.05),
            max_depth=self.params.get("max_depth", 6),
            max_leaf_nodes=self.params.get("num_leaves", 31),
            min_samples_leaf=self.params.get("min_child_samples", 30),
            l2_regularization=1.0,
            class_weight=self.params.get("class_weight", "balanced"),
            random_state=self.params.get("random_state", 42),
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "FraudGBDTModel":
        self.feature_names_ = list(X.columns)
        self.model.fit(X.values, np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict_proba(arr)

    def predict_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.predict_proba(X)[:, 1]
