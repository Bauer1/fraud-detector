"""Supervised fraud scoring models: Logistic Regression baseline & calibrated GBDT (LightGBM / HistGBDT)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across uniform probability bins."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if len(y) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= bins[i]) & (p <= bins[i + 1])
        else:
            mask = (p >= bins[i]) & (p < bins[i + 1])
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / len(y)) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return round(float(ece), 5)


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


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
    """Wrapper around LightGBM (or scikit-learn HistGradientBoostingClassifier fallback) with validation calibration.

    Supports post-hoc probability calibration (`sigmoid` Platt scaling or `isotonic`) fitted strictly
    on validation data so `class_weight="balanced"` scores map cleanly to empirical risk probabilities
    while preserving direct `self.model` access for `shap.TreeExplainer`.
    """

    def __init__(self, params: dict[str, Any] | None = None, prefer_lightgbm: bool = True):
        self.params = dict(params or {})
        self.prefer_lightgbm = prefer_lightgbm and HAS_LIGHTGBM
        self.backend_name = "lightgbm" if self.prefer_lightgbm else "hist_gbdt"
        self.feature_names_: list[str] = []
        self.calibration_method_: str = "none"
        self._calibrator: Any = None
        self.calibration_diagnostics_: dict[str, Any] = {}
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

    def _prepare_array(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.feature_names_ and set(self.feature_names_).issubset(X.columns):
                return X[self.feature_names_].values
            return X.values
        return np.asarray(X)

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "FraudGBDTModel":
        self.feature_names_ = list(X.columns)
        self._calibrator = None
        self.calibration_method_ = "none"
        self.model.fit(X.values, np.asarray(y, dtype=int))
        return self

    def predict_raw_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return uncalibrated positive-class score from base GBDT estimator."""
        arr = self._prepare_array(X)
        return self.model.predict_proba(arr)[:, 1]

    def calibrate(
        self,
        X_val: pd.DataFrame | np.ndarray,
        y_val: np.ndarray | pd.Series,
        method: str = "sigmoid",
    ) -> dict[str, Any]:
        """Fit probability calibrator on chronological validation slice (never test set!)."""
        y_arr = np.asarray(y_val, dtype=int)
        raw_probs = self.predict_raw_score(X_val)

        brier_before = float(brier_score_loss(y_arr, raw_probs)) if len(y_arr) > 0 else 0.0
        ece_before = compute_expected_calibration_error(y_arr, raw_probs)

        method_norm = (method or "none").strip().lower()
        if method_norm == "none" or len(np.unique(y_arr)) < 2 or int(y_arr.sum()) < 5:
            self._calibrator = None
            self.calibration_method_ = "none"
            diag = {
                "method": "none",
                "calibration_rows": int(len(y_arr)),
                "positive_rows": int(y_arr.sum()),
                "brier_before": round(brier_before, 6),
                "brier_after": round(brier_before, 6),
                "ece_before": ece_before,
                "ece_after": ece_before,
            }
            self.calibration_diagnostics_ = diag
            return diag

        if method_norm == "isotonic" and int(y_arr.sum()) >= 30:
            iso = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
            iso.fit(raw_probs, y_arr)
            self._calibrator = iso
            self.calibration_method_ = "isotonic"
        else:
            # 2-parameter Platt scaling on log-odds: low-variance, strictly monotonic, preserves ranking (PR-AUC)
            platt = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            platt.fit(_logit(raw_probs).reshape(-1, 1), y_arr)
            self._calibrator = platt
            self.calibration_method_ = "sigmoid"

        cal_probs = self.predict_score(X_val)
        brier_after = float(brier_score_loss(y_arr, cal_probs))
        ece_after = compute_expected_calibration_error(y_arr, cal_probs)

        diag = {
            "method": self.calibration_method_,
            "calibration_rows": int(len(y_arr)),
            "positive_rows": int(y_arr.sum()),
            "brier_before": round(brier_before, 6),
            "brier_after": round(brier_after, 6),
            "ece_before": ece_before,
            "ece_after": ece_after,
            "raw_mean_prob": round(float(raw_probs.mean()), 5),
            "calibrated_mean_prob": round(float(cal_probs.mean()), 5),
            "empirical_fraud_rate": round(float(y_arr.mean()), 5),
        }
        self.calibration_diagnostics_ = diag
        return diag

    def predict_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw = self.predict_raw_score(X)
        calibrator = getattr(self, "_calibrator", None)
        method = getattr(self, "calibration_method_", "none")
        if calibrator is None or method == "none":
            return raw
        if method == "isotonic":
            return np.asarray(calibrator.transform(raw), dtype=float)
        return np.asarray(calibrator.predict_proba(_logit(raw).reshape(-1, 1))[:, 1], dtype=float)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        p1 = np.clip(self.predict_score(X), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])
