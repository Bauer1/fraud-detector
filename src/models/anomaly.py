"""Unsupervised anomaly detection (Isolation Forest) as an auxiliary drift/novelty signal."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


class FraudAnomalyDetector:
    """Isolation Forest trained on normal (legitimate) transactions.

    Maps raw Isolation Forest decision scores into a [0, 1] auxiliary anomaly score using
    training percentiles. Per Technical Specification Section 5, anomaly score is strictly
    an auxiliary signal/monitor and never defines fraud on its own.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        p = dict(params or {})
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            n_estimators=p.get("n_estimators", 150),
            contamination=p.get("contamination", 0.02),
            max_samples=p.get("max_samples", 512),
            random_state=p.get("random_state", 42),
            n_jobs=1,
        )
        self.feature_names_: list[str] = []
        self._raw_min: float = -0.3
        self._raw_max: float = 0.3

    def fit(self, X_normal: pd.DataFrame) -> "FraudAnomalyDetector":
        """Fit on legitimate training transactions only."""
        self.feature_names_ = list(X_normal.columns)
        X_scaled = self.scaler.fit_transform(X_normal.values)
        self.model.fit(X_scaled)
        raw_scores = -self.model.decision_function(X_scaled)
        self._raw_min = float(np.percentile(raw_scores, 5))
        self._raw_max = float(np.percentile(raw_scores, 99.5))
        if self._raw_max <= self._raw_min:
            self._raw_max = self._raw_min + 1.0
        return self

    def predict_anomaly_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return normalized auxiliary anomaly score in [0, 1] (higher = more anomalous)."""
        if isinstance(X, pd.DataFrame):
            if self.feature_names_ and set(self.feature_names_).issubset(X.columns):
                arr = X[self.feature_names_].values
            else:
                arr = X.values
        else:
            arr = np.asarray(X)
        X_scaled = self.scaler.transform(arr)
        raw = -self.model.decision_function(X_scaled)
        norm = (raw - self._raw_min) / (self._raw_max - self._raw_min)
        return np.clip(norm, 0.0, 1.0)
