"""
Random Forest classifier (FULL FEATURES).

Feature set:
  - Matryoshka prefix-slice embedding statistics
  - Lexical overlap / length features
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data import PairRecord
from features import build_matrix, matryoshka_all_features, DEFAULT_MATRYOSHKA_DIMS
from hyperparameter_tuning import RandomizedSearchCV


# =========================
# Default hyperparameters
# =========================
_DEFAULTS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

param_space = {'n_estimators': {'type': 'int', 'low': 20, 'high': 1000}, #number of trees in the forest
               'max_depth': {'type': 'int', 'low': 4, 'high': 10}, #maximum depth of trees 
               'min_samples_split': {'type': 'int', 'low': 2, 'high': 20}, #minimum number of samples required to split an internal node
               'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 20} #minimum number of samples required to be at a leaf node
               }

class RandomForestModel:
    """
    Random Forest model using full feature set.

    Interface:
    ----------
    build_features(records)  → (X, y, feature_names)
    fit(X_train, y_train)
    predict_proba(X_test)
    feature_importances()
    get_config()
    """

    name = "RandomForest (full features)"

    def __init__(
        self,
        matryoshka_dims: tuple[int, ...] | None = None,
        **kwargs,
    ):
        params = {**_DEFAULTS, **kwargs}
        self._model = RandomForestClassifier(**params)
        self._dims = matryoshka_dims
        self._params = params
        self._tuning_info: dict[str, object] = {
            "enabled": False,
        }
        self._feature_names: list[str] = []

    @property
    def matryoshka_dims(self) -> tuple[int, ...] | None:
        return self._dims

    # =========================
    # Feature construction
    # =========================
    def _feature_fn(self, r: PairRecord) -> dict[str, float]:
        return matryoshka_all_features(r, dims=self._dims)

    def build_features(
        self, records: list[PairRecord]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        X, feature_names = build_matrix(records, self._feature_fn)
        y = np.array([r.label for r in records], dtype=np.int32)

        self._feature_names = feature_names
        return X, y, feature_names

    # =========================
    # Model training
    # =========================
    def tune(self, X: np.ndarray, y: np.ndarray) -> None:
        tuner = RandomizedSearchCV(
            estimator=RandomForestClassifier(**self._params),
            param_distributions=param_space,
            n_iter=20,
            cv=3,
            scoring="f1",
            random_state=42,
            n_jobs=-1,
        )
        tuner.fit(X, y)
        best_params = tuner.get_best_params()
        best_score = tuner.get_best_score()
        print("Best hyperparameters:", best_params)
        self._params.update(best_params)
        self._model.set_params(**best_params)
        self._tuning_info = {
            "enabled": True,
            "method": "RandomizedSearchCV",
            "best_cv_score": float(best_score),
            "best_params": best_params,
        }
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_test)[:, 1].astype(np.float32)

    # =========================
    # Feature importance
    # =========================
    def feature_importances(self) -> dict[str, float]:
        importances = self._model.feature_importances_
        return dict(zip(self._feature_names, importances.tolist()))

    # =========================
    # Config for reporting
    # =========================
    def get_config(self) -> dict:
        dims_used = (
            list(self._dims)
            if self._dims is not None
            else list(DEFAULT_MATRYOSHKA_DIMS)
        )

        return {
            "model_class": "RandomForestModel",
            "matryoshka_dims": dims_used,
            "hyperparams": self._params,
            "tuning": self._tuning_info,
            "n_features": len(self._feature_names),
            "feature_names": self._feature_names,
        }