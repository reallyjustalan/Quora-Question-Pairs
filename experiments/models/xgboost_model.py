"""
models/xgboost_model.py — XGBoost classifier with matryoshka-sliced features.

Feature set:
  - Matryoshka prefix-slice embedding statistics
  - Lexical overlap / length features
"""

from __future__ import annotations

import os
import sys

import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data import PairRecord
from features import build_matrix, matryoshka_all_features, DEFAULT_MATRYOSHKA_DIMS
from hyperparameter_tuning import RandomizedSearchCV


_DEFAULTS = dict(
    n_estimators=700,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

param_space = {'max_depth': {'type': 'int', 'low': 3, 'high': 10},
               'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
               'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0}, #prevents overfitting by sampling a fraction of training data
               'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0}, #prevents overfitting by sampling a fraction of features
               'reg_lambda': {'type': 'float', 'low': 1e-3, 'high': 100.0, 'log': True} #regularisation strength (L2)
               }


class XGBoostModel:
    """Plug-and-play wrapper around xgboost.XGBClassifier."""

    name = "XGBoost"

    def __init__(
        self,
        matryoshka_dims: tuple[int, ...] | None = None,
        **kwargs,
    ):
        params = {**_DEFAULTS, **kwargs}
        self._model = XGBClassifier(**params)
        self._dims = matryoshka_dims
        self._params = params
        self._feature_names: list[str] = []

    @property
    def matryoshka_dims(self) -> tuple[int, ...] | None:
        return self._dims

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------

    def _feature_fn(self, r: PairRecord) -> dict[str, float]:
        return matryoshka_all_features(r, dims=self._dims)

    def build_features(
        self,
        records: list[PairRecord],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        X, feature_names = build_matrix(records, self._feature_fn)
        y = np.array([r.label for r in records], dtype=np.int32)
        self._feature_names = feature_names
        return X, y, feature_names

    # ------------------------------------------------------------------
    # Sklearn-style interface
    # ------------------------------------------------------------------

    def tune(self, X: np.ndarray, y: np.ndarray) -> None:
        tuner = RandomizedSearchCV(
            estimator=XGBClassifier(**_DEFAULTS),
            param_distributions=param_space,
            n_iter=20,
            cv=3,
            scoring="f1",
            random_state=42,
            n_jobs=-1,
        )
        tuner.fit(X, y)
        best_params = tuner.best_params_
        self._model.set_params(**best_params)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_test)[:, 1].astype(np.float32)

    # ------------------------------------------------------------------
    # Optional extras consumed by report.py
    # ------------------------------------------------------------------

    def feature_importances(self) -> dict[str, float]:
        importances = self._model.feature_importances_
        return dict(zip(self._feature_names, importances.tolist()))

    def get_config(self) -> dict:
        """
        Return a serialisable dict describing this model's full configuration.
        Consumed by report.py to write config.json alongside other artefacts.
        """
        dims_used = list(self._dims) if self._dims is not None else list(DEFAULT_MATRYOSHKA_DIMS)
        return {
            "model_class": "XGBoostModel",
            "matryoshka_dims": dims_used,
            "hyperparams": {k: v for k, v in self._params.items()},
            "n_features": len(self._feature_names),
            "feature_names": self._feature_names,
        }
