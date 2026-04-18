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
from hyperparameter_tuning import OptunaSearchCV


_DEFAULTS = dict(
    n_estimators=700,
    early_stopping_rounds=50,
    min_child_weight=1,
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

param_space = {'max_depth': {'type': 'int', 'low': 3, 'high': 12},
               'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
               'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
               'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0}, #prevents overfitting by sampling a fraction of training data
               'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0}, #prevents overfitting by sampling a fraction of features
               'reg_alpha': {'type': 'float', 'low': 1e-2, 'high': 100.0, 'log': True} #regularisation strength (L2)
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
        self._last_tuner = None
        self._tuning_info: dict[str, object] = {
            "enabled": False,
        }

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
            n_iter=25,
            cv=5,
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
        self._last_tuner = tuner
        self._tuning_info = {
            "enabled": True,
            "method": "RandomizedSearchCV",
            "best_cv_score": float(best_score),
            "best_params": best_params,
        }
    
    def tune_optuna(self, X: np.ndarray, y: np.ndarray) -> None:
        tuner = OptunaSearchCV(
            estimator=XGBClassifier(**_DEFAULTS),
            param_distributions=param_space,
            n_trials=20,
            cv=5,
            scoring="f1",
            random_state=42
        )
        tuner.fit(X, y)
        best_params = tuner.get_best_params()
        best_score = tuner.get_best_score()
        print("Best hyperparameters:", best_params)
        self._params.update(best_params)
        self._model.set_params(**best_params)
        self._last_tuner = tuner
        self._tuning_info = {
            "enabled": True,
            "method": "OptunaSearchCV",
            "best_cv_score": float(best_score),
            "best_params": best_params,
        }

    # ------------------------------------------------------------------
    # Hooks used by experiments/tune.py (the dedicated tuning entry point)
    # ------------------------------------------------------------------

    @classmethod
    def get_tuning_spec(cls) -> dict:
        """
        Describe how this model should be tuned.

        Consumed by experiments/tune.py. Returns a fresh estimator seeded with
        the same _DEFAULTS that __init__ uses, plus the Optuna-style search
        space and the scoring metric. Kept as a classmethod so tune.py doesn't
        have to instantiate the model just to discover its tuning config.
        """
        return {
            "estimator":   XGBClassifier(**_DEFAULTS),
            "param_space": param_space,
            "scoring":     "f1",
        }

    def apply_tuned_params(
        self,
        best_params: dict,
        *,
        source: str | None = None,
        cv_score: float | None = None,
        method: str = "external",
    ) -> None:
        """
        Apply a dict of hyperparameters produced by an out-of-band tuning run
        (e.g. by experiments/tune.py writing best_params.json).

        Intentionally parallel to what tune_optuna() does internally, so the
        resulting experiment report is indistinguishable from an in-process
        tuned run — except for the `source` field, which records the JSON
        file the params came from.
        """
        self._params.update(best_params)
        self._model.set_params(**best_params)
        self._tuning_info = {
            "enabled":       True,
            "method":        method,
            "best_cv_score": float(cv_score) if cv_score is not None else None,
            "best_params":   dict(best_params),
            "source":        source,
        }

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:

        try:
            self._model.fit(X_train, y_train)
        except ValueError as exc:
            message = str(exc)
            if "validation dataset" not in message and "early stopping" not in message:
                raise

            # If no validation set is provided, disable early stopping and refit.
            self._model.set_params(early_stopping_rounds=None)
            self._model.fit(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_test)[:, 1].astype(np.float32)

    # ------------------------------------------------------------------
    # Optional extras consumed by report.py
    # ------------------------------------------------------------------

    def feature_importances(self) -> dict[str, float]:
        importances = self._model.feature_importances_
        return dict(zip(self._feature_names, importances.tolist()))

    def get_tuner(self):
        return self._last_tuner

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
            "tuning": self._tuning_info,
            "n_features": len(self._feature_names),
            "feature_names": self._feature_names,
        }
