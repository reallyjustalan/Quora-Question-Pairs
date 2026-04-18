"""
models/catboost_model.py — CatBoost classifier with matryoshka-sliced features.

Feature set (mirrors XGBoostModel):
  - Matryoshka prefix-slice embedding statistics
  - Lexical overlap / length features

Pass ``matryoshka_dims`` to control which prefix slices are used.
Omit it (or pass None) to use the library default defined in features.py.
"""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data import PairRecord
from features import build_matrix, matryoshka_all_features, DEFAULT_MATRYOSHKA_DIMS
from hyperparameter_tuning import RandomizedSearchCV
from hyperparameter_tuning import OptunaSearchCV


# Default hyper-parameters — override by subclassing or passing kwargs to __init__
_DEFAULTS = dict(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="F1",
    random_seed=42,
    verbose=100,
)

param_space = {'iterations': {'type': 'int', 'low': 100, 'high': 1000}, #number of trees
               'depth': {'type': 'int', 'low': 4, 'high': 10}, #maximum depth of trees -- large trees are more expressive but may overfit
               'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True}, #step size - affects speed of convergence to minima
               'l2_leaf_reg': {'type': 'float', 'low': 1.0, 'high': 20.0, 'log': True} #strength of Ridge Regularisation
               }

class CatBoostModel:
    """
    Plug-and-play wrapper around CatBoostClassifier.

    Interface required by run_experiment.py
    ----------------------------------------
    build_features(records)  → (X, y, feature_names)
    fit(X_train, y_train)
    predict_proba(X_test)    → 1-D array of positive-class probabilities
    feature_importances()    → dict[feature_name, importance]  (optional)
    get_config()             → dict  (hyperparams + feature config)
    """

    name = "CatBoost"

    def __init__(
        self,
        matryoshka_dims: tuple[int, ...] | None = None,
        **kwargs,
    ):
        params = {**_DEFAULTS, **kwargs}
        self._model = CatBoostClassifier(**params)
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
        """Matryoshka-sliced embedding features + lexical features."""
        return matryoshka_all_features(r, dims=self._dims)

    def build_features(
        self, records: list[PairRecord]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Build the feature matrix from a list of PairRecords.

        Returns
        -------
        X            : float32 array  (N, F)
        y            : int32  array   (N,)
        feature_names: list[str]
        """
        X, feature_names = build_matrix(records, self._feature_fn)
        y = np.array([r.label for r in records], dtype=np.int32)
        self._feature_names = feature_names
        return X, y, feature_names

    # ------------------------------------------------------------------
    # Sklearn-style interface
    # ------------------------------------------------------------------

    def tune(self, X: np.ndarray, y: np.ndarray) -> None:
        tuner = RandomizedSearchCV(
            estimator=CatBoostClassifier(**_DEFAULTS),
            param_distributions=param_space,
            n_iter=20,
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
            estimator=CatBoostClassifier(**_DEFAULTS),
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
        space and the scoring metric.
        """
        return {
            "estimator":   CatBoostClassifier(**_DEFAULTS),
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
        Apply hyperparameters produced by an out-of-band tuning run
        (e.g. experiments/tune.py writing best_params.json).
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

        self._model.fit(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_test)[:, 1].astype(np.float32)

    # ------------------------------------------------------------------
    # Optional extras consumed by report.py
    # ------------------------------------------------------------------

    def feature_importances(self) -> dict[str, float]:
        """Returns a name → importance mapping (only valid after fit)."""
        importances = self._model.get_feature_importance()
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
            "model_class": "CatBoostModel",
            "matryoshka_dims": dims_used,
            "hyperparams": {k: v for k, v in self._params.items()},
            "tuning": self._tuning_info,
            "n_features": len(self._feature_names),
            "feature_names": self._feature_names,
        }
