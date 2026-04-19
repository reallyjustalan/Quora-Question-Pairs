"""
Random Forest with top-k feature selection.

Workflow
--------
1. Build the FULL matryoshka + lexical feature matrix
2. Train an internal Random Forest on the full feature set
3. Rank features by feature importance
4. Keep only the top-k features
5. Retrain a second Random Forest on the reduced feature set
6. Use the reduced model for prediction

This lets the existing run_experiment.py pipeline stay unchanged:
    build_features(records) -> X_reduced, y, selected_feature_names
    fit(X_train, y_train)
    predict_proba(X_test)

Feature set source
------------------
Uses matryoshka_all_features(...) from features.py, then performs
importance-based column selection internally.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data import PairRecord
from features import build_matrix, matryoshka_all_features, DEFAULT_MATRYOSHKA_DIMS


# Default hyperparameters for the selector model and final reduced model
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
               'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 20}, #minimum number of samples required to be at a leaf node
               }

_DEFAULT_K_CANDIDATES = (5, 10, 15, 20, 30, 40, 50)

class RandomForestTopKModel:
    """
    Random Forest model with internal top-k feature selection.

    Interface required by run_experiment.py
    ---------------------------------------
    build_features(records)  -> (X, y, feature_names)
    fit(X_train, y_train)
    predict_proba(X_test)    -> 1-D array of positive-class probabilities
    feature_importances()    -> dict[feature_name, importance]
    get_config()             -> dict
    """

    name = "RandomForest (top-k features)"

    def __init__(
        self,
        k: int = 10,
        k_candidates: tuple[int, ...] | None = None,
        matryoshka_dims: tuple[int, ...] | None = None,
        **kwargs,
    ):
        params = {**_DEFAULTS, **kwargs}

        # First RF ranks features
        self._selector_model = RandomForestClassifier(**params)
        # Second RF is trained only on the selected features
        self._final_model = RandomForestClassifier(**params)

        self._k = int(k)
        self._k_candidates = tuple(sorted(set(k_candidates or _DEFAULT_K_CANDIDATES)))
        self._dims = matryoshka_dims
        self._params = params
        self._tuning_info: dict[str, object] = {
            "enabled": False,
        }

        self._all_feature_names: list[str] = []
        self._selected_feature_names: list[str] = []
        self._selected_indices: list[int] = []

    @property
    def matryoshka_dims(self) -> tuple[int, ...] | None:
        return self._dims

    @property
    def k(self) -> int:
        return self._k

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _feature_fn(self, r: PairRecord) -> dict[str, float]:
        return matryoshka_all_features(r, dims=self._dims)

    def build_features(
        self,
        records: list[PairRecord],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Build the FULL feature matrix first.

        We do not select top-k here because feature importance must be learned
        from the training data during fit(...).

        Returns
        -------
        X_full         : float32 array (N, F_full)
        y              : int32 array   (N,)
        feature_names  : list[str]     full feature names
        """
        X, feature_names = build_matrix(records, self._feature_fn)
        y = np.array([r.label for r in records], dtype=np.int32)

        self._all_feature_names = feature_names
        return X, y, feature_names

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        1. Fit selector RF on full features
        2. Rank features by importance
        3. Keep top-k columns
        4. Fit final RF on reduced feature matrix
        """
        # Train selector on full features
        self._selector_model.fit(X_train, y_train)

        importances = self._selector_model.feature_importances_
        order = np.argsort(importances)[::-1]

        k = min(self._k, X_train.shape[1])
        self._selected_indices = order[:k].tolist()
        self._selected_feature_names = [
            self._all_feature_names[i] for i in self._selected_indices
        ]

        X_train_selected = X_train[:, self._selected_indices]

        # Train final model on reduced feature set
        self._final_model.fit(X_train_selected, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict using only the selected columns.
        """
        if not self._selected_indices:
            raise RuntimeError(
                "Model has not been fitted yet; selected feature indices are missing."
            )

        X_test_selected = X_test[:, self._selected_indices]
        return self._final_model.predict_proba(X_test_selected)[:, 1].astype(np.float32)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def feature_importances(self) -> dict[str, float]:
        """
        Return feature importances of the FINAL reduced model,
        mapped only to the selected feature names.
        """
        if not self._selected_feature_names:
            return {}

        importances = self._final_model.feature_importances_
        return dict(zip(self._selected_feature_names, importances.tolist()))

    def get_config(self) -> dict:
        dims_used = (
            list(self._dims)
            if self._dims is not None
            else list(DEFAULT_MATRYOSHKA_DIMS)
        )

        return {
            "model_class": "RandomForestTopKModel",
            "matryoshka_dims": dims_used,
            "top_k": self._k,
            "k_candidates": list(self._k_candidates),
            "hyperparams": self._params,
            "tuning": self._tuning_info,
            "n_features_full": len(self._all_feature_names),
            "n_features_selected": len(self._selected_feature_names),
            "selected_feature_names": self._selected_feature_names,
        }