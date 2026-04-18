"""
models/xgboost_classical.py — XGBoost with the full classical feature suite.

Feature set (in build order):
  1. Matryoshka prefix-slice embedding statistics  (features.matryoshka_embedding_features)
  2. Basic lexical features                        (features.lexical_features)
  3. Classical text-mining features                (features.classical_text_features)
       • Edit / sequence distances
       • Word n-gram Jaccard (1 … 6-grams)
       • Length & surface features
       • Question-word indicator flags
  4. TF-IDF word-level pair features               (featurizers.TfidfPairFeaturizer)
       • Weighted word overlap, rare-word mismatch, top-k IDF alignment
       • TF-IDF diff (mean/max/L1/L2) + cosine similarity
  5. Character n-gram features (1 … 8-grams)       (featurizers.CharNgramFeaturizer)
       • TF-IDF reweighted cosine / L1 / L2 / dot
       • Binary Jaccard / cosine
  6. Topic-model similarity features               (featurizers.TopicModelFeaturizer)
       • LSI cosine / L1 / L2
       • LDA cosine / Hellinger / L1

All three featurizers are fit on training questions only (inside build_features,
before the train/test split is applied).  Test questions are cached separately
to avoid any leakage.

Usage (via run_experiment.py)
-----------------------------
    cd experiments
    python run_experiment.py --model xgboost_classical --name xgboost_classical_v1
"""

from __future__ import annotations

import os
import sys

import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data import PairRecord
from features import (
    build_matrix,
    matryoshka_classical_features,
    classical_text_features,
    DEFAULT_MATRYOSHKA_DIMS,
)
from featurizers import TfidfPairFeaturizer
from featurizers.char_ngram import CharNgramFeaturizer
from featurizers.topic_model import TopicModelFeaturizer
from hyperparameter_tuning import OptunaSearchCV


# ---------------------------------------------------------------------------
# Default hyperparameters (same as XGBoostModel for apples-to-apples comparison)
# ---------------------------------------------------------------------------

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

param_space = {
    "max_depth":        {"type": "int",   "low": 3,    "high": 12},
    "min_child_weight": {"type": "int",   "low": 1,    "high": 10},
    "learning_rate":    {"type": "float", "low": 0.01, "high": 0.3,  "log": True},
    "n_estimators":     {"type": "int",   "low": 100,  "high": 1000},
    "subsample":        {"type": "float", "low": 0.5,  "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.5,  "high": 1.0},
    "reg_alpha":        {"type": "float", "low": 1e-2, "high": 100.0, "log": True},
}


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class XGBoostClassicalModel:
    """
    XGBoost with the full classical feature suite.

    Parameters
    ----------
    matryoshka_dims : tuple[int, ...] | None
        Embedding prefix dimensions to use.  None → DEFAULT_MATRYOSHKA_DIMS.
    tfidf_max_features : int | None
        Vocabulary cap for the word-level TF-IDF featurizer.
    char_max_features : int | None
        Vocabulary cap for the char-ngram TF-IDF featurizer.
    topic_n_components : int
        Number of latent topics for both LDA and LSI.
    topic_lda_max_iter : int
        LDA EM iterations (increase for better convergence at the cost of speed).
    **kwargs
        Additional keyword arguments forwarded to XGBClassifier.
    """

    name = "XGBoost + Classical Features"

    def __init__(
        self,
        matryoshka_dims: tuple[int, ...] | None = None,
        tfidf_max_features: int | None = 50_000,
        char_max_features: int | None = 100_000,
        topic_n_components: int = 100,
        topic_lda_max_iter: int = 10,
        **kwargs,
    ) -> None:
        params = {**_DEFAULTS, **kwargs}
        self._model = XGBClassifier(**params)
        self._dims = matryoshka_dims
        self._params = params
        self._feature_names: list[str] = []
        self._last_tuner = None
        self._tuning_info: dict[str, object] = {"enabled": False}

        # Train-fitted featurizers (instantiated here, fit in build_features)
        self._tfidf   = TfidfPairFeaturizer(max_features=tfidf_max_features)
        self._char    = CharNgramFeaturizer(max_features=char_max_features)
        self._topics  = TopicModelFeaturizer(
            n_components=topic_n_components,
            lda_max_iter=topic_lda_max_iter,
        )

    @property
    def matryoshka_dims(self) -> tuple[int, ...] | None:
        return self._dims

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------

    def _make_feature_fn(self, train_records: list[PairRecord]):
        """
        Fit the three train-dependent featurizers on training questions,
        then return a closure that produces the full feature dict for any
        PairRecord (training or test).

        Parameters
        ----------
        train_records : list[PairRecord]
            The training split only — featurizers must never see test data.

        Returns
        -------
        Callable[[PairRecord], dict[str, float]]
        """
        train_qs = (
            [r.question1 for r in train_records]
            + [r.question2 for r in train_records]
        )

        print("[XGBoostClassicalModel] Fitting TF-IDF featurizer …", flush=True)
        self._tfidf.fit(train_qs)

        print("[XGBoostClassicalModel] Fitting char-ngram featurizer …", flush=True)
        self._char.fit(train_qs)

        print("[XGBoostClassicalModel] Fitting topic-model featurizer …", flush=True)
        self._topics.fit(train_qs)

        dims = self._dims

        def _feature_fn(r: PairRecord) -> dict[str, float]:
            return {
                **matryoshka_classical_features(r, dims=dims),
                **self._tfidf.transform(r),
                **self._char.transform(r),
                **self._topics.transform(r),
            }

        return _feature_fn

    def build_features(
        self,
        records: list[PairRecord],
        train_idx: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Build the full feature matrix for all records.

        Parameters
        ----------
        records   : all PairRecords (train + test combined)
        train_idx : indices of the training rows.  If None, ALL records are
                    used to fit the featurizers (useful for a quick smoke-test
                    where no split has been computed yet).

        Returns
        -------
        X             : float32 array, shape (N, F)
        y             : int32  array, shape (N,)
        feature_names : list[str], length F
        """
        if train_idx is None:
            train_records = records
        else:
            train_records = [records[i] for i in train_idx]

        feature_fn = self._make_feature_fn(train_records)

        # Cache test-question vectors for speed
        if train_idx is not None:
            train_set = set(train_idx.tolist())
            test_qs = []
            for i, r in enumerate(records):
                if i not in train_set:
                    test_qs.append(r.question1)
                    test_qs.append(r.question2)
            if test_qs:
                print(
                    "[XGBoostClassicalModel] Caching test-question vectors …",
                    flush=True,
                )
                self._tfidf.cache_questions(test_qs)
                self._char.cache_questions(test_qs)
                self._topics.cache_questions(test_qs)

        X, feature_names = build_matrix(
            records, feature_fn, log_prefix="[XGBoostClassicalModel]"
        )
        y = np.array([r.label for r in records], dtype=np.int32)
        self._feature_names = feature_names
        return X, y, feature_names

    # ------------------------------------------------------------------
    # Sklearn-style interface
    # ------------------------------------------------------------------

    def tune_optuna(self, X: np.ndarray, y: np.ndarray) -> None:
        tuner = OptunaSearchCV(
            estimator=XGBClassifier(**_DEFAULTS),
            param_distributions=param_space,
            n_trials=20,
            cv=5,
            scoring="f1",
            random_state=42,
        )
        tuner.fit(X, y)
        best_params = tuner.get_best_params()
        best_score  = tuner.get_best_score()
        print("Best hyperparameters:", best_params)
        self._params.update(best_params)
        self._model.set_params(**best_params)
        self._last_tuner = tuner
        self._tuning_info = {
            "enabled":       True,
            "method":        "OptunaSearchCV",
            "best_cv_score": float(best_score),
            "best_params":   best_params,
        }

    @classmethod
    def get_tuning_spec(cls) -> dict:
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
            msg = str(exc)
            if "validation dataset" not in msg and "early stopping" not in msg:
                raise
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
        dims_used = (
            list(self._dims) if self._dims is not None
            else list(DEFAULT_MATRYOSHKA_DIMS)
        )
        return {
            "model_class":      "XGBoostClassicalModel",
            "matryoshka_dims":  dims_used,
            "hyperparams":      {k: v for k, v in self._params.items()},
            "tuning":           self._tuning_info,
            "n_features":       len(self._feature_names),
            "feature_names":    self._feature_names,
            "featurizers": {
                "tfidf":  repr(self._tfidf),
                "char":   repr(self._char),
                "topics": repr(self._topics),
            },
        }
