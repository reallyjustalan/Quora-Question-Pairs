"""
models/cosine_baseline.py — Cosine-similarity threshold baseline.

Uses only the cosine similarity between the two question embeddings.
"Predicts" duplicate if cos_sim >= threshold.

This is intentionally the simplest possible model — one feature, no training.
"""

from __future__ import annotations

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data import PairRecord
from features import embedding_features, build_matrix


class CosineBaseline:
    """
    Threshold-based baseline on cosine similarity.

    fit() is a no-op (no parameters to learn).
    predict_proba() returns the raw cosine similarity score as the
    "probability", so that report.py can threshold it at 0.5 by default,
    which corresponds to threshold=0.5.  Pass a custom threshold to use
    a different cut-point.

    Interface required by run_experiment.py
    ----------------------------------------
    build_features(records)  → (X, y, feature_names)
    fit(X_train, y_train)    — no-op
    predict_proba(X_test)    → 1-D array (cosine similarities)
    """

    name = "CosineBaseline"

    def __init__(self, threshold: float = 0.76):
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Feature assembly — only cosine similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_fn(r: PairRecord) -> dict[str, float]:
        """Single feature: cosine similarity."""
        # norm_emb1/norm_emb2 removed from PairRecord; compute on the fly.
        cos_sim = float(np.dot(r.emb1, r.emb2)) / max(r.norm1 * r.norm2, 1e-12)
        return {"cos_sim": cos_sim}

    def build_features(
        self, records: list[PairRecord]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        X, feature_names = build_matrix(records, self._feature_fn)
        y = np.array([r.label for r in records], dtype=np.int32)
        return X, y, feature_names

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """No-op — cosine baseline has no trainable parameters."""
        pass

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Returns the raw cosine similarity as a proxy probability.
        run_experiment.py will threshold at `self.threshold` instead of 0.5
        because this model exposes it explicitly.
        """
        # X_test has shape (N, 1) — the single cos_sim column
        return X_test[:, 0].astype(np.float32)
