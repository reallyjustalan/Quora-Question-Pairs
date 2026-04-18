"""
featurizers/topic_model.py — Train-fitted LDA and LSI topic-model featurizer.

Produces similarity features in the latent topic space of the training corpus.
Two complementary topic models are used:

  • LSI / LSA (Latent Semantic Indexing / Analysis) via TruncatedSVD
    A deterministic linear model; captures correlations between terms
    by decomposing the TF-IDF matrix.

  • LDA (Latent Dirichlet Allocation)
    A probabilistic model; each question is represented as a distribution
    over K topics.  Because LDA output is a probability distribution,
    Hellinger distance is a natural (and bounded [0,1]) divergence.

Feature groups
--------------
(A) LSI similarity
    lsi_cosine_sim  — cosine similarity in the K-dim LSI topic space
    lsi_l1_diff     — L1 norm of |lsi(q1) − lsi(q2)|
    lsi_l2_diff     — L2 norm (Euclidean distance) in LSI space

(B) LDA similarity
    lda_cosine_sim     — cosine similarity in the K-dim topic probability space
    lda_hellinger_sim  — 1 − Hellinger distance  (1 = identical distributions)
    lda_l1_diff        — L1 norm of |lda(q1) − lda(q2)|

Performance note
----------------
All question embeddings are computed in batch during fit() / cache_questions()
and stored in a dict keyed by question string.  transform() is therefore O(1)
per pair for previously seen questions.

Usage
-----
    from featurizers import TopicModelFeaturizer

    feat = TopicModelFeaturizer(n_components=100)
    train_qs = [r.question1 for r in train] + [r.question2 for r in train]
    feat.fit(train_qs)
    feat.cache_questions([r.question1 for r in test] + [r.question2 for r in test])

    features = feat.transform(pair_record)   # dict[str, float]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

if TYPE_CHECKING:
    from data import PairRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hellinger_sim(p: np.ndarray, q: np.ndarray) -> float:
    """
    Return 1 − Hellinger distance between two non-negative vectors
    (which need not sum to 1; they are normalised internally).

    Result is in [0, 1]: 1 = identical, 0 = maximally different.
    """
    # Clip to avoid sqrt(negative) from float noise
    p_norm = p / max(p.sum(), 1e-12)
    q_norm = q / max(q.sum(), 1e-12)
    hellinger_dist = float(
        np.linalg.norm(np.sqrt(np.clip(p_norm, 0, None)) - np.sqrt(np.clip(q_norm, 0, None)))
        / np.sqrt(2)
    )
    return 1.0 - hellinger_dist


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TopicModelFeaturizer:
    """
    Train-fitted featurizer using LDA and LSI topic models.

    Parameters
    ----------
    n_components : int
        Number of latent topics / dimensions for both LDA and LSI.
    max_tfidf_features : int | None
        Vocabulary size cap for the internal TF-IDF vectorizer.
    lda_max_iter : int
        Maximum EM iterations for LDA.
    random_state : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 100,
        max_tfidf_features: int | None = 50_000,
        lda_max_iter: int = 10,
        random_state: int = 42,
    ) -> None:
        self._n_components = n_components
        self._max_tfidf_features = max_tfidf_features
        self._lda_max_iter = lda_max_iter
        self._random_state = random_state

        self._fitted = False

        # Pipelines (set during fit)
        self._lsi_pipeline: Pipeline | None = None   # TF-IDF → SVD → L2-norm
        self._lda_tfidf: TfidfVectorizer | None = None
        self._lda_model: LatentDirichletAllocation | None = None

        # Caches: question → embedding vector
        self._lsi_cache: dict[str, np.ndarray] = {}  # L2-normalised LSI vector
        self._lda_cache: dict[str, np.ndarray] = {}  # raw (sums-to-1) LDA vector

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, questions: list[str]) -> "TopicModelFeaturizer":
        """
        Fit both topic models on training questions only.

        Parameters
        ----------
        questions : list[str]
            All training question strings (q1 + q2 for all pairs).

        Returns
        -------
        self
        """
        print("[TopicModelFeaturizer] Fitting TF-IDF for LSI …", flush=True)
        tfidf_lsi = TfidfVectorizer(
            max_features=self._max_tfidf_features,
            sublinear_tf=True,
            smooth_idf=True,
        )
        svd = TruncatedSVD(
            n_components=self._n_components,
            random_state=self._random_state,
        )
        normalizer = Normalizer(copy=False)

        self._lsi_pipeline = Pipeline([
            ("tfidf",   tfidf_lsi),
            ("svd",     svd),
            ("l2norm",  normalizer),
        ])

        print("[TopicModelFeaturizer] Fitting LSI (TruncatedSVD) …", flush=True)
        self._lsi_pipeline.fit(questions)

        # LDA uses a plain count/TF-IDF matrix (LDA prefers raw counts or
        # non-sublinear TF).  We use a separate TfidfVectorizer without sublinear_tf.
        print("[TopicModelFeaturizer] Fitting TF-IDF for LDA …", flush=True)
        self._lda_tfidf = TfidfVectorizer(
            max_features=self._max_tfidf_features,
            sublinear_tf=False,   # LDA works better with raw TF or tf (not log-tf)
            smooth_idf=True,
        )
        X_lda = self._lda_tfidf.fit_transform(questions)

        print(
            f"[TopicModelFeaturizer] Fitting LDA "
            f"(n_components={self._n_components}, max_iter={self._lda_max_iter}) …",
            flush=True,
        )
        self._lda_model = LatentDirichletAllocation(
            n_components=self._n_components,
            max_iter=self._lda_max_iter,
            learning_method="batch",
            random_state=self._random_state,
        )
        self._lda_model.fit(X_lda)
        self._fitted = True

        # Pre-cache all training questions
        self.cache_questions(questions)
        print("[TopicModelFeaturizer] Done fitting.", flush=True)
        return self

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def cache_questions(self, questions: list[str]) -> None:
        """
        Batch-compute and cache topic vectors for a list of questions.
        Already-cached strings are skipped.
        """
        self._check_fitted()
        assert self._lsi_pipeline is not None
        assert self._lda_tfidf is not None
        assert self._lda_model is not None

        unique = [q for q in dict.fromkeys(questions) if q not in self._lsi_cache]
        if not unique:
            return

        # LSI embeddings (already L2-normalised by the pipeline)
        lsi_vecs = self._lsi_pipeline.transform(unique).astype(np.float32)

        # LDA embeddings — transform gives un-normalised expected topic counts;
        # normalise to a probability simplex (rows sum to 1)
        lda_raw  = self._lda_model.transform(
            self._lda_tfidf.transform(unique)
        ).astype(np.float32)
        lda_sums = lda_raw.sum(axis=1, keepdims=True)
        lda_vecs = lda_raw / np.clip(lda_sums, 1e-12, None)

        for q, lsi_v, lda_v in zip(unique, lsi_vecs, lda_vecs):
            self._lsi_cache[q] = lsi_v
            self._lda_cache[q] = lda_v

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "TopicModelFeaturizer is not fitted. "
                "Call .fit(questions) with training questions first."
            )

    def _get_lsi(self, text: str) -> np.ndarray:
        if text not in self._lsi_cache:
            self.cache_questions([text])
        return self._lsi_cache[text]

    def _get_lda(self, text: str) -> np.ndarray:
        if text not in self._lda_cache:
            self.cache_questions([text])
        return self._lda_cache[text]

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, r: "PairRecord") -> dict[str, float]:
        """
        Compute LDA + LSI similarity features for one question pair.

        Returns
        -------
        dict[str, float]
            lsi_cosine_sim, lsi_l1_diff, lsi_l2_diff,
            lda_cosine_sim, lda_hellinger_sim, lda_l1_diff
        """
        self._check_fitted()

        # LSI features
        lsi1 = self._get_lsi(r.question1)
        lsi2 = self._get_lsi(r.question2)
        lsi_diff = np.abs(lsi1 - lsi2)

        lsi_cosine_sim = float(np.dot(lsi1, lsi2))          # already L2-normed
        lsi_l1_diff    = float(lsi_diff.sum())
        lsi_l2_diff    = float(np.linalg.norm(lsi_diff))

        # LDA features
        lda1 = self._get_lda(r.question1)
        lda2 = self._get_lda(r.question2)
        lda_diff = np.abs(lda1 - lda2)

        lda1_norm = np.linalg.norm(lda1)
        lda2_norm = np.linalg.norm(lda2)
        lda_cos_den = lda1_norm * lda2_norm
        lda_cosine_sim = (
            float(np.dot(lda1, lda2) / lda_cos_den)
            if lda_cos_den > 1e-12 else 0.0
        )
        lda_hellinger_sim = _hellinger_sim(lda1, lda2)
        lda_l1_diff       = float(lda_diff.sum())

        return {
            "lsi_cosine_sim":    lsi_cosine_sim,
            "lsi_l1_diff":       lsi_l1_diff,
            "lsi_l2_diff":       lsi_l2_diff,
            "lda_cosine_sim":    lda_cosine_sim,
            "lda_hellinger_sim": lda_hellinger_sim,
            "lda_l1_diff":       lda_l1_diff,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = f"fitted, n_components={self._n_components}" \
            if self._fitted else "not fitted"
        return (
            f"TopicModelFeaturizer("
            f"n_components={self._n_components}, "
            f"lda_max_iter={self._lda_max_iter}, "
            f"{status})"
        )
