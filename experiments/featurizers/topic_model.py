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

import time
from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

if TYPE_CHECKING:
    from data import PairRecord


_LOG_PREFIX = "[TopicModelFeaturizer]"


def _fmt_secs(seconds: float) -> str:
    """Format seconds as a short human-readable string."""
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{seconds:.2f}s"


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
    verbose : bool
        If True (default), print progress logs during fit / caching.
    """

    def __init__(
        self,
        n_components: int = 100,
        max_tfidf_features: int | None = 50_000,
        lda_max_iter: int = 10,
        random_state: int = 42,
        *,
        verbose: bool = True,
    ) -> None:
        self._n_components = n_components
        self._max_tfidf_features = max_tfidf_features
        self._lda_max_iter = lda_max_iter
        self._random_state = random_state
        self._verbose = verbose

        self._fitted = False

        # Pipelines (set during fit)
        self._lsi_pipeline: Pipeline | None = None   # TF-IDF → SVD → L2-norm
        self._lda_tfidf: TfidfVectorizer | None = None
        self._lda_model: LatentDirichletAllocation | None = None

        # Caches: question → embedding vector
        self._lsi_cache: dict[str, np.ndarray] = {}  # L2-normalised LSI vector
        self._lda_cache: dict[str, np.ndarray] = {}  # raw (sums-to-1) LDA vector

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"{_LOG_PREFIX} {msg}", flush=True)

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
        t_total = time.time()
        n_docs = len(questions)
        n_unique = len(set(questions))
        self._log(
            f"fit(): starting on {n_docs:,} questions ({n_unique:,} unique) | "
            f"n_components={self._n_components}, "
            f"max_tfidf_features={self._max_tfidf_features}, "
            f"lda_max_iter={self._lda_max_iter}"
        )

        # ---- LSI pipeline (TF-IDF -> TruncatedSVD -> L2-normalise) ------
        self._log("fit(): [1/4] Fitting TF-IDF vectorizer for LSI …")
        t_stage = time.time()
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

        self._log(
            f"fit(): [2/4] Fitting LSI pipeline (TF-IDF → TruncatedSVD[k={self._n_components}] → L2-norm) …"
        )
        self._lsi_pipeline.fit(questions)
        lsi_vocab = len(tfidf_lsi.vocabulary_)
        try:
            explained = float(svd.explained_variance_ratio_.sum())
        except Exception:
            explained = float("nan")
        self._log(
            f"fit(): [2/4] LSI done in {_fmt_secs(time.time() - t_stage)} | "
            f"tfidf_vocab={lsi_vocab:,}, "
            f"svd_explained_variance_ratio_sum={explained:.4f}"
        )

        # ---- LDA: uses a plain TF-IDF (non-sublinear) or count matrix ----
        self._log("fit(): [3/4] Fitting TF-IDF vectorizer for LDA …")
        t_stage = time.time()
        self._lda_tfidf = TfidfVectorizer(
            max_features=self._max_tfidf_features,
            sublinear_tf=False,   # LDA works better with raw TF or tf (not log-tf)
            smooth_idf=True,
        )
        X_lda = self._lda_tfidf.fit_transform(questions)
        self._log(
            f"fit(): [3/4] LDA TF-IDF done in {_fmt_secs(time.time() - t_stage)} | "
            f"tfidf_vocab={len(self._lda_tfidf.vocabulary_):,}, "
            f"matrix_shape={X_lda.shape}, nnz={X_lda.nnz:,}"
        )

        self._log(
            f"fit(): [4/4] Fitting LDA "
            f"(n_components={self._n_components}, max_iter={self._lda_max_iter}, "
            f"learning_method='batch') — this is usually the slowest stage …"
        )
        t_stage = time.time()
        self._lda_model = LatentDirichletAllocation(
            n_components=self._n_components,
            max_iter=self._lda_max_iter,
            learning_method="batch",
            random_state=self._random_state,
        )
        self._lda_model.fit(X_lda)
        lda_perp = float("nan")
        try:
            lda_perp = float(self._lda_model.perplexity(X_lda))
        except Exception:
            pass
        self._log(
            f"fit(): [4/4] LDA done in {_fmt_secs(time.time() - t_stage)} | "
            f"n_iter={self._lda_model.n_iter_}, "
            f"bound={self._lda_model.bound_:.3f}, "
            f"perplexity={lda_perp:.3f}"
        )
        self._fitted = True

        # Pre-cache all training questions
        self._log(
            f"fit(): pre-caching topic vectors for {n_unique:,} unique training questions …"
        )
        self.cache_questions(questions)

        self._log(
            f"fit(): done in {_fmt_secs(time.time() - t_total)} "
            f"(LSI cache={len(self._lsi_cache):,}, LDA cache={len(self._lda_cache):,})"
        )
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

        t0 = time.time()
        n_requested = len(questions)
        unique = [q for q in dict.fromkeys(questions) if q not in self._lsi_cache]
        n_new = len(unique)
        n_cached_hits = n_requested - n_new if n_requested >= n_new else 0

        if not unique:
            self._log(
                f"cache_questions(): nothing to do "
                f"({n_requested:,} requested, all already cached)"
            )
            return

        self._log(
            f"cache_questions(): embedding {n_new:,} new questions "
            f"({n_requested:,} requested, {n_cached_hits:,} already in cache) "
            f"into LSI + LDA topic spaces (k={self._n_components}) …"
        )

        # LSI embeddings (already L2-normalised by the pipeline)
        t_lsi = time.time()
        lsi_vecs = self._lsi_pipeline.transform(unique).astype(np.float32)
        lsi_elapsed = time.time() - t_lsi

        # LDA embeddings — transform gives un-normalised expected topic counts;
        # normalise to a probability simplex (rows sum to 1)
        t_lda = time.time()
        lda_raw  = self._lda_model.transform(
            self._lda_tfidf.transform(unique)
        ).astype(np.float32)
        lda_sums = lda_raw.sum(axis=1, keepdims=True)
        lda_vecs = lda_raw / np.clip(lda_sums, 1e-12, None)
        lda_elapsed = time.time() - t_lda

        for q, lsi_v, lda_v in zip(unique, lsi_vecs, lda_vecs):
            self._lsi_cache[q] = lsi_v
            self._lda_cache[q] = lda_v

        elapsed = time.time() - t0
        rate = n_new / elapsed if elapsed > 0 else float("inf")
        self._log(
            f"cache_questions(): cached {n_new:,} questions in {_fmt_secs(elapsed)} "
            f"(LSI {_fmt_secs(lsi_elapsed)} + LDA {_fmt_secs(lda_elapsed)}, "
            f"{rate:,.0f} q/s) | "
            f"total cache size={len(self._lsi_cache):,}"
        )

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
