"""
featurizers/char_ngram.py — Train-fitted character n-gram featurizer for question pairs.

This featurizer must be fit on training questions only before being used to
transform PairRecords.  It computes similarity features derived from
bag-of-character-n-gram representations (n = 1 … 8).

Two variants are produced:
  • TF-IDF reweighted (analyzer='char_wb', sublinear_tf=True)
  • Binary occurrence (plain presence/absence, no TF-IDF weighting)

Feature groups
--------------
(A) TF-IDF character n-gram similarity
    char_tfidf_cosine_sim  — cosine similarity of the two TF-IDF vectors
    char_tfidf_l1_diff     — L1  norm of |tfidf(q1) - tfidf(q2)|
    char_tfidf_l2_diff     — L2  norm of |tfidf(q1) - tfidf(q2)|
    char_tfidf_dot         — dot product of raw (un-normalised) TF-IDF vectors

(B) Binary (unweighted) character n-gram overlap
    char_bin_cosine_sim    — cosine similarity of binary indicator vectors
    char_bin_jaccard       — |support(q1) ∩ support(q2)| / |support(q1) ∪ support(q2)|
                             ("support" = the set of n-grams present in the question)

Performance note
----------------
Vectors are batch-computed during fit() and cached keyed by question string.
transform() is therefore O(1) per pair for seen questions.
Unseen questions are computed and cached on demand.

Usage
-----
    from featurizers import CharNgramFeaturizer

    featurizer = CharNgramFeaturizer()
    train_qs = [r.question1 for r in train] + [r.question2 for r in train]
    featurizer.fit(train_qs)
    featurizer.cache_questions([r.question1 for r in test] + [r.question2 for r in test])

    feats = featurizer.transform(pair_record)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

if TYPE_CHECKING:
    from data import PairRecord


_LOG_PREFIX = "[CharNgramFeaturizer]"


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


class CharNgramFeaturizer:
    """
    Train-fitted featurizer based on bag-of-character-n-grams (1 ≤ n ≤ 8).

    Parameters
    ----------
    ngram_range : tuple[int, int]
        (min_n, max_n) for character n-grams.  Default (1, 8).
    max_features : int | None
        Vocabulary cap for each internal TfidfVectorizer.
    sublinear_tf : bool
        Apply sublinear (log) TF scaling in the TF-IDF vectorizer.
    analyzer : str
        'char_wb' (pads at word boundaries, default) or 'char'.
    verbose : bool
        If True (default), print progress logs during fit / caching.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 8),
        max_features: int | None = 100_000,
        sublinear_tf: bool = True,
        analyzer: str = "char_wb",
        *,
        verbose: bool = True,
    ) -> None:
        self._ngram_range = ngram_range
        self._max_features = max_features
        self._sublinear_tf = sublinear_tf
        self._analyzer = analyzer
        self._verbose = verbose

        self._tfidf_vec: TfidfVectorizer | None = None
        self._fitted: bool = False

        # cache: question → (raw_tfidf_vec, normed_tfidf_vec, binary_vec)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"{_LOG_PREFIX} {msg}", flush=True)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, questions: list[str]) -> "CharNgramFeaturizer":
        """
        Fit on a flat list of question strings (training data only).
        All unique questions are automatically cached after fitting.
        """
        t0 = time.time()
        n_docs = len(questions)
        n_unique = len(set(questions))
        self._log(
            f"fit(): starting char-ngram TF-IDF fit on {n_docs:,} questions "
            f"({n_unique:,} unique) | "
            f"analyzer={self._analyzer!r}, ngram_range={self._ngram_range}, "
            f"max_features={self._max_features}, sublinear_tf={self._sublinear_tf}"
        )

        self._tfidf_vec = TfidfVectorizer(
            analyzer=self._analyzer,
            ngram_range=self._ngram_range,
            max_features=self._max_features,
            sublinear_tf=self._sublinear_tf,
            smooth_idf=True,
        )
        self._tfidf_vec.fit(questions)
        self._fitted = True

        fit_elapsed = time.time() - t0
        self._log(
            f"fit(): TF-IDF fit complete in {_fmt_secs(fit_elapsed)} | "
            f"vocab_size={len(self._tfidf_vec.vocabulary_):,}"
        )

        self._log(
            f"fit(): pre-caching vectors for {n_unique:,} unique training questions …"
        )
        self.cache_questions(questions)

        total_elapsed = time.time() - t0
        self._log(
            f"fit(): done in {_fmt_secs(total_elapsed)} "
            f"(cache size={len(self._cache):,} vectors)"
        )
        return self

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def cache_questions(self, questions: list[str]) -> None:
        """
        Pre-compute and cache vectors for a list of questions.
        Safe to call multiple times; already-cached strings are skipped.
        """
        self._check_fitted()
        assert self._tfidf_vec is not None

        t0 = time.time()
        n_requested = len(questions)
        unique = [q for q in dict.fromkeys(questions) if q not in self._cache]
        n_new = len(unique)
        n_cached_hits = n_requested - n_new if n_requested >= n_new else 0

        if not unique:
            self._log(
                f"cache_questions(): nothing to do "
                f"({n_requested:,} requested, all already cached)"
            )
            return

        self._log(
            f"cache_questions(): transforming {n_new:,} new questions "
            f"({n_requested:,} requested, {n_cached_hits:,} already in cache) | "
            f"vocab_size={len(self._tfidf_vec.vocabulary_):,}"
        )

        # TF-IDF vectors (sparse → dense)
        sparse_tfidf = self._tfidf_vec.transform(unique)
        dense_tfidf  = sparse_tfidf.toarray().astype(np.float32)

        # L2-normalise for cosine similarity
        norms  = np.linalg.norm(dense_tfidf, axis=1, keepdims=True)
        normed = dense_tfidf / np.clip(norms, 1e-12, None)

        # Binary vectors (1 where TF-IDF > 0, else 0)
        binary = (dense_tfidf > 0).astype(np.float32)

        for q, raw, norm, bin_vec in zip(unique, dense_tfidf, normed, binary):
            self._cache[q] = (raw, norm, bin_vec)

        elapsed = time.time() - t0
        rate = n_new / elapsed if elapsed > 0 else float("inf")
        self._log(
            f"cache_questions(): cached {n_new:,} vectors in {_fmt_secs(elapsed)} "
            f"({rate:,.0f} q/s) | total cache size={len(self._cache):,}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "CharNgramFeaturizer is not fitted. "
                "Call .fit(questions) with training questions first."
            )

    def _get_vectors(
        self, text: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (raw_tfidf, normed_tfidf, binary) vectors, using cache."""
        if text not in self._cache:
            # on-demand for unseen strings
            self.cache_questions([text])
        return self._cache[text]

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, r: "PairRecord") -> dict[str, float]:
        """
        Compute character n-gram similarity features for one pair.

        Returns
        -------
        dict[str, float]
            char_tfidf_cosine_sim, char_tfidf_l1_diff, char_tfidf_l2_diff,
            char_tfidf_dot,
            char_bin_cosine_sim, char_bin_jaccard
        """
        self._check_fitted()

        raw1, norm1, bin1 = self._get_vectors(r.question1)
        raw2, norm2, bin2 = self._get_vectors(r.question2)

        # (A) TF-IDF reweighted features
        diff = np.abs(raw1 - raw2)
        char_tfidf_cosine_sim = float(np.dot(norm1, norm2))
        char_tfidf_l1_diff    = float(diff.sum())
        char_tfidf_l2_diff    = float(np.linalg.norm(diff))
        char_tfidf_dot        = float(np.dot(raw1, raw2))

        # (B) Binary overlap features
        # Cosine similarity on binary vectors
        bin_norm1 = np.linalg.norm(bin1)
        bin_norm2 = np.linalg.norm(bin2)
        bin_cos_den = bin_norm1 * bin_norm2
        char_bin_cosine_sim = (
            float(np.dot(bin1, bin2) / bin_cos_den)
            if bin_cos_den > 1e-12 else 0.0
        )

        # Jaccard on n-gram supports
        inter = float(np.minimum(bin1, bin2).sum())   # both > 0
        union = float(np.maximum(bin1, bin2).sum())   # either > 0
        char_bin_jaccard = inter / union if union > 1e-12 else 0.0

        return {
            "char_tfidf_cosine_sim": char_tfidf_cosine_sim,
            "char_tfidf_l1_diff":    char_tfidf_l1_diff,
            "char_tfidf_l2_diff":    char_tfidf_l2_diff,
            "char_tfidf_dot":        char_tfidf_dot,
            "char_bin_cosine_sim":   char_bin_cosine_sim,
            "char_bin_jaccard":      char_bin_jaccard,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = f"fitted, vocab_size={len(self._tfidf_vec.vocabulary_)}" \
            if self._fitted and self._tfidf_vec is not None else "not fitted"
        return (
            f"CharNgramFeaturizer("
            f"ngram_range={self._ngram_range}, "
            f"max_features={self._max_features}, "
            f"analyzer={self._analyzer!r}, "
            f"{status})"
        )
