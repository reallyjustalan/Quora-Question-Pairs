"""
featurizers/tfidf_pair.py — Train-fitted TF-IDF featurizer for question pairs.

This featurizer must be fit on a corpus of questions (training data only) before
it can transform PairRecords.  It computes five groups of IDF-based features:

(B) Weighted word overlap
    weighted_word_overlap — Σ_{w ∈ W} idf(w) / Σ_{w ∈ q1∪q2} idf(w)
    where W = shared tokens between q1 and q2.
    Interpretation: overlap fraction weighted by how *rare* shared words are.

(C) Rare word mismatch
    rare_word_mismatch_count  — number of high-IDF words exclusive to one question
    rare_word_mismatch_weight — Σ_{w ∈ exclusive} idf(w)
    Interpretation: captures semantic divergence caused by rare discriminative words
    (e.g., "Python" vs "Java").

(D) Max-IDF word alignment
    top1_word_match    — 1 if the single highest-IDF word in q1 also appears in q2
    top3_overlap_count — how many of the top-3 highest-IDF words in q1 appear in q2
    Interpretation: do the most topic-distinctive words agree?

(E) TF-IDF difference vector (reduced to scalars)
    tfidf_diff_mean — mean of |tfidf(q1) - tfidf(q2)|
    tfidf_diff_max  — max  of |tfidf(q1) - tfidf(q2)|
    tfidf_diff_l1   — L1 norm of the difference vector
    tfidf_diff_l2   — L2 norm of the difference vector
    Interpretation: distributional divergence across the full vocabulary.

(F) TF-IDF vector similarity
    tfidf_cosine_sim — cosine similarity between the two TF-IDF vectors
    tfidf_dot        — raw dot product of the (already L2-normalised) TF-IDF vectors
    Interpretation: A direct similarity measure in TF-IDF space.

Performance note
----------------
At transform time, vectors are looked up from a pre-computed cache keyed by
question string (populated during fit_transform / a warm-up pass), so
repeated or batched calls are O(1) per pair rather than O(vocab_size).
For unseen questions the vector is computed on-demand and cached.

Usage example
-------------
    from featurizers import TfidfPairFeaturizer
    from data import PairRecord

    featurizer = TfidfPairFeaturizer()
    train_questions = [r.question1 for r in train_records] + \
                      [r.question2 for r in train_records]
    featurizer.fit(train_questions)

    # Optionally pre-cache all test questions for speed:
    test_questions = [r.question1 for r in test_records] + \
                     [r.question2 for r in test_records]
    featurizer.cache_questions(test_questions)

    # Inside a model's _feature_fn closure:
    def feature_fn(r: PairRecord) -> dict[str, float]:
        return {
            **matryoshka_all_features(r, dims=dims),
            **featurizer.transform(r),
        }
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

if TYPE_CHECKING:
    from data import PairRecord


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IDF percentile above which a word is considered "rare / high-IDF".
# At 75th percentile roughly the top quarter of vocabulary by rarity is used.
_IDF_RARE_PERCENTILE: float = 75.0

# Number of top-IDF words to look at for feature (D).
_TOP_K_ALIGNMENT: int = 3


# ---------------------------------------------------------------------------
# Simple tokeniser matching the one in features.py
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lower-case, split on whitespace/punctuation, drop empty tokens."""
    return [tok for tok in re.split(r"[\s\W]+", (text or "").lower()) if tok]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TfidfPairFeaturizer:
    """
    Train-fitted featurizer that produces IDF-based pair features.

    Parameters
    ----------
    max_features : int | None
        Maximum vocabulary size for the underlying TfidfVectorizer.
        None means unlimited.
    idf_rare_percentile : float
        Tokens whose IDF score is above this percentile of the fitted
        vocabulary are considered "rare" for feature (C).
    top_k_alignment : int
        Number of highest-IDF tokens per question used in feature (D).

    Attributes (available after fit)
    ----------
    vocab_ : dict[str, int]         token → column index in TF-IDF matrix
    idf_   : np.ndarray             IDF weight per vocabulary token
    _idf_rare_threshold : float     IDF cutoff for "rare" words (feature C)
    """

    def __init__(
        self,
        max_features: int | None = 50_000,
        idf_rare_percentile: float = _IDF_RARE_PERCENTILE,
        top_k_alignment: int = _TOP_K_ALIGNMENT,
        *,
        sublinear_tf: bool = True,
    ) -> None:
        self._max_features = max_features
        self._idf_rare_percentile = float(idf_rare_percentile)
        self._top_k = int(top_k_alignment)
        self._sublinear_tf = sublinear_tf

        self._vectorizer: TfidfVectorizer | None = None
        self.vocab_: dict[str, int] = {}
        self.idf_: np.ndarray = np.empty(0)
        self._idf_rare_threshold: float = 0.0
        self._fitted: bool = False

        # Cache: question string → dense L2-normalised TF-IDF vector
        self._vec_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, questions: list[str]) -> "TfidfPairFeaturizer":
        """
        Fit the TF-IDF vocabulary and IDF weights on a list of question strings.

        Call this with training questions only (never include test questions).
        All unique question strings in `questions` are cached automatically so
        that subsequent transform() calls on training data are O(1).

        Parameters
        ----------
        questions : list[str]
            Flat list of question strings.  Typically:
                [r.question1 for r in train_records] +
                [r.question2 for r in train_records]

        Returns
        -------
        self  (for chaining)
        """
        tfidf = TfidfVectorizer(
            tokenizer=_tokenize,
            token_pattern=None,         # we supply our own tokenizer
            max_features=self._max_features,
            sublinear_tf=self._sublinear_tf,
            smooth_idf=True,
        )
        tfidf.fit(questions)

        self._vectorizer = tfidf
        self.vocab_ = tfidf.vocabulary_          # token → col index
        self.idf_ = tfidf.idf_.astype(np.float32)

        self._idf_rare_threshold = float(
            np.percentile(self.idf_, self._idf_rare_percentile)
        )
        self._fitted = True

        # Pre-compute & cache vectors for all training questions
        self.cache_questions(questions)

        return self

    def cache_questions(self, questions: list[str]) -> None:
        """
        Pre-compute and cache TF-IDF vectors for a list of question strings.

        Call this with your test questions after fit() to avoid recomputing
        vectors on every transform() call.

        Parameters
        ----------
        questions : list[str]
        """
        self._check_fitted()
        assert self._vectorizer is not None

        unique = list({q for q in questions if q not in self._vec_cache})
        if not unique:
            return

        sparse = self._vectorizer.transform(unique)   # (n, vocab)
        dense  = sparse.toarray().astype(np.float32)
        # L2-normalise each row for cosine similarity
        norms  = np.linalg.norm(dense, axis=1, keepdims=True)
        normed = dense / np.clip(norms, 1e-12, None)

        for q, raw_vec, norm_vec in zip(unique, dense, normed):
            # Store both raw (for diff features) and normalised (for cosine)
            self._vec_cache[q] = (raw_vec, norm_vec)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "TfidfPairFeaturizer has not been fitted yet. "
                "Call .fit(questions) with your training questions first."
            )

    def _idf_of(self, token: str) -> float:
        """Return the IDF weight for a token, or 0.0 if out-of-vocabulary."""
        idx = self.vocab_.get(token)
        if idx is None:
            return 0.0
        return float(self.idf_[idx])

    def _tfidf_vectors(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (raw_vector, l2_normalised_vector) for a single text string.
        Looks up the cache first; on miss, computes and caches.
        """
        assert self._vectorizer is not None
        if text not in self._vec_cache:
            sparse = self._vectorizer.transform([text])
            raw = sparse.toarray()[0].astype(np.float32)
            norm_val = np.linalg.norm(raw)
            normed = raw / max(norm_val, 1e-12)
            self._vec_cache[text] = (raw, normed)
        return self._vec_cache[text]

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, r: "PairRecord") -> dict[str, float]:
        """
        Compute all IDF-based features for one question pair.

        Returns a flat dict of scalar features (B), (C), (D), (E).

        Parameters
        ----------
        r : PairRecord

        Returns
        -------
        dict[str, float]
            Keys:
              weighted_word_overlap,
              rare_word_mismatch_count, rare_word_mismatch_weight,
              top1_word_match, top3_overlap_count,
              tfidf_diff_mean, tfidf_diff_max, tfidf_diff_l1, tfidf_diff_l2
        """
        self._check_fitted()

        tokens1 = _tokenize(r.question1)
        tokens2 = _tokenize(r.question2)
        set1, set2 = set(tokens1), set(tokens2)

        shared    = set1 & set2
        union     = set1 | set2
        exclusive = set1.symmetric_difference(set2)   # in exactly one question

        # ---------------------------------------------------------------
        # (B) Weighted word overlap
        # ---------------------------------------------------------------
        shared_idf_sum = sum(self._idf_of(w) for w in shared)
        union_idf_sum  = sum(self._idf_of(w) for w in union)
        weighted_word_overlap = (
            shared_idf_sum / union_idf_sum if union_idf_sum > 0.0 else 0.0
        )

        # ---------------------------------------------------------------
        # (C) Rare word mismatch
        # ---------------------------------------------------------------
        rare_exclusive = [
            w for w in exclusive
            if self._idf_of(w) >= self._idf_rare_threshold
        ]
        rare_word_mismatch_count  = float(len(rare_exclusive))
        rare_word_mismatch_weight = sum(self._idf_of(w) for w in rare_exclusive)

        # ---------------------------------------------------------------
        # (D) Max-IDF word alignment
        # Top-k highest-IDF words from q1, checked against q2
        # ---------------------------------------------------------------
        # Sort q1 tokens by IDF descending (deduplicated)
        tokens1_ranked = sorted(set1, key=lambda w: self._idf_of(w), reverse=True)

        top1_word = tokens1_ranked[0] if tokens1_ranked else None
        top1_word_match = float(top1_word in set2) if top1_word is not None else 0.0

        top3_words = tokens1_ranked[: self._top_k]
        top3_overlap_count = float(sum(1 for w in top3_words if w in set2))

        # ---------------------------------------------------------------
        # (E) TF-IDF difference vector (reduced)
        # ---------------------------------------------------------------
        vec1, norm_vec1 = self._tfidf_vectors(r.question1)
        vec2, norm_vec2 = self._tfidf_vectors(r.question2)
        diff = np.abs(vec1 - vec2)

        tfidf_diff_mean = float(diff.mean())
        tfidf_diff_max  = float(diff.max())
        tfidf_diff_l1   = float(diff.sum())
        tfidf_diff_l2   = float(np.linalg.norm(diff))

        # ---------------------------------------------------------------
        # (F) TF-IDF vector similarity
        # ---------------------------------------------------------------
        tfidf_cosine_sim = float(np.dot(norm_vec1, norm_vec2))
        tfidf_dot        = float(np.dot(vec1, vec2))

        return {
            # (B)
            "weighted_word_overlap":    weighted_word_overlap,
            # (C)
            "rare_word_mismatch_count":  rare_word_mismatch_count,
            "rare_word_mismatch_weight": rare_word_mismatch_weight,
            # (D)
            "top1_word_match":           top1_word_match,
            "top3_overlap_count":        top3_overlap_count,
            # (E)
            "tfidf_diff_mean":           tfidf_diff_mean,
            "tfidf_diff_max":            tfidf_diff_max,
            "tfidf_diff_l1":             tfidf_diff_l1,
            "tfidf_diff_l2":             tfidf_diff_l2,
            # (F)
            "tfidf_cosine_sim":          tfidf_cosine_sim,
            "tfidf_dot":                 tfidf_dot,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = (
            f"fitted, vocab={len(self.vocab_)}, "
            f"idf_rare_threshold={self._idf_rare_threshold:.3f}"
            if self._fitted
            else "not fitted"
        )
        return (
            f"TfidfPairFeaturizer("
            f"max_features={self._max_features}, "
            f"idf_rare_percentile={self._idf_rare_percentile}, "
            f"top_k_alignment={self._top_k}, "
            f"{status})"
        )
