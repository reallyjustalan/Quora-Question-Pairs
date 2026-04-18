"""
features.py — Primitive feature-building functions.

These are pure functions that operate on a single PairRecord and return
a flat dict of named scalar values. Models import whatever primitives
they need and assemble their own feature matrix.

None of these functions do any scaling, normalisation, or model logic.
"""

from __future__ import annotations

import difflib
import time

import numpy as np

from data import PairRecord


DEFAULT_MATRYOSHKA_DIMS = (128, 256, 512, 1024, 1536, 2048, 2560)

# Question-starter words used for indicator features.
_QUESTION_STARTERS = (
    "are", "can", "could", "did", "do", "does",
    "has", "have", "how", "if", "is",
    "what", "when", "where", "which", "who", "whom", "whose", "why",
    "will", "would",
)


# ---------------------------------------------------------------------------
# Tokenisation helper (shared)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return (text or "").lower().strip().split()


def _word_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Return all word n-gram tuples of size n from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


# ---------------------------------------------------------------------------
# Embedding-based primitives
# ---------------------------------------------------------------------------

def embedding_features(r: PairRecord) -> dict[str, float]:
    """
    Vector-space features derived from the raw and normalised embeddings.

    Keys returned
    -------------
    cos_sim, dot_raw, euclidean, manhattan,
    abs_diff_mean, abs_diff_max, abs_diff_std,
    prod_mean, prod_std,
    norm1, norm2, norm_diff
    """
    u_raw, v_raw = r.emb1, r.emb2
    u, v = r.norm_emb1, r.norm_emb2

    abs_diff = np.abs(u_raw - v_raw)
    prod = u_raw * v_raw

    return {
        "cos_sim":       float(np.dot(u, v)),
        "dot_raw":       float(np.dot(u_raw, v_raw)),
        "euclidean":     float(np.linalg.norm(u_raw - v_raw)),
        "manhattan":     float(np.abs(u_raw - v_raw).sum()),
        "abs_diff_mean": float(abs_diff.mean()),
        "abs_diff_max":  float(abs_diff.max()),
        "abs_diff_std":  float(abs_diff.std()),
        "prod_mean":     float(prod.mean()),
        "prod_std":      float(prod.std()),
        "norm1":         r.norm1,
        "norm2":         r.norm2,
        "norm_diff":     abs(r.norm1 - r.norm2),
    }


def _resolve_matryoshka_dims(
    emb_dim: int,
    dims: tuple[int, ...] | None,
) -> list[int]:
    """Sanitise requested prefix dimensions against the real embedding size."""
    if dims is None:
        dims = DEFAULT_MATRYOSHKA_DIMS

    resolved: list[int] = []
    seen: set[int] = set()

    for d in dims:
        d = int(d)
        if d <= 0:
            continue
        d = min(d, emb_dim)
        if d not in seen:
            resolved.append(d)
            seen.add(d)

    # Always include the full vector as the final slice.
    if emb_dim not in seen:
        resolved.append(emb_dim)

    return resolved


def matryoshka_embedding_features(
    r: PairRecord,
    dims: tuple[int, ...] | None = None,
) -> dict[str, float]:
    """
    Embedding features computed over matryoshka prefix slices.

    For each prefix dimension d, this returns:
      d{d}_cos_sim, d{d}_dot_raw, d{d}_euclidean, d{d}_manhattan,
      d{d}_abs_diff_mean, d{d}_abs_diff_max, d{d}_abs_diff_std,
      d{d}_prod_mean, d{d}_prod_std
    """
    u_raw, v_raw = r.emb1, r.emb2
    emb_dim = min(len(u_raw), len(v_raw))
    slice_dims = _resolve_matryoshka_dims(emb_dim, dims)

    feats: dict[str, float] = {}

    for d in slice_dims:
        u_d = u_raw[:d]
        v_d = v_raw[:d]

        abs_diff = np.abs(u_d - v_d)
        prod = u_d * v_d

        u_norm = float(np.linalg.norm(u_d))
        v_norm = float(np.linalg.norm(v_d))
        cos_den = max(u_norm * v_norm, 1e-12)
        cos_sim = float(np.dot(u_d, v_d) / cos_den)

        p = f"d{d}_"
        feats[f"{p}cos_sim"] = cos_sim
        feats[f"{p}dot_raw"] = float(np.dot(u_d, v_d))
        feats[f"{p}euclidean"] = float(np.linalg.norm(u_d - v_d))
        feats[f"{p}manhattan"] = float(abs_diff.sum())
        feats[f"{p}abs_diff_mean"] = float(abs_diff.mean())
        feats[f"{p}abs_diff_max"] = float(abs_diff.max())
        feats[f"{p}abs_diff_std"] = float(abs_diff.std())
        feats[f"{p}prod_mean"] = float(prod.mean())
        feats[f"{p}prod_std"] = float(prod.std())

    return feats


# ---------------------------------------------------------------------------
# Lexical / surface-form primitives
# ---------------------------------------------------------------------------

def lexical_features(r: PairRecord) -> dict[str, float]:
    """
    Token-overlap and character/word-length features derived from the raw
    question strings — no embeddings used.

    Keys returned
    -------------
    len_q1_chars, len_q2_chars, char_len_diff,
    len_q1_words, len_q2_words, word_len_diff,
    token_intersection, token_union,
    jaccard, overlap_min
    """
    q1, q2 = r.question1, r.question2
    t1, t2 = _tokenize(q1), _tokenize(q2)
    s1, s2 = set(t1), set(t2)

    inter = len(s1 & s2)
    union = len(s1 | s2)
    min_len = min(len(s1), len(s2))

    return {
        "len_q1_chars":       float(len(q1)),
        "len_q2_chars":       float(len(q2)),
        "char_len_diff":      float(abs(len(q1) - len(q2))),
        "len_q1_words":       float(len(t1)),
        "len_q2_words":       float(len(t2)),
        "word_len_diff":      float(abs(len(t1) - len(t2))),
        "token_intersection": float(inter),
        "token_union":        float(union),
        "jaccard":            _safe_div(inter, union),
        "overlap_min":        _safe_div(inter, min_len) if min_len > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Classical text-mining primitives (no embeddings, no fitting required)
# ---------------------------------------------------------------------------

def classical_text_features(r: PairRecord) -> dict[str, float]:
    """
    Pure lexical / surface-form features that require no trained model.

    Feature groups
    --------------
    (1) Edit / sequence-match distances
    (2) Word n-gram overlap fractions (1 … 6-grams), also conditioned on
        whether the questions start or end with the same word
    (3) Length features (unique-word count, avg word length, char ratio)
    (4) Surface / punctuation counts (capitals, ?, !, digits, math symbols)
    (5) Question-word indicators (starts with "how", "what", …)

    Returns a flat dict[str, float].
    """
    q1: str = r.question1 or ""
    q2: str = r.question2 or ""

    # Raw token lists (lower-cased, split on whitespace)
    t1 = q1.lower().split()
    t2 = q2.lower().split()

    # ------------------------------------------------------------------ #
    # (1) Edit / sequence-match distances                                 #
    # ------------------------------------------------------------------ #
    # difflib.SequenceMatcher.ratio() = 2*M / T  where M = matching chars
    # and T = total chars in both strings.  We use char-level by default.
    sm_char  = difflib.SequenceMatcher(None, q1, q2)
    sm_word  = difflib.SequenceMatcher(None, t1, t2)

    seq_char_ratio  = float(sm_char.ratio())       # char-level similarity
    seq_word_ratio  = float(sm_word.ratio())       # word-level similarity

    # Approximate edit distance from SequenceMatcher:
    # edit_dist ≈ (1 - ratio) * max_len  (not exact Levenshtein, but fast)
    max_char_len = max(len(q1), len(q2), 1)
    edit_dist_approx      = (1.0 - seq_char_ratio) * max_char_len
    edit_dist_norm        = 1.0 - seq_char_ratio   # normalised to [0, 1]

    # Longest common subsequence (char-level): difflib gives us matching blocks
    lcs_len_chars = sum(triple.size for triple in sm_char.get_matching_blocks())
    lcs_len_words = sum(triple.size for triple in sm_word.get_matching_blocks())

    # ------------------------------------------------------------------ #
    # (2) Word n-gram overlap fractions (1 … 6-grams)                    #
    # ------------------------------------------------------------------ #
    # Also produce "when same start / same end" conditional variants.
    same_start_word = int(bool(t1) and bool(t2) and t1[0] == t2[0])
    same_end_word   = int(bool(t1) and bool(t2) and t1[-1] == t2[-1])
    same_start_2gram = int(
        len(t1) >= 2 and len(t2) >= 2 and t1[:2] == t2[:2]
    )

    ngram_feats: dict[str, float] = {}
    for n in range(1, 7):
        ng1 = set(_word_ngrams(t1, n))
        ng2 = set(_word_ngrams(t2, n))
        inter = len(ng1 & ng2)
        union = len(ng1 | ng2)
        jaccard = inter / union if union > 0 else 0.0
        ngram_feats[f"ngram_{n}_jaccard"] = jaccard

    # Common-token percentage (unigrams, token sets)
    s1, s2 = set(t1), set(t2)
    inter_words = len(s1 & s2)
    union_words = len(s1 | s2)
    common_word_pct = inter_words / union_words if union_words > 0 else 0.0

    # Conditional: common_pct when questions start / end the same
    common_pct_when_same_start = common_word_pct if same_start_word else 0.0
    common_pct_when_same_end   = common_word_pct if same_end_word   else 0.0

    # ------------------------------------------------------------------ #
    # (3) Length features                                                 #
    # ------------------------------------------------------------------ #
    len_q1_chars   = float(len(q1))
    len_q2_chars   = float(len(q2))
    len_q1_words   = float(len(t1))
    len_q2_words   = float(len(t2))

    len_q1_unique  = float(len(s1))
    len_q2_unique  = float(len(s2))

    avg_word_len_q1 = (
        float(sum(len(w) for w in t1) / len(t1)) if t1 else 0.0
    )
    avg_word_len_q2 = (
        float(sum(len(w) for w in t2) / len(t2)) if t2 else 0.0
    )

    char_len_ratio = (
        len_q1_chars / len_q2_chars if len_q2_chars > 0.0 else 0.0
    )

    # ------------------------------------------------------------------ #
    # (4) Surface / punctuation features                                  #
    # ------------------------------------------------------------------ #
    n_caps_q1       = float(sum(1 for c in q1 if c.isupper()))
    n_caps_q2       = float(sum(1 for c in q2 if c.isupper()))
    caps_diff       = abs(n_caps_q1 - n_caps_q2)

    n_qmarks_q1     = float(q1.count("?"))
    n_qmarks_q2     = float(q2.count("?"))
    n_excl_q1       = float(q1.count("!"))
    n_excl_q2       = float(q2.count("!"))

    n_digits_q1     = float(sum(1 for c in q1 if c.isdigit()))
    n_digits_q2     = float(sum(1 for c in q2 if c.isdigit()))

    _MATH = set("+-*/=^%<>")
    n_math_q1       = float(sum(1 for c in q1 if c in _MATH))
    n_math_q2       = float(sum(1 for c in q2 if c in _MATH))

    # ------------------------------------------------------------------ #
    # (5) Question-word indicator features                                #
    # ------------------------------------------------------------------ #
    first_q1 = t1[0] if t1 else ""
    first_q2 = t2[0] if t2 else ""

    starter_feats: dict[str, float] = {}
    for word in _QUESTION_STARTERS:
        starter_feats[f"q1_starts_{word}"] = float(first_q1 == word)
        starter_feats[f"q2_starts_{word}"] = float(first_q2 == word)

    q1q2_same_starter = float(
        bool(first_q1) and first_q1 in _QUESTION_STARTERS and first_q1 == first_q2
    )

    # Is the question a question? (ends with ?)
    q1_is_question = float(q1.rstrip().endswith("?"))
    q2_is_question = float(q2.rstrip().endswith("?"))

    # ------------------------------------------------------------------ #
    # Assemble and return                                                 #
    # ------------------------------------------------------------------ #
    feats: dict[str, float] = {
        # (1) Edit / sequence
        "seq_char_ratio":        seq_char_ratio,
        "seq_word_ratio":        seq_word_ratio,
        "edit_dist_approx":      edit_dist_approx,
        "edit_dist_norm":        edit_dist_norm,
        "lcs_len_chars":         float(lcs_len_chars),
        "lcs_len_words":         float(lcs_len_words),
        # (2) N-gram overlap
        **ngram_feats,
        "common_word_pct":               common_word_pct,
        "same_start_word":               float(same_start_word),
        "same_end_word":                 float(same_end_word),
        "same_start_2gram":              float(same_start_2gram),
        "common_pct_when_same_start":    common_pct_when_same_start,
        "common_pct_when_same_end":      common_pct_when_same_end,
        # (3) Length
        "len_q1_chars":          len_q1_chars,
        "len_q2_chars":          len_q2_chars,
        "char_len_diff":         abs(len_q1_chars - len_q2_chars),
        "len_q1_words":          len_q1_words,
        "len_q2_words":          len_q2_words,
        "word_len_diff":         abs(len_q1_words - len_q2_words),
        "len_q1_unique_words":   len_q1_unique,
        "len_q2_unique_words":   len_q2_unique,
        "avg_word_len_q1":       avg_word_len_q1,
        "avg_word_len_q2":       avg_word_len_q2,
        "char_len_ratio":        char_len_ratio,
        # (4) Surface / punctuation
        "n_caps_q1":             n_caps_q1,
        "n_caps_q2":             n_caps_q2,
        "caps_diff":             caps_diff,
        "n_qmarks_q1":           n_qmarks_q1,
        "n_qmarks_q2":           n_qmarks_q2,
        "n_excl_q1":             n_excl_q1,
        "n_excl_q2":             n_excl_q2,
        "n_digits_q1":           n_digits_q1,
        "n_digits_q2":           n_digits_q2,
        "n_math_q1":             n_math_q1,
        "n_math_q2":             n_math_q2,
        # (5) Question-word indicators
        **starter_feats,
        "q1q2_same_starter":     q1q2_same_starter,
        "q1_is_question":        q1_is_question,
        "q2_is_question":        q2_is_question,
    }
    return feats


# ---------------------------------------------------------------------------
# Convenience combiner
# ---------------------------------------------------------------------------

def all_features(r: PairRecord) -> dict[str, float]:
    """Return every available feature for a pair (embedding + lexical)."""
    return {**embedding_features(r), **lexical_features(r)}


def matryoshka_all_features(
    r: PairRecord,
    dims: tuple[int, ...] | None = None,
) -> dict[str, float]:
    """Return matryoshka-sliced embedding features + lexical features."""
    return {
        **matryoshka_embedding_features(r, dims=dims),
        **lexical_features(r),
    }


def matryoshka_classical_features(
    r: PairRecord,
    dims: tuple[int, ...] | None = None,
) -> dict[str, float]:
    """
    Return matryoshka-sliced embedding features + lexical features +
    the full classical text-mining feature set.

    This is the richest pure-Python feature function; it is used by
    XGBoostClassicalModel and its friends.
    """
    return {
        **matryoshka_embedding_features(r, dims=dims),
        **lexical_features(r),
        **classical_text_features(r),
    }


# ---------------------------------------------------------------------------
# Batch helpers — convert a list of PairRecords → (X, feature_names)
# ---------------------------------------------------------------------------

def build_matrix(
    records: list[PairRecord],
    feature_fn,
    log_every: int = 50_000,
    log_prefix: str = "[features]",
) -> tuple[np.ndarray, list[str]]:
    """
    Apply `feature_fn` to every PairRecord and stack into a float32 matrix.

    Parameters
    ----------
    records    : list of PairRecord
    feature_fn : callable(PairRecord) → dict[str, float]
                 e.g. embedding_features, lexical_features, all_features,
                 or any custom function that returns a flat dict.

    Returns
    -------
    X            : np.ndarray  shape (N, F), dtype float32
    feature_names: list[str]   length F, in column order
    """
    if not records:
        raise ValueError("records list is empty")

    def _fmt(seconds: float) -> str:
        s = int(seconds)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m:02d}m {s:02d}s"
        if m:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    # Use first record to discover column order
    sample = feature_fn(records[0])
    feature_names = list(sample.keys())
    n_rows = len(records)
    n_feats = len(feature_names)

    print(
        f"{log_prefix} Building matrix: rows={n_rows}, features={n_feats}",
        flush=True,
    )

    X = np.empty((n_rows, n_feats), dtype=np.float32)
    start = last_log = time.time()

    for i, rec in enumerate(records):
        feat = feature_fn(rec)
        for j, name in enumerate(feature_names):
            X[i, j] = feat[name]

        done = i + 1
        now = time.time()
        if done == n_rows or done % log_every == 0 or (now - last_log) >= 30:
            elapsed = now - start
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = n_rows - done
            eta_sec = (remaining / rate) if rate > 0 else 0.0
            print(
                f"{log_prefix} {done}/{n_rows} rows | "
                f"elapsed {_fmt(elapsed)} | eta {_fmt(eta_sec)} | "
                f"{rate:.0f} rows/s",
                flush=True,
            )
            last_log = now

    return X, feature_names
