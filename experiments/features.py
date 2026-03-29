"""
features.py — Primitive feature-building functions.

These are pure functions that operate on a single PairRecord and return
a flat dict of named scalar values. Models import whatever primitives
they need and assemble their own feature matrix.

None of these functions do any scaling, normalisation, or model logic.
"""

from __future__ import annotations

import time

import numpy as np

from data import PairRecord


DEFAULT_MATRYOSHKA_DIMS = (128, 256, 512, 1024, 1536, 2048, 2560)


# ---------------------------------------------------------------------------
# Tokenisation helper (shared)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return (text or "").lower().strip().split()


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
