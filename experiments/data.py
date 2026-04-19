"""
data.py — Shared data loader.

Loads the zarr embedding store and the Quora question-pairs CSV,
then builds a list of PairRecord namedtuples (one per usable row).

This module is intentionally free of any model or feature logic.
"""

import csv
import os
import time
from typing import NamedTuple

import kagglehub
import numpy as np
import zarr

ZARR_FILE = "embeddings.zarr"
DATASET_HANDLE = "quora/question-pairs-dataset"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class PairRecord(NamedTuple):
    """All raw data for one question pair."""
    qid1: int
    qid2: int
    question1: str
    question2: str
    label: int          # 0 or 1
    emb1: np.ndarray    # raw (un-normalised) embedding for q1
    emb2: np.ndarray    # raw (un-normalised) embedding for q2
    norm1: float        # L2 norm of emb1
    norm2: float        # L2 norm of emb2
    # NOTE: norm_emb1 / norm_emb2 were removed to avoid keeping a full
    # (N_unique × dim) normalised-embedding matrix alive alongside the raw
    # one. With 2560-dim vectors that duplicated matrix cost ~5-10 GB of RAM.
    # Callers that need the unit vector should compute it inline:
    #   u = r.emb1 / max(r.norm1, 1e-12)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _find_pairs_csv(dataset_path: str) -> str:
    required = {"qid1", "qid2", "question1", "question2", "is_duplicate"}
    candidates = []

    preferred = os.path.join(dataset_path, "questions.csv")
    if os.path.exists(preferred):
        candidates.append(preferred)

    for fname in os.listdir(dataset_path):
        if fname.endswith(".csv"):
            full = os.path.join(dataset_path, fname)
            if full not in candidates:
                candidates.append(full)

    for path in candidates:
        try:
            with open(path, newline="", encoding="utf-8") as f:
                headers = set(csv.DictReader(f).fieldnames or [])
            if required.issubset(headers):
                return path
        except Exception:
            pass

    raise FileNotFoundError(
        f"No CSV with headers {sorted(required)} found in {dataset_path}"
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_pairs(
    zarr_file: str = ZARR_FILE,
    dataset_handle: str = DATASET_HANDLE,
    max_rows: int | None = None,
) -> list[PairRecord]:
    """
    Load the zarr embedding store and the pairs CSV; return a list of
    PairRecord objects in CSV row order (skipping rows with missing qids).

    Parameters
    ----------
    zarr_file       : path to the zarr store produced by embed_quora.py
    dataset_handle  : kagglehub dataset identifier
    max_rows        : cap on how many pairs to load (None = all)

    Returns
    -------
    list[PairRecord]
    """

    # --- zarr ---
    print(f"[data] Loading zarr store: {zarr_file}", flush=True)
    store = zarr.open(zarr_file, mode="r")

    ids_arr = store["ids"][:].astype(np.int64)
    emb_arr = store["embeddings"][:].astype(np.float32)

    print(f"[data] ids shape        : {ids_arr.shape}", flush=True)
    print(f"[data] embeddings shape : {emb_arr.shape}", flush=True)

    qid_to_pos = {int(qid): i for i, qid in enumerate(ids_arr)}

    raw_norms = np.linalg.norm(emb_arr, axis=1)
    # Do NOT pre-compute a full normalised-embedding matrix here.
    # With 2560-dim vectors it would duplicate the 5+ GB emb_arr in RAM
    # and keep it alive for the entire run.  Cosine similarity is computed
    # on-the-fly in features.py using the stored scalar norms instead.

    # --- CSV ---
    print(f"[data] Downloading dataset: {dataset_handle}", flush=True)
    dataset_path = kagglehub.dataset_download(dataset_handle)
    pairs_csv = _find_pairs_csv(dataset_path)
    print(f"[data] Using pairs CSV: {pairs_csv}", flush=True)

    # --- build records ---
    print("[data] Building pair records...", flush=True)
    records: list[PairRecord] = []
    missing = bad = 0
    start = last_log = time.time()

    with open(pairs_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_rows is not None and len(records) >= max_rows:
                break

            try:
                qid1 = int(row["qid1"])
                qid2 = int(row["qid2"])
                q1 = row["question1"] or ""
                q2 = row["question2"] or ""
                label = int(row["is_duplicate"])
            except (KeyError, ValueError, TypeError):
                bad += 1
                continue

            pos1 = qid_to_pos.get(qid1)
            pos2 = qid_to_pos.get(qid2)
            if pos1 is None or pos2 is None:
                missing += 1
                continue

            records.append(PairRecord(
                qid1=qid1, qid2=qid2,
                question1=q1, question2=q2,
                label=label,
                emb1=emb_arr[pos1], emb2=emb_arr[pos2],
                norm1=float(raw_norms[pos1]), norm2=float(raw_norms[pos2]),
            ))

            now = time.time()
            n = len(records)
            if n % 50_000 == 0 or (now - last_log) >= 30:
                elapsed = now - start
                rate = n / elapsed if elapsed > 0 else 0
                print(
                    f"[data] {n} pairs | "
                    f"elapsed {_format_duration(elapsed)} | "
                    f"{rate:.0f} rows/s",
                    flush=True,
                )
                last_log = now

    print(f"[data] Total pairs loaded : {len(records)}", flush=True)
    print(f"[data] Missing qids skipped: {missing}", flush=True)
    print(f"[data] Bad rows skipped    : {bad}", flush=True)

    if not records:
        raise RuntimeError("No usable pairs found.")

    return records
