"""
kaggle_submit.py — Train a model on ALL training data, then generate a
Kaggle-ready submission CSV for the Quora Question Pairs competition.

Background
----------
The standard experiment pipeline (run_experiment.py) holds out 20 % of
training data for local evaluation.  For a real Kaggle submission we want
to use every labelled pair for training so the model sees as much signal
as possible before predicting on the held-out test set.

Design: the "combined-records stub trick"
-----------------------------------------
EnsembleModel uses a stub matrix (N_total × n_members) where column 0
carries the original row index.  fit() and predict_proba() use those
indices to slice each member's full feature matrix.

This script exploits that mechanism directly:
  1. Build combined records : train_records  +  test_records
  2. model.build_features(all_records)
       → stub (N_total, k),  y_all,  feature_names
       → _member_X_all[i] has shape (N_total, ...)
  3. X_train_stub = stub[:N_train]   # stub col-0 = 0 .. N_train-1
     X_test_stub  = stub[N_train:]   # stub col-0 = N_train .. N_total-1
  4. model.fit(X_train_stub, y[:N_train])
  5. proba = model.predict_proba(X_test_stub)
  6. Write  submissions/<name>/submission.csv

For non-ensemble models the full feature matrix is returned (no stub), so
the same slice X_all[:N_train] / X_all[N_train:] works identically.

Prerequisites
-------------
1. embeddings.zarr        — training question embeddings (from embed_quora.py)
2. test_embeddings.zarr   — test question embeddings   (from embed_quora_test.py)
3. test.csv metadata      — downloaded from the public Kaggle dataset
     (quora/question-pairs-dataset); no authentication required.

Usage
-----
    # Recommended: run through submit.sh on the cluster
    ./submit.sh cpu kaggle_submit.py --model ensemble_mean --name submission_v1

    # Or locally:
    uv run kaggle_submit.py --model ensemble_mean --name submission_v1
    uv run kaggle_submit.py --model ensemble_stack --name submission_stack
    uv run kaggle_submit.py --model catboost       --name catboost_submission

    # Smoke-test with fewer training rows:
    uv run kaggle_submit.py --model ensemble_mean --name smoke_test --max-train-rows 50000

Available --model values mirror run_experiment.py's MODEL_REGISTRY:
    xgboost, catboost, logreg, cosine,
    randforest, randforesttopk,
    gru, gru_v2, gru_v3,
    ensemble_mean, ensemble_mean_weighted, ensemble_stack, ensemble_trees_mean

Output
------
    submissions/<name>/submission.csv      — Kaggle upload file
    submissions/<name>/config.json         — full reproducibility record
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import NamedTuple

from dotenv import load_dotenv
load_dotenv()  # loads environment variables from .env

import numpy as np
import zarr

# ---------------------------------------------------------------------------
# Make experiments/ importable from the repo root
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.join(_SCRIPT_DIR, "experiments")
sys.path.insert(0, _EXPERIMENTS)

import kagglehub
from data import PairRecord, load_pairs

from models import (
    CatBoostModel,
    CosineBaseline,
    EnsembleModel,
    EnsembleClassicalModel,
    GRUModel,
    GRUModelV2,
    GRUModelV3,
    GRUModelV4,
    LSTMModel,
    LogRegModel,
    RandomForestModel,
    RandomForestTopKModel,
    XGBoostModel,
    XGBoostClassicalModel,
)

# ---------------------------------------------------------------------------
# Model registry (mirrors run_experiment.py)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, object] = {
    "xgboost":      XGBoostModel(),
    "catboost":     CatBoostModel(),
    "logreg":       LogRegModel(),
    "cosine":       CosineBaseline(),
    "randforest":   RandomForestModel(),
    "randforesttopk": RandomForestTopKModel(),
    "gru":          GRUModel(),
    "gru_v2":       GRUModelV2(),
    "gru_v3":       GRUModelV3(),
    # ------------------------------------------------------------------ #
    # Ensemble models                                                     #
    # ------------------------------------------------------------------ #
    "ensemble_mean": EnsembleModel(
        members=[XGBoostModel(), CatBoostModel(), GRUModelV3()],
        strategy="mean",
    ),
    "ensemble_mean_weighted": EnsembleModel(
        members=[XGBoostModel(), CatBoostModel(), GRUModelV3()],
        strategy="mean",
        weights=[2.0, 2.0, 1.0],
    ),
    "ensemble_stack": EnsembleModel(
        members=[XGBoostModel(), CatBoostModel(), GRUModelV3()],
        strategy="stacking",
        meta_folds=5,
    ),
    "ensemble_trees_mean": EnsembleModel(
        members=[XGBoostModel(), CatBoostModel(), RandomForestModel()],
        strategy="mean",
    ),
    # ------------------------------------------------------------------ #
    # Classical ensemble models                                           #
    #   Members: XGBoostClassicalModel (tuned) + GRUModelV3 (tuned)      #
    #            + CatBoostModel                                          #
    #   Best hyperparameters are loaded automatically from               #
    #   experiments/tuning/ at instantiation time.                       #
    # ------------------------------------------------------------------ #
    "ensemble_classical_mean": EnsembleClassicalModel(
        strategy="mean",
    ),
    "ensemble_classical_weighted": EnsembleClassicalModel(
        strategy="mean",
        weights=[2.0, 1.0, 2.0],
    ),
    "ensemble_classical_stack": EnsembleClassicalModel(
        strategy="stacking",
        meta_folds=5,
    ),
}

# ---------------------------------------------------------------------------
# Test-data loader
# ---------------------------------------------------------------------------

def _find_test_csv(competition_path: str) -> str:
    """Locate the test CSV (test.csv) inside the competition download."""
    preferred = os.path.join(competition_path, "test.csv")
    if os.path.exists(preferred):
        return preferred

    required = {"test_id", "question1", "question2"}
    for fname in sorted(os.listdir(competition_path)):
        if not fname.endswith(".csv"):
            continue
        full = os.path.join(competition_path, fname)
        try:
            with open(full, newline="", encoding="utf-8") as f:
                headers = set(csv.DictReader(f).fieldnames or [])
            if required.issubset(headers):
                return full
        except Exception:
            pass

    raise FileNotFoundError(
        f"No test CSV with headers {sorted(required)} found in {competition_path}"
    )


class TestPair(NamedTuple):
    """Holds the Kaggle test_id alongside a PairRecord (dummy label=0)."""
    test_id: int
    record:  PairRecord


def load_test_pairs(
    test_zarr_file: str = "test_embeddings.zarr",
    dataset_handle: str = "quora/question-pairs-dataset",
    local_test_csv: str | None = None,
) -> list[TestPair]:
    """
    Build one TestPair per row of the competition test.csv.

    Embeddings are loaded from test_embeddings.zarr (produced by
    embed_quora_test.py).  Lookup is by question text — a Python dict
    `text → zarr position` is built once and used for every row.

    Parameters
    ----------
    test_zarr_file  : path to the zarr store written by embed_quora_test.py
    dataset_handle  : kagglehub dataset handle (public, no auth required)
    local_test_csv  : if provided, use this local file instead of downloading
                      (avoids any network / kaggle-auth dependency at submission time)

    Returns
    -------
    list[TestPair], one entry per test.csv row in file order.
    """
    # --- load test zarr --------------------------------------------------- #
    print(f"[test_data] Loading zarr : {test_zarr_file}", flush=True)
    store = zarr.open(test_zarr_file, mode="r")

    texts_np = store["texts"][:]            # str array, shape (N_unique,)
    emb_np   = store["embeddings"][:].astype(np.float32)   # (N_unique, dim)

    print(f"[test_data] texts shape      : {texts_np.shape}", flush=True)
    print(f"[test_data] embeddings shape : {emb_np.shape}", flush=True)

    # Build lookup dict for O(1) access, then free the texts array.
    text_to_pos: dict[str, int] = {str(t): i for i, t in enumerate(texts_np)}
    del texts_np  # no longer needed; free memory now

    # Compute scalar L2 norms only — do NOT pre-compute a normalised-embedding
    # matrix.  With 2560-dim vectors that matrix would duplicate the already
    # large emb_np array in RAM (~10 GB for the test set) and cause OOM.
    # Cosine similarity is computed on-the-fly in features.py from the stored
    # scalar norms (r.norm1, r.norm2) instead.
    raw_norms = np.linalg.norm(emb_np, axis=1)                     # (N_unique,)

    # --- locate test.csv -------------------------------------------------- #
    # Prefer a locally supplied path (no network needed) over kagglehub.
    if local_test_csv:
        test_csv = os.path.abspath(local_test_csv)
        if not os.path.isfile(test_csv):
            raise FileNotFoundError(
                f"--local-test-csv path does not exist: {test_csv!r}"
            )
        print(f"[test_data] Using local test CSV : {test_csv}", flush=True)
    else:
        print(f"[test_data] Downloading dataset: {dataset_handle}", flush=True)
        comp_path = kagglehub.dataset_download(dataset_handle)
        test_csv  = _find_test_csv(comp_path)
        print(f"[test_data] Using test CSV : {test_csv}", flush=True)

    # --- build TestPair list ---------------------------------------------- #
    print("[test_data] Building test pair records...", flush=True)
    results: list[TestPair] = []
    missing = bad = 0
    start = last_log = time.time()

    with open(test_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                test_id = int(row["test_id"])
                q1      = (row.get("question1") or "").strip()
                q2      = (row.get("question2") or "").strip()
            except (KeyError, ValueError, TypeError):
                bad += 1
                continue

            pos1 = text_to_pos.get(q1)
            pos2 = text_to_pos.get(q2)
            if pos1 is None or pos2 is None:
                missing += 1
                continue

            record = PairRecord(
                qid1      = -1,              # no qid in test data
                qid2      = -1,
                question1 = q1,
                question2 = q2,
                label     = 0,              # dummy — not used during predict
                emb1      = emb_np[pos1],   # numpy view into emb_np
                emb2      = emb_np[pos2],   # numpy view into emb_np
                norm1     = float(raw_norms[pos1]),
                norm2     = float(raw_norms[pos2]),
            )
            results.append(TestPair(test_id=test_id, record=record))

            now = time.time()
            n   = len(results)
            if n % 100_000 == 0 or (now - last_log) >= 30:
                elapsed = now - start
                rate    = n / elapsed if elapsed > 0 else 0
                print(
                    f"[test_data] {n:,} pairs | "
                    f"elapsed {_fmt(elapsed)} | "
                    f"{rate:,.0f} rows/s",
                    flush=True,
                )
                last_log = now

    print(f"[test_data] Total test pairs loaded  : {len(results):,}", flush=True)
    if missing:
        print(f"[test_data] Missing embedding lookups: {missing:,}", flush=True)
    if bad:
        print(f"[test_data] Malformed rows skipped   : {bad:,}", flush=True)

    if not results:
        raise RuntimeError(
            "No test pairs loaded.  "
            "Did you run embed_quora_test.py first to build test_embeddings.zarr?"
        )

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Kaggle submission for Quora Question Pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to train and use for prediction.",
    )
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Submission name (used as output subfolder).",
    )
    parser.add_argument(
        "--train-zarr",
        default=None,
        metavar="PATH",
        help=(
            "Path to training embeddings.zarr. "
            "Defaults to embeddings.zarr in the repo root."
        ),
    )
    parser.add_argument(
        "--test-zarr",
        default=None,
        metavar="PATH",
        help=(
            "Path to test_embeddings.zarr (built by embed_quora_test.py). "
            "Defaults to test_embeddings.zarr in the repo root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Root directory for all submissions. Defaults to submissions/ in the repo root.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="T",
        help=(
            "Decision threshold written into config.json for reference. "
            "Does NOT affect the submission file — Kaggle wants raw probabilities."
        ),
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        metavar="N",
        help="Cap training data at N rows (useful for smoke-tests).",
    )
    parser.add_argument(
        "--local-test-csv",
        default=None,
        metavar="PATH",
        help=(
            "Path to a local test.csv file.  When provided, kagglehub is NOT "
            "contacted — useful on air-gapped cluster nodes where the file was "
            "already downloaded when running embed_quora_test.py."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    repo_root   = _SCRIPT_DIR
    train_zarr  = args.train_zarr  or os.path.join(repo_root, "embeddings.zarr")
    test_zarr   = args.test_zarr   or os.path.join(repo_root, "test_embeddings.zarr")
    output_dir  = args.output_dir  or os.path.join(repo_root, "submissions")
    model       = MODEL_REGISTRY[args.model]
    threshold   = args.threshold if args.threshold is not None else getattr(model, "threshold", 0.5)

    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"[submit] Submission name : {args.name}", flush=True)
    print(f"[submit] Model           : {getattr(model, 'name', type(model).__name__)}", flush=True)
    print(f"[submit] Train zarr      : {train_zarr}", flush=True)
    print(f"[submit] Test zarr       : {test_zarr}", flush=True)
    print(f"[submit] Local test CSV  : {args.local_test_csv or '(none — will use kagglehub)'}", flush=True)
    print(f"[submit] Output dir      : {output_dir}/{args.name}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ------------------------------------------------------------------
    # 1. Load training pairs (all of them — no held-out split)
    # ------------------------------------------------------------------
    print("[submit] Loading training pairs...", flush=True)
    train_records = load_pairs(
        zarr_file      = train_zarr,
        max_rows       = args.max_train_rows,
    )
    N_train = len(train_records)
    print(f"[submit] Training pairs loaded: {N_train:,}", flush=True)

    # ------------------------------------------------------------------
    # 2. Load test pairs
    # ------------------------------------------------------------------
    print("\n[submit] Loading test pairs...", flush=True)
    test_data    = load_test_pairs(
        test_zarr_file = test_zarr,
        local_test_csv = args.local_test_csv,
    )
    test_ids     = [tp.test_id for tp in test_data]
    test_records = [tp.record  for tp in test_data]
    N_test       = len(test_records)
    print(f"[submit] Test pairs loaded: {N_test:,}", flush=True)

    # ------------------------------------------------------------------
    # 3. Build features over combined train + test records
    #
    #    The EnsembleModel stub trick requires that _member_X_all be built
    #    from the FULL combined list.  fit() receives stub[:N_train] whose
    #    col-0 values are exactly 0..N_train-1; predict_proba() receives
    #    stub[N_train:] whose col-0 values are N_train..N_total-1.
    #    Both correctly index into _member_X_all.
    #
    #    For non-ensemble models the full X matrix is returned and the same
    #    [:N_train] / [N_train:] slicing works directly.
    # ------------------------------------------------------------------
    all_records = train_records + test_records
    N_total     = len(all_records)
    print(
        f"\n[submit] Building features for {N_total:,} combined records "
        f"({N_train:,} train + {N_test:,} test)...",
        flush=True,
    )
    # Featurizers (TF-IDF, char-ngrams, topic model, etc.) are fit on ALL
    # records — train + test combined.  The Kaggle test labels are dummy 0s
    # (unknown), so there is no label leakage from including test rows here.
    # Fitting on the full corpus gives better vocabulary coverage.
    # We deliberately do NOT pass train_idx so EnsembleClassicalModel passes
    # None to XGBoostClassicalModel, which then fits its featurizers on every
    # row rather than just the training subset.
    t_feat = time.time()
    X_all, y_all, feature_names = model.build_features(all_records)
    print(
        f"[submit] Feature matrix shape: {X_all.shape}  "
        f"({time.time() - t_feat:.1f}s)",
        flush=True,
    )

    # Slice: training portion uses labelled records, test portion has dummy 0s
    X_train = X_all[:N_train]
    X_test  = X_all[N_train:]
    y_train = y_all[:N_train]

    print(f"[submit] X_train: {X_train.shape}  X_test: {X_test.shape}", flush=True)

    # Release the record lists now that all scalar features are in X_all.
    # This allows Python / NumPy to reclaim the embedding arrays that were
    # kept alive by the PairRecord views:
    #   • emb_arr  (training zarr)  ≈ N_unique_train × dim × 4 B  (~5.5 GB)
    #   • emb_np   (test zarr)      ≈ N_unique_test  × dim × 4 B  (~10  GB)
    # Total: up to ~15 GB freed before the (potentially large) fit() call.
    del all_records, train_records, test_records, test_data

    # ------------------------------------------------------------------
    # 4. Fit on ALL training data (no held-out split)
    # ------------------------------------------------------------------
    print(f"\n[submit] Fitting model...", flush=True)
    t_fit = time.time()
    model.fit(X_train, y_train)
    print(f"[submit] Fit complete in {time.time() - t_fit:.1f}s", flush=True)

    # Training features and labels are no longer needed after fit.
    del X_train, y_train

    # ------------------------------------------------------------------
    # 5. Predict on test set
    # ------------------------------------------------------------------
    print(f"\n[submit] Predicting on {N_test:,} test pairs...", flush=True)
    t_pred = time.time()
    proba  = model.predict_proba(X_test)   # shape (N_test,), values in [0, 1]
    print(f"[submit] Predict complete in {time.time() - t_pred:.1f}s", flush=True)
    del X_test  # test features no longer needed
    print(
        f"[submit] Proba  min={proba.min():.4f}  "
        f"max={proba.max():.4f}  "
        f"mean={proba.mean():.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 6. Write submission.csv
    #
    #    Kaggle expects exactly two columns:
    #        test_id,is_duplicate
    #        0,0.512345
    #        1,0.073421
    #        ...
    #    is_duplicate must be a probability (float), NOT a binary label.
    # ------------------------------------------------------------------
    sub_dir  = os.path.join(output_dir, args.name)
    _ensure_dir(sub_dir)
    sub_path = os.path.join(sub_dir, "submission.csv")

    with open(sub_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_id", "is_duplicate"])
        for tid, p in zip(test_ids, proba):
            writer.writerow([tid, f"{float(p):.6f}"])

    print(f"\n[submit] Wrote {len(test_ids):,} rows → {sub_path}", flush=True)

    # ------------------------------------------------------------------
    # 7. Write config.json for reproducibility
    # ------------------------------------------------------------------
    run_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config  = {
        "run_at":          run_at,
        "submission_name": args.name,
        "model":           getattr(model, "name", type(model).__name__),
        "threshold":       threshold,
        "train_zarr":      train_zarr,
        "test_zarr":       test_zarr,
        "n_train_pairs":   N_train,
        "n_test_pairs":    N_test,
        "n_features":      X_all.shape[1] if X_all.ndim > 1 else 1,
        "feature_names":   feature_names,
        "cli_args": {
            "model":           args.model,
            "name":            args.name,
            "train_zarr":      args.train_zarr,
            "test_zarr":       args.test_zarr,
            "output_dir":      args.output_dir,
            "threshold":       args.threshold,
            "max_train_rows":  args.max_train_rows,
            "local_test_csv":  args.local_test_csv,
        },
    }
    if hasattr(model, "get_config"):
        try:
            config["model_config"] = model.get_config()
        except Exception as exc:
            config["model_config"] = {"error": str(exc)}

    cfg_path = os.path.join(sub_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[submit] Wrote config    → {cfg_path}", flush=True)

    total_time = time.time() - t0
    print(f"\n[submit] Total wall time : {_fmt(total_time)}", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"  Submission ready: {sub_path}", flush=True)
    print(f"  Upload to: https://www.kaggle.com/c/quora-question-pairs/submit", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    run(_parse_args())
