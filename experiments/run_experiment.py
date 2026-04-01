"""
run_experiment.py — Fixed experiment pipeline.

HOW TO USE
----------
Pass the model and experiment name as CLI arguments:

    cd experiments
    python run_experiment.py --model xgboost --name xgboost_matryoshka_all_features

Available models: xgboost, catboost, logreg, cosine

Optional flags:
    --max-rows   INT    Subsample the dataset (e.g. 50000 for smoke-tests)
    --test-size  FLOAT  Fraction held out for testing (default: 0.20)
    --threshold  FLOAT  Decision threshold (default: model's own, else 0.5)
    --tune              Force hyperparameter tuning when supported by model
    --no-tune           Skip hyperparameter tuning even if model supports it
    --zarr       PATH   Path to embeddings.zarr (default: ../embeddings.zarr)
    --split-file PATH   Path to .npz split file (default: splits/default_split.npz)
    --results-dir PATH  Where to write reports (default: results/)
    --dvc-push          If set, runs `uv run dvc push experiments/results` after reporting
    --dvc-push-target   DVC target path to push (default: experiments/results)

The split indices are saved to splits/default_split.npz on the first run
and reused identically on every subsequent run, guaranteeing that all
experiments are evaluated on exactly the same test rows.

ADDING A NEW MODEL
------------------
1. Create experiments/models/my_model.py  (follow any existing model as template)
2. Import it in experiments/models/__init__.py
3. Add a new entry in the MODEL_REGISTRY dict below.
4. Run with --model <your_key>.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

# Make sure local modules are importable when running from inside experiments/
sys.path.insert(0, os.path.dirname(__file__))

from data import load_pairs
from report import generate_report
from models import CatBoostModel, CosineBaseline, LogRegModel, XGBoostModel, RandomForestModel, RandomForestTopKModel, GRUModel

# ---------------------------------------------------------------------------
# Registry — maps CLI --model name → model instance
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, object] = {
    "xgboost": XGBoostModel(),
    "catboost": CatBoostModel(),
    "logreg":   LogRegModel(),
    "cosine":   CosineBaseline(),
    "randforest": RandomForestModel(),
    "randforesttopk": RandomForestTopKModel(),
    "gru": GRUModel(),
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a duplicate-question-detection experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Which model to use.",
    )
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Unique experiment name (used for the output report directory).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Subsample to N rows (useful for smoke-tests).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        metavar="FRAC",
        help="Fraction of data held out for testing.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="T",
        help="Decision threshold. Defaults to the model's own threshold, or 0.5.",
    )
    tune_group = parser.add_mutually_exclusive_group()
    tune_group.add_argument(
        "--tune",
        dest="tune",
        action="store_true",
        help="Run hyperparameter tuning (if model supports tune()).",
    )
    tune_group.add_argument(
        "--no-tune",
        dest="tune",
        action="store_false",
        help="Skip hyperparameter tuning.",
    )
    parser.set_defaults(tune=None)
    parser.add_argument(
        "--zarr",
        default=None,
        metavar="PATH",
        help="Path to embeddings.zarr. Defaults to ../embeddings.zarr relative to this script.",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        metavar="PATH",
        help="Path to .npz file storing train/test indices. Created on first run.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        metavar="PATH",
        help="Directory where reports are written.",
    )
    parser.add_argument(
        "--dvc-push",
        action="store_true",
        help=(
            "After a successful run, execute `uv run dvc push experiments/results` "
            "from the repository root."
        ),
    )
    parser.add_argument(
        "--dvc-push-target",
        default="experiments/results",
        metavar="PATH",
        help="DVC target path to push when --dvc-push is enabled.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_or_create_split(n: int, split_file: str) -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(split_file):
        data = np.load(split_file)
        train_idx = data["train_idx"]
        test_idx  = data["test_idx"]
        print(
            f"[split] Loaded existing split from {split_file} "
            f"(train={len(train_idx)}, test={len(test_idx)})",
            flush=True,
        )
        if train_idx.max() >= n or test_idx.max() >= n:
            raise RuntimeError(
                f"Saved split has indices up to {max(train_idx.max(), test_idx.max())} "
                f"but dataset only has {n} rows. "
                f"Delete {split_file} to regenerate."
            )
        return train_idx, test_idx

    raise RuntimeError(
        "_load_or_create_split called before labels are available; "
        "use _get_split(n, y, ...) instead."
    )


def _get_split(
    n: int,
    y: np.ndarray,
    split_file: str,
    test_size: float,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx), creating and saving the split if needed."""
    if os.path.exists(split_file):
        return _load_or_create_split(n, split_file)

    print(f"[split] No saved split found — creating and saving to {split_file}", flush=True)
    os.makedirs(os.path.dirname(split_file), exist_ok=True)

    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    np.savez(split_file, train_idx=train_idx, test_idx=test_idx)
    print(
        f"[split] Saved split to {split_file} "
        f"(train={len(train_idx)}, test={len(test_idx)})",
        flush=True,
    )
    return train_idx, test_idx


def _maybe_dvc_push(*, enabled: bool, script_dir: str, target: str) -> None:
    if not enabled:
        return

    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    cmd = ["uv", "run", "dvc", "push", target]
    print(f"\n[dvc] Running: {' '.join(cmd)} (cwd={repo_root})", flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)
    print("[dvc] Push complete.", flush=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # Resolve config from args
    model           = MODEL_REGISTRY[args.model]
    experiment_name = args.name
    max_rows        = args.max_rows
    test_size       = args.test_size
    threshold       = args.threshold if args.threshold is not None else getattr(model, "threshold", 0.5)

    script_dir  = os.path.dirname(__file__)
    zarr_path   = args.zarr        or os.path.join(script_dir, "..", "embeddings.zarr")
    split_file  = args.split_file  or os.path.join(script_dir, "splits", "default_split.npz")
    results_dir = args.results_dir or os.path.join(script_dir, "results")

    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"[run] Experiment  : {experiment_name}", flush=True)
    print(f"[run] Model       : {getattr(model, 'name', type(model).__name__)}", flush=True)
    print(f"[run] Threshold   : {threshold}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    records = load_pairs(zarr_file=zarr_path, max_rows=max_rows)

    # ------------------------------------------------------------------
    # 2. Build features (model owns this step)
    # ------------------------------------------------------------------
    print(f"\n[run] Building features with {getattr(model, 'name', type(model).__name__)}...", flush=True)
    X, y, feature_names = model.build_features(records)
    print(f"[run] Feature matrix: {X.shape}  labels: {y.shape}", flush=True)

    # ------------------------------------------------------------------
    # 3. Fixed train/test split (saved on first run, reused thereafter)
    # ------------------------------------------------------------------
    train_idx, test_idx = _get_split(len(records), y, split_file, test_size)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_records = [records[i] for i in test_idx]

    print(f"[run] Train: {len(train_idx)}  Test: {len(test_idx)}", flush=True)

    # ------------------------------------------------------------------
    # 4. Fit
    # ------------------------------------------------------------------
    print(f"\n[run] Fitting model...", flush=True)
    t_fit = time.time()
    model_supports_tune = hasattr(model, "tune")
    if args.tune is None:
        do_tune = model_supports_tune
    else:
        do_tune = bool(args.tune)

    if do_tune and not model_supports_tune:
        print(
            f"[run] Tuning requested but {getattr(model, 'name', type(model).__name__)} "
            "does not implement tune(); continuing without tuning.",
            flush=True,
        )
        do_tune = False

    if do_tune:
        print("[run] Hyperparameter tuning enabled.", flush=True)
        model.tune(X_train, y_train)
    else:
        print("[run] Hyperparameter tuning skipped.", flush=True)

    model.fit(X_train, y_train)
    print(f"[run] Fit complete in {time.time() - t_fit:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 5. Predict
    # ------------------------------------------------------------------
    proba = model.predict_proba(X_test)

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
    print(f"\n[run] Generating report...", flush=True)
    cli_args_dict = {
        "model":       args.model,
        "name":        args.name,
        "max_rows":    args.max_rows,
        "test_size":   args.test_size,
        "threshold":   args.threshold,
        "tune":        args.tune,
        "zarr":        zarr_path,
        "split_file":  split_file,
        "results_dir": results_dir,
        "dvc_push":    args.dvc_push,
        "dvc_push_target": args.dvc_push_target,
    }
    generate_report(
        experiment_name=experiment_name,
        y_true=y_test,
        proba=proba,
        test_records=test_records,
        feature_names=feature_names,
        model=model,
        threshold=threshold,
        results_dir=results_dir,
        cli_args=cli_args_dict,
    )

    _maybe_dvc_push(
        enabled=args.dvc_push,
        script_dir=script_dir,
        target=args.dvc_push_target,
    )

    print(f"\n[run] Total wall time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    run(parse_args())
