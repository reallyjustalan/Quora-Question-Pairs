"""
tune_deep.py — Optuna hyperparameter tuning for LSTM and GRU v3.

HOW TO USE
----------
    cd experiments

    # Tune LSTM (default 20 trials)
    uv run python tune_deep.py --model lstm --name lstm_tuning

    # Tune GRU v3
    uv run python tune_deep.py --model gru_v3 --name gru_v3_tuning

    # More trials
    uv run python tune_deep.py --model lstm --name lstm_tuning --n-trials 30

    # Subsample for smoke-test
    uv run python tune_deep.py --model lstm --name lstm_tuning --max-rows 50000

On the cluster (recommended):
    ./submit.sh gpu experiments/tune_deep.py --model lstm --name lstm_tuning
    ./submit.sh gpu experiments/tune_deep.py --model gru_v3 --name gru_v3_tuning

OUTPUT
------
Each run saves to experiments/results/<name>/:
    best_params.json   — best hyperparameters found
    tuning_summary.txt — all trials ranked by val F1

HOW IT WORKS
------------
Each Optuna trial:
  1. Samples a set of hyperparameters from the search space
  2. Builds a fresh model with those hyperparameters
  3. Calls model.fit() which trains with the built-in val split (val_frac=0.05)
     and returns the val F1 on the best checkpoint
  4. Reports val F1 back to Optuna
  5. Optuna uses all previous results to suggest smarter next parameters (TPE)

No k-fold CV — each trial is one training run (~1 min on H200).
20 trials ≈ 20 min on H200.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, os.path.dirname(__file__))

from data import load_pairs
from models import GRUModelV3, LSTMModel

# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

# Parameters to tune and their ranges.
# These are shared between LSTM and GRU v3 since they have the same architecture.
SEARCH_SPACE = {
    "hidden_size":   {"type": "categorical", "choices": [128, 256, 512]},
    "num_layers":    {"type": "int",         "low": 1,    "high": 3},
    "dropout":       {"type": "float",       "low": 0.1,  "high": 0.5},
    "lr":            {"type": "float",       "low": 1e-4, "high": 1e-2, "log": True},
    "weight_decay":  {"type": "float",       "low": 1e-5, "high": 1e-2, "log": True},
    "mlp_hidden":    {"type": "categorical", "choices": [256, 512]},
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for LSTM and GRU v3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=["lstm", "gru_v3"],
        help="Which model to tune.",
    )
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Experiment name (used for output directory).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Subsample to N rows (for smoke-tests).",
    )
    parser.add_argument(
        "--zarr",
        default=None,
        metavar="PATH",
        help="Path to embeddings.zarr. Defaults to ../embeddings.zarr.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        metavar="PATH",
        help="Directory where results are written.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _sample_params(trial: optuna.Trial) -> dict:
    """Sample hyperparameters from the search space for one trial."""
    params = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"],
                log=spec.get("log", False),
            )
    return params


def make_objective(model_key: str, X: np.ndarray, y: np.ndarray):
    """Return an Optuna objective function closed over the data."""

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial)

        print(f"\n[trial {trial.number}] Params: {params}", flush=True)
        t0 = time.time()

        if model_key == "lstm":
            model = LSTMModel(**params)
        else:
            model = GRUModelV3(**params)

        val_f1 = model.fit(X, y)

        elapsed = time.time() - t0
        print(
            f"[trial {trial.number}] Val F1={val_f1:.4f}  ({elapsed:.1f}s)",
            flush=True,
        )
        return val_f1

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    script_dir  = os.path.dirname(__file__)
    zarr_path   = args.zarr or os.path.join(script_dir, "..", "embeddings.zarr")
    results_dir = args.results_dir or os.path.join(script_dir, "results")
    out_dir     = os.path.join(results_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"[tune] Model    : {args.model}", flush=True)
    print(f"[tune] Trials   : {args.n_trials}", flush=True)
    print(f"[tune] Output   : {out_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ------------------------------------------------------------------
    # 1. Load data and build features once (reused across all trials)
    # ------------------------------------------------------------------
    records = load_pairs(zarr_file=zarr_path, max_rows=args.max_rows)

    # Use a dummy model just to build the feature matrix
    if args.model == "lstm":
        dummy = LSTMModel()
    else:
        dummy = GRUModelV3()

    print("[tune] Building features...", flush=True)
    X, y, _ = dummy.build_features(records)
    print(f"[tune] Feature matrix: {X.shape}", flush=True)

    # ------------------------------------------------------------------
    # 2. Run Optuna study
    # ------------------------------------------------------------------
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
    )

    study.optimize(
        make_objective(args.model, X, y),
        n_trials=args.n_trials,
    )

    # ------------------------------------------------------------------
    # 3. Save results
    # ------------------------------------------------------------------
    best_params = study.best_trial.params
    best_f1     = study.best_trial.value

    print(f"\n{'='*60}", flush=True)
    print(f"[tune] Best Val F1 : {best_f1:.4f}", flush=True)
    print(f"[tune] Best Params : {best_params}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # best_params.json
    best_params_path = os.path.join(out_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump({"model": args.model, "best_val_f1": best_f1, **best_params}, f, indent=2)
    print(f"[tune] Saved best params to {best_params_path}", flush=True)

    # tuning_summary.txt — all trials ranked by val F1
    summary_path = os.path.join(out_dir, "tuning_summary.txt")
    trials_sorted = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else -1,
        reverse=True,
    )
    with open(summary_path, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Total trials: {len(study.trials)}\n")
        f.write(f"Best Val F1: {best_f1:.4f}\n\n")
        f.write(f"{'Rank':<6} {'Trial':<8} {'Val F1':<10} Params\n")
        f.write("-" * 80 + "\n")
        for rank, t in enumerate(trials_sorted, 1):
            f.write(
                f"{rank:<6} {t.number:<8} "
                f"{t.value:<10.4f} {t.params}\n"
            )
    print(f"[tune] Saved tuning summary to {summary_path}", flush=True)

    # ------------------------------------------------------------------
    # 4. Save Optuna visualisations (HTML, open in browser)
    # ------------------------------------------------------------------
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        import optuna.visualization as vis

        vis.plot_optimization_history(study).write_html(
            os.path.join(plots_dir, "optimization_history.html")
        )
        vis.plot_param_importances(study).write_html(
            os.path.join(plots_dir, "param_importances.html")
        )
        vis.plot_parallel_coordinate(study).write_html(
            os.path.join(plots_dir, "parallel_coordinates.html")
        )
        for param in SEARCH_SPACE.keys():
            vis.plot_slice(study, params=[param]).write_html(
                os.path.join(plots_dir, f"slice_{param}.html")
            )
        print(f"[tune] Saved visualisations to {plots_dir}", flush=True)
    except Exception as e:
        print(f"[tune] Visualisations failed (non-fatal): {e}", flush=True)


if __name__ == "__main__":
    run(parse_args())
