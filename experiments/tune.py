"""
tune.py — Dedicated hyperparameter tuning entry point (Optuna).

Why this is a separate script
-----------------------------
Hyperparameter search and final-model evaluation are two different concerns.
Running them in the same process (as `run_experiment.py --tune-optuna` does)
produces a single experiment record whose hyperparameters were chosen inside
that same run, and throws away Optuna's most useful capabilities: persistent
studies, resumability, and parallel trials against a shared storage.

This script runs an Optuna study on the TRAIN split only (it reuses
`splits/default_split.npz` so the held-out test rows are never touched),
persists trials to a SQLite-backed study, and emits:

    results/tuning/<name>/
        best_params.json      # machine-readable, consumed by run_experiment.py
        study.db              # resumable Optuna storage
        trials.csv            # every trial as a row
        tuning_config.json    # audit trail (param space, CV, seeds, timing…)
        plots/                # optimisation_history, param_importances, …

Typical workflow
----------------

    cd experiments

    # 1) Search (can be interrupted + resumed; can be parallelised by running
    #    another process with the same --name against the same SQLite file).
    uv run python tune.py --model xgboost --name xgb_search_v1 --n-trials 50

    # 2) Evaluate the best params on the held-out test set — a regular,
    #    reproducible experiment run that does NO tuning of its own.
    uv run python run_experiment.py \\
        --model xgboost --name xgb_tuned_eval_v1 \\
        --params-file results/tuning/xgb_search_v1/best_params.json

Add a new tunable model
-----------------------
Implement a classmethod on the model:

    @classmethod
    def get_tuning_spec(cls) -> dict:
        return {
            "estimator":   <sklearn-compatible estimator with sane defaults>,
            "param_space": {...},   # same dict schema as OptunaSearchCV
            "scoring":     "f1",
        }

…and register the model class in TUNING_REGISTRY below.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, train_test_split

# Make sure local modules are importable when running from inside experiments/
sys.path.insert(0, os.path.dirname(__file__))

from data import load_pairs
from models import CatBoostModel, XGBoostModel, XGBoostClassicalModel

# ---------------------------------------------------------------------------
# Registry — maps CLI --model name → model class that implements
# get_tuning_spec(). Only models that expose a tuning spec are tunable here.
# ---------------------------------------------------------------------------

TUNING_REGISTRY: dict[str, type] = {
    "xgboost":           XGBoostModel,
    "catboost":          CatBoostModel,
    "xgboost_classical": XGBoostClassicalModel,
}


# ---------------------------------------------------------------------------
# Split helpers (intentionally mirror run_experiment.py so tuning and
# evaluation share the same train indices byte-for-byte).
# ---------------------------------------------------------------------------

def _get_split(
    n: int,
    y: np.ndarray,
    split_file: str,
    test_size: float,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx), creating and saving the split if needed."""
    if os.path.exists(split_file):
        data = np.load(split_file)
        train_idx = data["train_idx"]
        test_idx = data["test_idx"]
        if train_idx.max() >= n or test_idx.max() >= n:
            raise RuntimeError(
                f"Saved split has indices up to {max(train_idx.max(), test_idx.max())} "
                f"but dataset only has {n} rows. Delete {split_file} to regenerate."
            )
        print(
            f"[tune] Loaded existing split from {split_file} "
            f"(train={len(train_idx)}, test={len(test_idx)})",
            flush=True,
        )
        return train_idx, test_idx

    print(f"[tune] No saved split found — creating and saving to {split_file}", flush=True)
    os.makedirs(os.path.dirname(split_file), exist_ok=True)
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y,
    )
    np.savez(split_file, train_idx=train_idx, test_idx=test_idx)
    print(
        f"[tune] Saved split to {split_file} "
        f"(train={len(train_idx)}, test={len(test_idx)})",
        flush=True,
    )
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Optuna objective builder
# ---------------------------------------------------------------------------

def _suggest_params(trial: optuna.Trial, param_space: dict) -> dict:
    """Translate the project's dict-based param spec into Optuna suggest calls."""
    params: dict = {}
    for name, spec in param_space.items():
        ptype = spec.get("type", "float")
        if ptype == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False),
            )
        elif ptype == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], log=spec.get("log", False),
            )
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported param type for '{name}': {ptype}")
    return params


def _fit_with_eval_set_fallback(model, X_train, y_train, X_val, y_val) -> None:
    """
    Some estimators (XGBoost/CatBoost with early stopping enabled) require an
    eval_set at fit time. Fall back transparently when that happens.
    """
    try:
        model.fit(X_train, y_train)
    except ValueError as exc:
        msg = str(exc)
        if "validation dataset" not in msg and "early stopping" not in msg:
            raise
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


def build_objective(
    estimator,
    param_space: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv: int,
    scoring: str | None,
    random_state: int,
):
    """Return an Optuna objective that does stratified CV with per-fold pruning."""
    scorer = get_scorer(scoring) if scoring else None

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, param_space)
        model = clone(estimator)
        model.set_params(**params)

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        fold_scores: list[float] = []
        for fold_idx, (tr, va) in enumerate(skf.split(X, y)):
            X_tr, X_va = X[tr], X[va]
            y_tr, y_va = y[tr], y[va]
            _fit_with_eval_set_fallback(model, X_tr, y_tr, X_va, y_va)
            score = scorer(model, X_va, y_va) if scorer else model.score(X_va, y_va)
            fold_scores.append(float(score))

            # Report running average to enable median-pruning mid-CV.
            trial.report(float(np.mean(fold_scores)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Optuna hyperparameter search for a registered model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=list(TUNING_REGISTRY.keys()),
        help="Which model to tune.",
    )
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Tuning run name. Also used as the Optuna study_name and output folder.",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--timeout", type=int, default=None, help="Wall-clock timeout (seconds).")
    parser.add_argument("--cv", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument(
        "--scoring",
        default=None,
        help="Scoring metric override (default: the model's own default from get_tuning_spec()).",
    )
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--max-rows", type=int, default=None, metavar="N",
                        help="Subsample to N rows (smoke tests).")
    parser.add_argument("--test-size", type=float, default=0.20,
                        help="Used only if the split file has to be created.")
    parser.add_argument("--zarr", default=None, metavar="PATH",
                        help="Path to embeddings.zarr. Defaults to ../embeddings.zarr.")
    parser.add_argument("--cross-encoder-zarr", default=None, metavar="PATH",
                        help="Path to cross_encoder_scores.zarr.")
    parser.add_argument("--split-file", default=None, metavar="PATH",
                        help="Path to saved split .npz. Defaults to splits/default_split.npz.")
    parser.add_argument("--results-dir", default="experiments/results", metavar="PATH",
                        help="Results root. Output goes under <results-dir>/tuning/<name>/.")
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

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume the study if its SQLite storage already exists (default).",
    )
    resume_group.add_argument(
        "--fresh",
        dest="resume",
        action="store_false",
        help="Refuse to run if the study's storage already exists.",
    )
    parser.set_defaults(resume=True)

    return parser.parse_args()


def _maybe_dvc_push(*, enabled: bool, script_dir: str, target: str) -> None:
    if not enabled:
        return

    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    cmd = ["uv", "run", "dvc", "push", target]
    print(f"\n[dvc] Running: {' '.join(cmd)} (cwd={repo_root})", flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)
    print("[dvc] Push complete.", flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    script_dir = os.path.dirname(__file__)
    zarr_path      = args.zarr               or os.path.join(script_dir, "..", "embeddings.zarr")
    cross_enc_path = args.cross_encoder_zarr or os.path.join(script_dir, "..", "cross_encoder_scores.zarr")
    split_file     = args.split_file         or os.path.join(script_dir, "splits", "default_split.npz")
    results_dir    = args.results_dir        or os.path.join(script_dir, "results")

    tuning_dir = os.path.join(results_dir, "tuning", args.name)
    plots_dir  = os.path.join(tuning_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Resolve tuning spec from the model class
    model_cls = TUNING_REGISTRY[args.model]
    if not hasattr(model_cls, "get_tuning_spec"):
        raise RuntimeError(
            f"{model_cls.__name__} does not implement get_tuning_spec(); "
            "cannot tune this model with tune.py."
        )
    spec         = model_cls.get_tuning_spec()
    estimator    = spec["estimator"]
    param_space  = spec["param_space"]
    scoring      = args.scoring or spec.get("scoring")

    # Build a model instance just to reuse its feature pipeline.
    feature_builder = model_cls()
    if hasattr(feature_builder, "cfg") and isinstance(feature_builder.cfg, dict) \
            and "cross_encoder_zarr" in feature_builder.cfg:
        feature_builder.cfg["cross_encoder_zarr"] = cross_enc_path

    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"[tune] Tuning run : {args.name}",                 flush=True)
    print(f"[tune] Model      : {model_cls.__name__}",        flush=True)
    print(f"[tune] N trials   : {args.n_trials}",             flush=True)
    print(f"[tune] Scoring    : {scoring}",                   flush=True)
    print(f"[tune] CV folds   : {args.cv}",                   flush=True)
    print(f"[tune] Output dir : {tuning_dir}",                flush=True)
    print(f"{'='*60}\n", flush=True)

    # ------------------------------------------------------------------
    # 1. Load data + build features
    # ------------------------------------------------------------------
    records = load_pairs(zarr_file=zarr_path, max_rows=args.max_rows)

    print(f"\n[tune] Building features with {model_cls.__name__}...", flush=True)
    X, y, feature_names = feature_builder.build_features(records)
    print(f"[tune] Feature matrix: {X.shape}  labels: {y.shape}", flush=True)

    # ------------------------------------------------------------------
    # 2. Use the SAME split as run_experiment.py — tune on train only.
    # ------------------------------------------------------------------
    train_idx, test_idx = _get_split(
        len(records), y, split_file, args.test_size, args.random_state,
    )
    X_train, y_train = X[train_idx], y[train_idx]
    print(f"[tune] Tuning on {len(train_idx)} train rows "
          f"(held-out test rows {len(test_idx)} are untouched).", flush=True)

    # ------------------------------------------------------------------
    # 3. Create / resume a persistent Optuna study.
    # ------------------------------------------------------------------
    storage_path = os.path.join(tuning_dir, "study.db")
    storage_url  = f"sqlite:///{os.path.abspath(storage_path)}"

    if not args.resume and os.path.exists(storage_path):
        raise RuntimeError(
            f"Storage already exists at {storage_path}. "
            "Re-run with --resume (default) to continue, or delete the file to start fresh."
        )

    _study_exists = os.path.exists(storage_path)
    if _study_exists:
        # Peek at the trial count before loading so we can report it.
        _peek = optuna.load_study(
            study_name=args.name,
            storage=storage_url,
        )
        _n_done = len([t for t in _peek.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])
        _n_total = len(_peek.trials)
        print(
            f"[tune] RESUMING existing study '{args.name}' "
            f"({_n_done} complete / {_n_total} total trials so far). "
            f"Storage: {storage_path}",
            flush=True,
        )
        del _peek
    else:
        print(
            f"[tune] Starting FRESH study '{args.name}'. "
            f"Storage will be created at: {storage_path}",
            flush=True,
        )

    study = optuna.create_study(
        study_name=args.name,
        storage=storage_url,
        sampler=TPESampler(seed=args.random_state),
        pruner=MedianPruner(),
        direction="maximize",
        load_if_exists=True,
    )

    objective = build_objective(
        estimator=estimator,
        param_space=param_space,
        X=X_train,
        y=y_train,
        cv=args.cv,
        scoring=scoring,
        random_state=args.random_state,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    print(f"\n[tune] Best trial #{best_trial.number}: score={best_trial.value:.6f}", flush=True)
    print(f"[tune] Best params: {json.dumps(best_trial.params, indent=2)}", flush=True)

    # ------------------------------------------------------------------
    # 4. Persist artifacts
    # ------------------------------------------------------------------
    best_params_path = os.path.join(tuning_dir, "best_params.json")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model":       args.model,
                "tuning_name": args.name,
                "best_score":  float(best_trial.value),
                "scoring":     scoring,
                "method":      "OptunaSearchCV",
                "best_params": best_trial.params,
            },
            f,
            indent=2,
        )
    print(f"[tune] Wrote {best_params_path}", flush=True)

    trials_path = os.path.join(tuning_dir, "trials.csv")
    with open(trials_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_number", "value", "state", "duration_s", "params"])
        for t in study.trials:
            dur = ""
            if t.datetime_complete and t.datetime_start:
                dur = f"{(t.datetime_complete - t.datetime_start).total_seconds():.3f}"
            writer.writerow([
                t.number,
                "" if t.value is None else f"{t.value:.6f}",
                t.state.name,
                dur,
                json.dumps(t.params, sort_keys=True),
            ])
    print(f"[tune] Wrote {trials_path}", flush=True)

    config_path = os.path.join(tuning_dir, "tuning_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tuning_name":         args.name,
                "model":               args.model,
                "model_class":         model_cls.__name__,
                "n_trials_requested":  args.n_trials,
                "n_trials_in_study":   len(study.trials),
                "n_complete_trials":   len([t for t in study.trials
                                            if t.state == optuna.trial.TrialState.COMPLETE]),
                "n_pruned_trials":     len([t for t in study.trials
                                            if t.state == optuna.trial.TrialState.PRUNED]),
                "cv":                  args.cv,
                "scoring":             scoring,
                "random_state":        args.random_state,
                "timeout":             args.timeout,
                "param_space":         param_space,
                "storage_url":         storage_url,
                "split_file":          split_file,
                "n_train_rows":        int(len(train_idx)),
                "n_test_rows_excluded": int(len(test_idx)),
                "cli_args":            vars(args),
                "wall_time_seconds":   time.time() - t0,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"[tune] Wrote {config_path}", flush=True)

    # ------------------------------------------------------------------
    # 5. Optuna visualisations (best-effort — don't fail the run).
    # ------------------------------------------------------------------
    try:
        optuna.visualization.plot_optimization_history(study).write_html(
            os.path.join(plots_dir, "optimization_history.html"))
        optuna.visualization.plot_param_importances(study).write_html(
            os.path.join(plots_dir, "param_importances.html"))
        optuna.visualization.plot_parallel_coordinate(study).write_html(
            os.path.join(plots_dir, "parallel_coordinates.html"))
        for var in param_space.keys():
            optuna.visualization.plot_slice(study, params=[var]).write_html(
                os.path.join(plots_dir, f"slice_{var}.html"))
        print(f"[tune] Wrote plots to {plots_dir}", flush=True)
    except Exception as exc:
        print(f"[tune] Could not write Optuna visualisations: {exc}", flush=True)

    _maybe_dvc_push(
        enabled=args.dvc_push,
        script_dir=script_dir,
        target=args.dvc_push_target,
    )

    print(f"\n[tune] Total wall time: {time.time() - t0:.1f}s", flush=True)
    print("\n[tune] Next step — evaluate the tuned model on the held-out test set:")
    print(
        f"    python run_experiment.py --model {args.model} "
        f"--name {args.name}_eval --params-file {best_params_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
