"""
tune_random.py — Dedicated RandomizedSearchCV hyperparameter tuning entry point.

This mirrors experiments/tune.py but uses sklearn's RandomizedSearchCV
instead of Optuna. It tunes on the TRAIN split only (reusing
splits/default_split.npz), writes best_params.json for run_experiment.py,
and optionally pushes results with DVC.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time

import numpy as np
from run_experiment import _maybe_dvc_push
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split

# Make sure local modules are importable when running from inside experiments/
sys.path.insert(0, os.path.dirname(__file__))

from data import load_pairs
from models import CatBoostModel, XGBoostModel, XGBoostClassicalModel
from tune import _get_split


TUNING_REGISTRY: dict[str, type] = {
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
    "xgboost_classical": XGBoostClassicalModel,
}


def _to_randomizedsearch_space(param_space: dict, random_state: int) -> dict:
    """Convert project param-space schema to RandomizedSearchCV distributions."""
    del random_state  # Reserved for future deterministic list generation.
    out: dict[str, object] = {}

    for name, spec in param_space.items():
        ptype = spec.get("type", "float")

        if ptype == "categorical":
            out[name] = list(spec["choices"])
            continue

        if ptype == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            if spec.get("log", False):
                # Use a discrete log-spaced candidate list for integer params.
                n_points = min(40, max(8, high - low + 1))
                vals = np.unique(np.rint(np.logspace(np.log10(low), np.log10(high), n_points)).astype(int))
                vals = vals[(vals >= low) & (vals <= high)]
                out[name] = vals.tolist()
            else:
                out[name] = randint(low, high + 1)
            continue

        if ptype == "float":
            low = float(spec["low"])
            high = float(spec["high"])
            if spec.get("log", False):
                out[name] = loguniform(low, high)
            else:
                out[name] = uniform(loc=low, scale=high - low)
            continue

        raise ValueError(f"Unsupported param type for '{name}': {ptype}")

    return out


def _prepare_estimator_for_cv(estimator):
    """Disable early stopping defaults that require eval_set during CV folds."""
    params = estimator.get_params()
    if "early_stopping_rounds" in params and params["early_stopping_rounds"] is not None:
        estimator.set_params(early_stopping_rounds=None)
        print(
            "[tune-random] Disabled estimator early_stopping_rounds for CV compatibility.",
            flush=True,
        )
    return estimator


def _write_plotly_scatter_html(
    *,
    x: list,
    y: list[float],
    title: str,
    x_title: str,
    y_title: str,
    output_path: str,
) -> None:
        """Write a Plotly scatter HTML file using the Python plotly package."""
        try:
                go = importlib.import_module("plotly.graph_objects")
        except ModuleNotFoundError as exc:
                raise RuntimeError(
                        "Plotly is required for visualisations. Install it with `uv add plotly`."
                ) from exc

        fig = go.Figure(
                data=[
                        go.Scatter(
                                x=x,
                                y=y,
                                mode="markers",
                                marker={"size": 8, "opacity": 0.8},
                        )
                ]
        )
        fig.update_layout(
                title=title,
                xaxis_title=x_title,
                yaxis_title=y_title,
        )
        fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)


def _write_visualisations(results: dict, param_space: dict, plots_dir: str) -> None:
    """Write objective-vs-trial and objective-vs-parameter plots as HTML."""
    scores = [float(s) for s in results["mean_test_score"]]
    trial_ids = list(range(len(scores)))

    _write_plotly_scatter_html(
        x=trial_ids,
        y=scores,
        title="Optimization History (Objective vs Trial)",
        x_title="n_trials",
        y_title="objective value",
        output_path=os.path.join(plots_dir, "optimization_history.html"),
    )

    for var in param_space.keys():
        x_vals = [params.get(var) for params in results["params"]]
        _write_plotly_scatter_html(
            x=x_vals,
            y=scores,
            title=f"Objective vs {var}",
            x_title=var,
            y_title="objective value",
            output_path=os.path.join(plots_dir, f"slice_{var}.html"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a RandomizedSearchCV hyperparameter search for a registered model.",
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
        help="Tuning run name. Output goes under <results-dir>/tuning/<name>/.",
    )
    parser.add_argument("--n-iter", type=int, default=50, help="Number of random search iterations.")
    parser.add_argument("--cv", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument(
        "--scoring",
        default=None,
        help="Scoring metric override (default: model default from get_tuning_spec()).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel workers for RandomizedSearchCV.")

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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = os.path.dirname(__file__)
    zarr_path = args.zarr or os.path.join(script_dir, "..", "embeddings.zarr")
    cross_enc_path = args.cross_encoder_zarr or os.path.join(script_dir, "..", "cross_encoder_scores.zarr")
    split_file = args.split_file or os.path.join(script_dir, "splits", "default_split.npz")
    results_dir = args.results_dir or os.path.join(script_dir, "results")

    tuning_dir = os.path.join(results_dir, "tuning", args.name)
    plots_dir = os.path.join(tuning_dir, "plots")
    os.makedirs(tuning_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    model_cls = TUNING_REGISTRY[args.model]
    if not hasattr(model_cls, "get_tuning_spec"):
        raise RuntimeError(
            f"{model_cls.__name__} does not implement get_tuning_spec(); "
            "cannot tune this model with tune_random.py."
        )

    spec = model_cls.get_tuning_spec()
    estimator = _prepare_estimator_for_cv(spec["estimator"])
    param_space = spec["param_space"]
    scoring = args.scoring or spec.get("scoring")

    feature_builder = model_cls()
    if hasattr(feature_builder, "cfg") and isinstance(feature_builder.cfg, dict) \
            and "cross_encoder_zarr" in feature_builder.cfg:
        feature_builder.cfg["cross_encoder_zarr"] = cross_enc_path

    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"[tune-random] Tuning run : {args.name}", flush=True)
    print(f"[tune-random] Model      : {model_cls.__name__}", flush=True)
    print(f"[tune-random] N iter     : {args.n_iter}", flush=True)
    print(f"[tune-random] Scoring    : {scoring}", flush=True)
    print(f"[tune-random] CV folds   : {args.cv}", flush=True)
    print(f"[tune-random] Output dir : {tuning_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    records = load_pairs(zarr_file=zarr_path, max_rows=args.max_rows)

    print(f"\n[tune-random] Building features with {model_cls.__name__}...", flush=True)
    X, y, _ = feature_builder.build_features(records)
    print(f"[tune-random] Feature matrix: {X.shape}  labels: {y.shape}", flush=True)

    train_idx, test_idx = _get_split(
        len(records), y, split_file, args.test_size, args.random_state,
    )
    X_train, y_train = X[train_idx], y[train_idx]
    print(
        f"[tune-random] Tuning on {len(train_idx)} train rows "
        f"(held-out test rows {len(test_idx)} are untouched).",
        flush=True,
    )

    param_distributions = _to_randomizedsearch_space(param_space, args.random_state)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring=scoring,
        n_jobs=args.n_jobs,
        cv=cv,
        verbose=2,
        random_state=args.random_state,
        refit=True,
        return_train_score=False,
    )

    print("[tune-random] Starting RandomizedSearchCV...", flush=True)
    t_search = time.time()
    search.fit(X_train, y_train)
    print(f"[tune-random] Search complete in {time.time() - t_search:.1f}s", flush=True)

    best_params = search.best_params_
    best_score = float(search.best_score_)

    print(f"\n[tune-random] Best score: {best_score:.6f}", flush=True)
    print(f"[tune-random] Best params: {json.dumps(best_params, indent=2)}", flush=True)

    best_params_path = os.path.join(tuning_dir, "best_params.json")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "tuning_name": args.name,
                "best_score": best_score,
                "scoring": scoring,
                "method": "RandomizedSearchCV",
                "best_params": best_params,
            },
            f,
            indent=2,
        )
    print(f"[tune-random] Wrote {best_params_path}", flush=True)

    trials_path = os.path.join(tuning_dir, "trials.csv")
    results = search.cv_results_
    with open(trials_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "mean_fit_time",
            "mean_score_time",
            "params",
        ])
        for i in range(len(results["params"])):
            writer.writerow([
                int(results["rank_test_score"][i]),
                float(results["mean_test_score"][i]),
                float(results["std_test_score"][i]),
                float(results["mean_fit_time"][i]),
                float(results["mean_score_time"][i]),
                json.dumps(results["params"][i], sort_keys=True),
            ])
    print(f"[tune-random] Wrote {trials_path}", flush=True)

    config_path = os.path.join(tuning_dir, "tuning_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tuning_name": args.name,
                "model": args.model,
                "model_class": model_cls.__name__,
                "n_iter_requested": args.n_iter,
                "n_candidates_evaluated": len(results["params"]),
                "cv": args.cv,
                "scoring": scoring,
                "random_state": args.random_state,
                "n_jobs": args.n_jobs,
                "param_space": param_space,
                "split_file": split_file,
                "n_train_rows": int(len(train_idx)),
                "n_test_rows_excluded": int(len(test_idx)),
                "cli_args": vars(args),
                "wall_time_seconds": time.time() - t0,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"[tune-random] Wrote {config_path}", flush=True)

    # ------------------------------------------------------------------
    # 5. Visualisations (objective vs n_trials + objective vs each param).
    # ------------------------------------------------------------------
    try:
        _write_visualisations(results, param_space, plots_dir)
        print(f"[tune-random] Wrote plots to {plots_dir}", flush=True)
    except Exception as exc:
        print(f"[tune-random] Could not write visualisations: {exc}", flush=True)

    _maybe_dvc_push(
        enabled=args.dvc_push,
        script_dir=script_dir,
        target=args.dvc_push_target,
    )

    print(f"\n[tune-random] Total wall time: {time.time() - t0:.1f}s", flush=True)
    print("\n[tune-random] Next step - evaluate the tuned model on the held-out test set:")
    print(
        f"    python run_experiment.py --model {args.model} "
        f"--name {args.name}_eval --params-file {best_params_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
