"""
report.py — Experiment reporting.

Given predictions and metadata from a completed experiment, this module:
  1. Prints a structured summary to stdout.
  2. Writes results/  <experiment_name>/metrics.txt
  3. Writes results/<experiment_name>/errors.csv  (FP + FN rows)
  4. Writes results/<experiment_name>/feature_importance.txt  (if available)
  5. Appends one summary row to results/all_experiments.csv

Call generate_report(…) from run_experiment.py after predicting.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data import PairRecord

RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "cm":        confusion_matrix(y_true, y_pred),
    }


def _extract_optimal_hyperparameters(model) -> dict | None:
        """
        Return tuned hyperparameters if the model exposes them, else None.

        Expected shape in model.get_config():
            {
                "tuning": {
                    "enabled": true,
                    "best_params": {...}
                }
            }
        """
        if not hasattr(model, "get_config"):
                return None

        try:
                config = model.get_config()
        except Exception:
                return None

        tuning = config.get("tuning") if isinstance(config, dict) else None
        if not isinstance(tuning, dict) or not tuning.get("enabled", False):
                return None

        best_params = tuning.get("best_params")
        return best_params if isinstance(best_params, dict) else None


def _format_metrics_block(
    experiment_name: str,
    model_name: str,
    feature_names: list[str],
    optimal_hyperparams: dict | None,
    threshold: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    run_at: str,
) -> str:
    m = _metrics_dict(y_true, y_pred)
    tn, fp, fn, tp = m["cm"].ravel()

    lines = [
        "=" * 60,
        f"Experiment : {experiment_name}",
        f"Model      : {model_name}",
        f"Run at     : {run_at}",
        f"Threshold  : {threshold:.4f}",
        (
            f"Hyperparameters: {json.dumps(optimal_hyperparams, sort_keys=True)}"
            if optimal_hyperparams
            else "Hyperparameters: None"
        ),
        f"Features   : {len(feature_names)}  ({', '.join(feature_names)})",
        f"Test size  : {len(y_true)}",
        "-" * 60,
        f"Accuracy   : {m['accuracy']:.4f}",
        f"Precision  : {m['precision']:.4f}",
        f"Recall     : {m['recall']:.4f}",
        f"F1 score   : {m['f1']:.4f}",
        "-" * 60,
        "Confusion matrix (rows=true, cols=pred):",
        f"  TN={tn}  FP={fp}",
        f"  FN={fn}  TP={tp}",
        "-" * 60,
        "Classification report:",
        classification_report(y_true, y_pred, digits=4, zero_division=0),
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    experiment_name: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    test_records: list[PairRecord],
    feature_names: list[str],
    model,
    threshold: float = 0.5,
    results_dir: str = RESULTS_DIR,
    cli_args: dict | None = None,
) -> dict:
    """
    Generate and persist a full experiment report.

    Parameters
    ----------
    experiment_name : unique name for this run (used as folder name)
    y_true          : ground-truth labels for the test set (int32, shape N)
    proba           : positive-class probabilities from model.predict_proba()
    test_records    : PairRecord list for the test split (same order as y_true)
    feature_names   : list of feature column names
    model           : the fitted model object — used to get .name and optionally
                      .feature_importances() and .get_config()
    threshold       : decision threshold (default 0.5; CosineBaseline uses 0.76)
    results_dir     : root folder for all results (default "results/")
    cli_args        : optional dict of CLI arguments passed to run_experiment.py;
                      included verbatim in config.json for full reproducibility.

    Returns
    -------
    dict with keys accuracy, precision, recall, f1
    """
    run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    y_pred = (proba >= threshold).astype(np.int32)
    m = _metrics_dict(y_true, y_pred)

    model_name = getattr(model, "name", type(model).__name__)
    optimal_hyperparams = _extract_optimal_hyperparameters(model)

    # ------------------------------------------------------------------
    # 1. Print to stdout
    # ------------------------------------------------------------------
    block = _format_metrics_block(
        experiment_name, model_name, feature_names,
        optimal_hyperparams,
        threshold, y_true, y_pred, run_at,
    )
    print(block, flush=True)

    # ------------------------------------------------------------------
    # 2. Write metrics.txt
    # ------------------------------------------------------------------
    exp_dir = os.path.join(results_dir, experiment_name)
    _ensure_dir(exp_dir)

    metrics_path = os.path.join(exp_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(block + "\n")
    print(f"[report] Wrote {metrics_path}", flush=True)

    # ------------------------------------------------------------------
    # 3. Write errors.csv
    # ------------------------------------------------------------------
    errors_path = os.path.join(exp_dir, "errors.csv")
    n_errors = 0
    with open(errors_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "qid1", "qid2", "question1", "question2",
            "true_label", "pred_label", "pred_prob", "error_type",
        ])
        for i, rec in enumerate(test_records):
            true_label = int(y_true[i])
            pred_label = int(y_pred[i])
            if true_label == pred_label:
                continue
            error_type = "FN" if true_label == 1 else "FP"
            writer.writerow([
                rec.qid1, rec.qid2,
                rec.question1, rec.question2,
                true_label, pred_label,
                f"{proba[i]:.6f}",
                error_type,
            ])
            n_errors += 1

    print(f"[report] Wrote {errors_path}  ({n_errors} errors)", flush=True)

    # ------------------------------------------------------------------
    # 4. Write config.json  (always)
    # ------------------------------------------------------------------
    config: dict = {
        "experiment_name": experiment_name,
        "run_at": run_at,
        "model": model_name,
        "threshold": threshold,
        "optimal_hyperparameters": optimal_hyperparams,
        "test_size": int(len(y_true)),
        "cli_args": cli_args or {},
    }
    # Merge model-level config if the model exposes get_config()
    if hasattr(model, "get_config"):
        try:
            config["model_config"] = model.get_config()
        except Exception as exc:
            config["model_config"] = {"error": str(exc)}
    else:
        # Fallback: at least record the feature names
        config["model_config"] = {
            "n_features": len(feature_names),
            "feature_names": feature_names,
        }

    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[report] Wrote {config_path}", flush=True)

    # ------------------------------------------------------------------
    # 5. Write feature_importance.txt  (optional)
    # ------------------------------------------------------------------
    if hasattr(model, "feature_importances"):
        try:
            importances: dict[str, float] = model.feature_importances()
            sorted_imp = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
            imp_path = os.path.join(exp_dir, "feature_importance.txt")
            with open(imp_path, "w", encoding="utf-8") as f:
                f.write(f"Feature importances — {experiment_name}\n")
                f.write(f"{'Feature':<25}  Importance\n")
                f.write("-" * 40 + "\n")
                for name, imp in sorted_imp:
                    f.write(f"{name:<25}  {imp:.6f}\n")
            print(f"[report] Wrote {imp_path}", flush=True)
        except Exception as exc:
            print(f"[report] Could not write feature importances: {exc}", flush=True)

    # ------------------------------------------------------------------
    # 6. Append to all_experiments.csv
    # ------------------------------------------------------------------
    summary_path = os.path.join(results_dir, "all_experiments.csv")
    write_header = not os.path.exists(summary_path)

    tn, fp, fn, tp = m["cm"].ravel()

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_at", "experiment_name", "model", "n_features",
                "threshold", "test_size",
                "accuracy", "precision", "recall", "f1",
                "TP", "FP", "TN", "FN",
            ])
        writer.writerow([
            run_at, experiment_name, model_name, len(feature_names),
            f"{threshold:.4f}", len(y_true),
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            int(tp), int(fp), int(tn), int(fn),
        ])

    print(f"[report] Appended row to {summary_path}", flush=True)

    return {
        "accuracy":  m["accuracy"],
        "precision": m["precision"],
        "recall":    m["recall"],
        "f1":        m["f1"],
    }
