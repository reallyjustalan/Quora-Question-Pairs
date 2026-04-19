# Quora Question Pairs

Duplicate question detection on the [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) dataset using dense embeddings and classical ML classifiers.

## Overview

Questions are embedded with [`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B) and stored in a [Zarr](https://zarr.dev/) store (`embeddings.zarr`). Pairwise features derived from the embeddings (cosine similarity, Euclidean/Manhattan distance, lexical overlap, etc.) are fed into downstream classifiers.

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

> **What is uv?**
> `uv` is a modern Python package and project manager (written in Rust) that replaces the `pip` + `venv` workflow. It is dramatically faster than `pip`, automatically creates and manages virtual environments for you, and pins dependencies in a lockfile (`uv.lock`) so everyone gets exactly the same versions. You never have to manually run `python -m venv` or worry about environment activation — `uv run` handles it all in one step.

Installing uv is easy and painless:

```bash
# Download and install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then, when you run a script, uv will automatically download and set up the necessary virtual environments. Never worry about venvs ever again!

You run a script with 

```bash
uv run <script_name.py>
```

*Dependencies include `torch` (CUDA 12.6 wheel on Linux), `sentence-transformers`, `catboost`, `zarr`, `dvc[s3]`, and `scikit-learn`. See `pyproject.toml` for the full list.*

---

## Getting the Embeddings (DVC)
    
> **What is DVC?**
> DVC (Data Version Control) is like Git, but for large data files. Git isn't suited for storing gigabytes of binary data, so instead DVC stores a small pointer file (`embeddings.zarr.dvc`) in the repo and keeps the actual data on a separate remote storage (e.g. S3). Running `dvc pull` downloads the real data from that remote. This keeps the repo lightweight while still letting everyone reproduce the exact same data.

The pre-computed `embeddings.zarr` store (~3.7 GB, 18 k files) is tracked with DVC. The DVC remote config files will be shared separately. If you want to just get the embeddings, the same applies to you too!

All you have to do is:

```bash
# Place the provided .dvc/config (and credentials) in the repo, then:
uv run dvc pull
```

This pulls `embeddings.zarr` from the configured remote. There is no need to re-run `embed_quora.py` unless you want to regenerate the embeddings.
Once the pull finishes, `embeddings.zarr/` will be present and ready to use.

`embeddings.zarr` is necessary for the other scripts that rely on this to function.

### Zarr store layout

> **What is Zarr?**
> Zarr is a format for storing large numerical arrays on disk (or in the cloud). Think of it like `.npy`, but designed for data that won't fit comfortably in memory. Instead of one monolithic file, Zarr splits the array into many small compressed chunks, meaning you can load only the rows you need rather than the entire dataset at once. This makes random access fast and efficient, which matters when you have hundreds of thousands of embeddings.

| Array | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `ids` | `(N,)` | int64 | Unique question IDs, sorted ascending |
| `texts` | `(N,)` | str | Question text, aligned with `ids` |
| `embeddings` | `(N, 2560)` | float32 | Qwen3-Embedding-4B vectors |

Look up a question by ID with `np.searchsorted(store["ids"], qid)`.

---

## Scripts

| Script | What it does |
|--------|-------------|
| `embed_quora.py` | Downloads the Quora **training** dataset, embeds every unique question with Qwen3-Embedding-4B (SDPA, batches of 128), and writes the result to `embeddings.zarr`. |
| `embed_quora_test.py` | Downloads the Kaggle **competition test** data, embeds every unique question, and writes to `test_embeddings.zarr`. Run this before `kaggle_submit.py`. Requires Kaggle credentials. |
| `kaggle_submit.py` | Trains the chosen model on **all** training pairs, predicts on the competition test set, and writes `submissions/<name>/submission.csv` ready to upload to Kaggle. |
| `experiments/run_experiment.py` | Plug-and-play experiment runner (local evaluation with train/test split) — see the [Experiment Harness](#experiment-harness) section below. |
| `experiments/tune.py` | Dedicated Optuna hyperparameter search; runs a persistent, resumable study on the **train split only** — see [Hyperparameter Tuning](#hyperparameter-tuning) below. |

---

## Running on the NUS SOC Cluster (Slurm)
*Follow this to do what I did!*

> [!WARNING]
> **DO NOT run heavy scripts (like embedding or training) directly on the login node.**
> Running intensive processes on the login node can slow down the system for all users and may result in your process being terminated by administrators. Always use `submit.sh` or `sbatch` to queue your tasks.

> **What is Slurm?**
> Slurm is a job scheduler used by HPC (high-performance computing) clusters. Because many users share the same machines, you don't run scripts directly — you submit a *job* describing what resources you need (GPUs, RAM, time limit) and Slurm queues it up and runs it when those resources are free. Slurm scripts (like `slurm_gpu.sh` or `slurm_cpu.sh`) are those job descriptions: the `#SBATCH` lines at the top are directives to Slurm (not regular shell comments), and the last line is the actual command to execute.
>
> Useful Slurm commands:
> | Command | What it does |
> |---------|-------------|
> | `./submit.sh gpu <script.py> [args...]` | Submit a GPU job (recommended — auto-names logs) |
> | `./submit.sh cpu <script.py> [args...]` | Submit a CPU-only job (recommended — auto-names logs) |
> | `squeue -u $USER` | Check the status of your queued/running jobs, $USER being your e1111.. username |
> | `scancel <jobid>` | Cancel a job |
> | `sinfo` | See available partitions and node status |

I mainly figured things out using this link:
https://www.comp.nus.edu.sg/~cs3210/student-guide/accessing/

The basic sequence of events are:
1. Create a SOC account from https://mysoc.nus.edu.sg/~newacct
2. Turn on The SOC Compute Cluster here: https://mysoc.nus.edu.sg/~myacct/services.cgi
3. Get the Forticlient VPN and connect to it
4. ssh into the cluster (the password is the password you created for the SOC account)
```bash
ssh <your e0111.. login>@xlogin.comp.nus.edu.sg
```

Once on the cluster, download uv, clone the repo and install dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/reallyjustalan/Quora-Question-Pairs.git
cd Quora-Question-Pairs
```
Notice how we didn't even have to bother installing dependencies? uv will do that automatically at runtime!

Pull the embeddings with DVC (place the provided config files first - you may copy files over using `cp`):

```bash
uv run dvc pull
```

There are two Slurm scripts (GPU and CPU-only) and a convenience wrapper `submit.sh` that handles log naming and folder organisation for you. **Use `submit.sh` — it is the recommended way to submit jobs.**

### Using `submit.sh` (recommended)

`submit.sh` automatically:
- derives the job name from the script filename (e.g. `embed_quora` from `embed_quora.py`)
- routes stdout/stderr to `logs/<script>_<jobid>.log` / `logs/<script>_<jobid>.err`
- creates the `logs/` directory if it doesn't exist

```bash
# GPU job (H200 GPU, 64 GB RAM, 8 CPUs, 2 h wall time)
./submit.sh gpu embed_quora.py

# CPU-only job (64 GB RAM, 16 CPUs, 2 h wall time)
./submit.sh cpu catboost_thresh.py

# Extra arguments are forwarded to the script
./submit.sh gpu embed_quora.py --batch-size 64
./submit.sh cpu catboost_thresh.py --threshold 0.5
```

Log files will appear under `logs/` and be named after the script, e.g.:
```
logs/embed_quora_12345.log
logs/embed_quora_12345.err
```

### Calling sbatch directly (advanced)

If you prefer to call `sbatch` manually, `%x` in the output path resolves to the job name, so you'll want to set `--job-name` yourself:

```bash
sbatch --job-name=embed_quora --output=logs/embed_quora_%j.log --error=logs/embed_quora_%j.err slurm_gpu.sh embed_quora.py
```

---

## Experiment Harness

All new model experiments live in `experiments/`. The key idea is that **model logic, feature building, and the evaluation pipeline are completely separate files** — you snap them together.

### How it works

```
embeddings.zarr  ──►  data.py  ──►  model.build_features()  ──►  run_experiment.py  ──►  report.py
                      (loads)        (each model picks its          (fixed pipeline:        (metrics.txt
                                      own feature set from           split → fit →           errors.csv
                                      features.py primitives)        predict)                config.json
                                                                                            all_experiments.csv)
```

- **`data.py`** — loads the zarr store + CSV into a list of `PairRecord` objects. No model logic here.
- **`features.py`** — pure feature functions (`embedding_features`, `lexical_features`, `all_features`, `matryoshka_embedding_features`, `matryoshka_all_features`). Each returns a `dict[str, float]` per pair. Models import whatever they need.
- **`models/<name>.py`** — each model owns its feature selection and any internal pre-processing (e.g. `StandardScaler` in `logreg_model.py`). Every model exposes three methods: `build_features()`, `fit()`, and `predict_proba()`. Optionally `feature_importances()` and `get_config()`.
- **`run_experiment.py`** — the fixed pipeline entry point; model and experiment name are passed as CLI arguments.
- **`report.py`** — takes predictions and writes the full report (see below).

### Fixed split — fair comparison

The first run saves `experiments/splits/default_split.npz`. Every subsequent run loads those exact train/test indices. All models are therefore evaluated on **identical test rows**, making comparisons meaningful.

### Running an experiment

On the cluster, submit via `submit.sh` **(recommended)**:

```bash
# CPU job — scripts resolve all paths from the repo root automatically
./submit.sh cpu experiments/run_experiment.py --model catboost --name catboost_v1

# With optional DVC push after the run
./submit.sh cpu experiments/run_experiment.py --model catboost --name catboost_v1 --dvc-push

# Quick smoke-test with a subset of rows
./submit.sh cpu experiments/run_experiment.py --model catboost --name catboost_smoke --max-rows 50000
```

If you want to run **locally** (outside the cluster):

```bash
uv run experiments/run_experiment.py --model catboost --name catboost_v1

```

Available `--model` values: `xgboost`, `catboost`, `logreg`, `cosine`

For quick smoke-tests, add `--max-rows 50000`. Other useful flags: `--threshold`, `--test-size`, `--zarr`, `--split-file`, `--results-dir`. See `experiments/README.md` for the full flag reference.

### Publishing cluster-generated results

If you run experiments on the cluster and want your laptop to be the only machine
with DVC/Git write credentials, use `publish_results.sh` from your local checkout.
It pulls `experiments/results/` over SSH with `rsync`, then runs `dvc add`,
`dvc push`, and `git push` locally.

```bash
# store these locally in .env (already gitignored)
cat > .env <<'EOF'
CLUSTER_HOST=e1234567@xcna1.comp.nus.edu.sg
CLUSTER_REPO=/home/e1234567/final_project
EOF

# then publish from your laptop
./publish_results.sh "Publish cluster catboost run"

# or override for a one-off invocation
./publish_results.sh \
  --host e1234567@xcna1.comp.nus.edu.sg \
  --repo /home/e1234567/final_project \
  "Publish cluster catboost run"
```

By default the sync step uses `rsync --delete`, so your local
`experiments/results/` becomes an exact mirror of the cluster copy. This is
useful because old experiment folders that were removed or overwritten on the
cluster do not linger locally and get re-published by accident. If you want to
keep local-only files, pass `--keep-local-extra`.

### Report output

Each run produces:

```
experiments/results/
├── all_experiments.csv               ← one row per run (accuracy, precision, recall, F1, TP/FP/TN/FN)
└── <experiment_name>/
    ├── metrics.txt                   ← full metrics block + classification report
    ├── errors.csv                    ← every FP and FN with question text and predicted probability
    ├── config.json                   ← full reproducibility record: CLI args, model class,
    │                                    matryoshka_dims, hyperparams, and ordered feature list
    └── feature_importance.txt        ← ranked feature importances (if the model supports it)
```

### Adding a new model

1. Create `experiments/models/my_model.py` — copy any existing model as a template.
2. Implement `build_features(records)`, `fit(X_train, y_train)`, and `predict_proba(X_test)`.
3. Optionally implement:
   - `feature_importances() → dict[str, float]` for automatic importance reporting.
   - `get_config() → dict` so `config.json` captures your model's full hyperparameters and feature list.
4. Register it in `experiments/models/__init__.py` and add an entry to `MODEL_REGISTRY` in `run_experiment.py`.

### Adding a new feature set

Add a function to `features.py` that takes a `PairRecord` and returns `dict[str, float]`, then reference it from your model's `_feature_fn`.

---

## Hyperparameter Tuning

Hyperparameter search and final-model evaluation are **deliberately kept as separate pipeline stages**.  Running them in the same process conflates search and evaluation, discards the Optuna study when the process exits, and gives no way to resume or parallelise trials.  The dedicated entry point `experiments/tune.py` handles search only; `run_experiment.py` handles evaluation only.

### Two-stage workflow

```
tune.py  ──►  results/tuning/<name>/best_params.json  ──►  run_experiment.py --params-file …
  (search on train split only)                               (evaluate on held-out test split,
   persistent + resumable SQLite study)                       no tuning inside this run)
```

#### Stage 1 — search (on the cluster via `submit.sh`)

```bash
# CPU job — tune XGBoost with 50 Optuna trials
./submit.sh cpu experiments/tune.py --model xgboost --name xgb_search_v1 --n-trials 50

# Tune CatBoost with a wall-clock timeout instead of a fixed trial count
./submit.sh cpu experiments/tune.py --model catboost --name cat_search_v1 --timeout 3600

# Smoke-test on a subset of rows
./submit.sh cpu experiments/tune.py --model xgboost --name xgb_smoke --n-trials 10 --max-rows 50000
```

Re-submitting with the **same `--name`** automatically resumes the existing SQLite study (add `--fresh` to refuse resumption).  You can also run multiple jobs against the same `--name` in parallel — Optuna's SQLite storage coordinates trials across workers.

If you want to run **locally** (outside the cluster):

```bash
uv run experiments/tune.py --model xgboost --name xgb_search_v1 --n-trials 50
```

#### Stage 2 — evaluate (load the best params, run on held-out test split)

```bash
# On the cluster
./submit.sh cpu experiments/run_experiment.py \
    --model xgboost --name xgb_tuned_v1 \
    --params-file experiments/results/tuning/xgb_search_v1/best_params.json

# Locally
uv run experiments/run_experiment.py \
    --model xgboost --name xgb_tuned_v1 \
    --params-file experiments/results/tuning/xgb_search_v1/best_params.json
```

#### Output of `tune.py`

```
experiments/results/tuning/
└── <name>/
    ├── best_params.json      ← feed into run_experiment.py via --params-file
    ├── study.db              ← resumable Optuna SQLite storage
    ├── trials.csv            ← every trial as a row
    ├── tuning_config.json    ← audit trail (param space, CV folds, seeds, timing…)
    └── plots/
        ├── optimization_history.html
        ├── param_importances.html
        ├── parallel_coordinates.html
        └── slice_<param>.html
```

### `tune.py` CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` / `-m` | *(required)* | Tunable model: `xgboost`, `catboost`, `xgboost_classical` |
| `--name` / `-n`  | *(required)* | Tuning run name (= Optuna `study_name` + output folder) |
| `--n-trials`     | 50 | Number of Optuna trials to run this invocation |
| `--timeout`      | None | Wall-clock timeout in seconds |
| `--cv`           | 5 | Stratified CV folds |
| `--scoring`      | model default (f1) | Override scoring metric |
| `--random-state` | 42 | Seed for TPE sampler + CV shuffle |
| `--max-rows`     | None | Subsample N rows (smoke-tests) |
| `--zarr`         | `../embeddings.zarr` | Path to embeddings store |
| `--split-file`   | `splits/default_split.npz` | Same split as `run_experiment.py` |
| `--results-dir`  | `results/` | Output goes under `<results-dir>/tuning/<name>/` |
| `--resume` / `--fresh` | `--resume` | Resume an existing study, or refuse to |

### Making a model tunable

Implement `get_tuning_spec()` on the model class and register it in `TUNING_REGISTRY` inside `tune.py`:

```python
@classmethod
def get_tuning_spec(cls) -> dict:
    return {
        "estimator":   MyClassifier(**_DEFAULTS),  # fresh sklearn-compatible estimator
        "param_space": {                            # Optuna-style dict schema
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "max_depth":     {"type": "int",   "low": 3,    "high": 10},
        },
        "scoring": "f1",
    }

def apply_tuned_params(self, best_params, *, source=None, cv_score=None, method="external"):
    self._params.update(best_params)
    self._model.set_params(**best_params)
    self._tuning_info = {
        "enabled": True, "method": method,
        "best_cv_score": float(cv_score) if cv_score is not None else None,
        "best_params": dict(best_params), "source": source,
    }
```

The existing `XGBoostModel` and `CatBoostModel` are working references.

---

## Generating a Kaggle Submission

The standard experiment pipeline (`run_experiment.py`) holds out 20 % of training data for local evaluation.  For an actual Kaggle submission you want to train on **every labelled pair** and predict on the competition's unlabelled test set.  Two new scripts handle this end-to-end.

### Step-by-step

#### 1 — Set up Kaggle credentials

The competition test data is downloaded via `kagglehub`, which needs your Kaggle API token.

```bash
# Option A: file  (recommended)
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Option B: environment variables
export KAGGLE_USERNAME=YOUR_USERNAME
export KAGGLE_KEY=YOUR_API_KEY
```

Your API key is at https://www.kaggle.com/settings → "Create New Token".

#### 2 — Embed the test questions  (`embed_quora_test.py`)

The competition `test.csv` contains ~2.3 M question pairs whose questions are **not** in the training zarr store.  This script downloads `test.csv`, extracts every unique question text, embeds them with Qwen3-Embedding-4B, and saves to `test_embeddings.zarr`.

```bash
# GPU job (strongly recommended — ~1M+ unique questions to embed)
./submit.sh gpu embed_quora_test.py

# With a custom batch size or output path:
./submit.sh gpu embed_quora_test.py --batch-size 64 --output test_embeddings.zarr
```

The output zarr store layout:

| Array | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `texts` | `(N,)` | str | Unique question texts, **sorted alphabetically** |
| `embeddings` | `(N, 2560)` | float32 | Qwen3-Embedding-4B vectors |

Look up a question by text:
```python
import zarr, numpy as np
store    = zarr.open("test_embeddings.zarr", mode="r")
texts    = store["texts"][:]
text2pos = {t: i for i, t in enumerate(texts)}
pos      = text2pos["What is the capital of France?"]
emb      = store["embeddings"][pos]
```

#### 3 — Train and submit  (`kaggle_submit.py`)

Trains the chosen model on **all** training pairs (no held-out split), predicts on the full test set, and writes a `submissions/<name>/submission.csv` that you upload directly to Kaggle.

```bash
# Best ensemble — unweighted mean of XGBoost + CatBoost + GRU v3
./submit.sh cpu kaggle_submit.py --model ensemble_mean --name ensemble_mean_v1

# Stacking ensemble (OOF LogReg meta-learner)
./submit.sh cpu kaggle_submit.py --model ensemble_stack --name ensemble_stack_v1

# Weighted ensemble (up-weights tree models 2× vs GRU)
./submit.sh cpu kaggle_submit.py --model ensemble_mean_weighted --name ensemble_weighted_v1

# Single model (fast sanity-check)
./submit.sh cpu kaggle_submit.py --model catboost --name catboost_v1

# Smoke-test with fewer training rows
./submit.sh cpu kaggle_submit.py --model ensemble_mean --name smoke --max-train-rows 50000
```

Available `--model` values: `xgboost`, `catboost`, `logreg`, `cosine`,
`randforest`, `randforesttopk`, `gru`, `gru_v2`, `gru_v3`,
`ensemble_mean`, `ensemble_mean_weighted`, `ensemble_stack`, `ensemble_trees_mean`

#### Output

```
submissions/
└── <name>/
    ├── submission.csv   ← upload this to Kaggle (test_id, is_duplicate probability)
    └── config.json      ← full reproducibility record (model, zarr paths, row counts, …)
```

> **Note on `is_duplicate`:** Kaggle scores this competition with **log-loss**, which requires *probabilities* (floats 0–1), not binary predictions.  `kaggle_submit.py` always writes raw `predict_proba()` output — never hard labels.

#### How the "combined-records stub trick" works

`EnsembleModel` stores each base model's full feature matrix internally and uses stub row-indices to route samples.  `kaggle_submit.py` exploits this by concatenating `train_records + test_records` before calling `build_features`, so the stub indices 0…N_train−1 are naturally the training rows and N_train…N_total−1 are the test rows.  `fit()` sees only the training slice; `predict_proba()` sees only the test slice — no changes to any model code required.

---

## Project Structure

```
.
├── embed_quora.py              # Step 1a: embed training questions → embeddings.zarr
├── embed_quora_test.py         # Step 1b: embed competition test questions → test_embeddings.zarr
├── kaggle_submit.py            # Step 3: train on all data → submissions/<name>/submission.csv
├── experiments/
│   ├── run_experiment.py       # Step 2a: entry point for evaluation experiments
│   ├── tune.py                 # Step 2b: Optuna hyperparameter search (separate stage)
│   ├── data.py                 # Shared loader: zarr + CSV → list[PairRecord]
│   ├── features.py             # Primitive feature functions (embedding, lexical)
│   ├── report.py               # Metrics printer + results writer
│   ├── models/
│   │   ├── ensemble_model.py   # Ensemble / stacking wrapper
│   │   ├── catboost_model.py
│   │   ├── xgboost_model.py
│   │   ├── gru_model_v3.py
│   │   ├── logreg_model.py
│   │   ├── cosine_baseline.py
│   │   └── ...
│   ├── splits/                 # Auto-created; holds the fixed train/test split
│   └── results/                # Auto-created; one subfolder per experiment run
├── embeddings.zarr.dvc         # DVC pointer to the training zarr store
├── test_embeddings.zarr        # Auto-created by embed_quora_test.py (gitignored)
├── submissions/                # Auto-created by kaggle_submit.py (gitignored)
├── slurm_gpu.sh                # Slurm job script for GPU tasks (e.g. embedding)
├── slurm_cpu.sh                # Slurm job script for CPU-only tasks (e.g. classifiers)
├── submit.sh                   # Convenience wrapper: auto-names logs and routes to logs/
├── logs/                       # Created automatically; contains <script>_<jobid>.log/.err
├── pyproject.toml
└── uv.lock
```
