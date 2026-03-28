# Quora Question Pairs

Duplicate question detection on the [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) dataset using dense embeddings and classical ML classifiers.

## Overview

Questions are embedded with [`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B) and stored in a [Zarr](https://zarr.dev/) store (`embeddings.zarr`). Pairwise features derived from the embeddings (cosine similarity, Euclidean/Manhattan distance, lexical overlap, etc.) are fed into downstream classifiers.

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync
```

*Dependencies include `torch` (CUDA 12.6 wheel on Linux), `sentence-transformers`, `catboost`, `zarr`, `dvc[s3]`, and `scikit-learn`. See `pyproject.toml` for the full list.*

---

## Getting the Embeddings (DVC)

The pre-computed `embeddings.zarr` store (~3.7 GB, 18 k files) is tracked with DVC. The DVC remote config files will be shared separately.

```bash
# Place the provided .dvc/config (and credentials) in the repo, then:
dvc pull
```

This pulls `embeddings.zarr` from the configured remote. There is no need to re-run `embed_quora.py` unless you want to regenerate the embeddings.

### Zarr store layout

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
| `embed_quora.py` | Downloads the Quora dataset via `kagglehub`, embeds every unique question with Qwen3-Embedding-4B (SDPA, batches of 128), and writes the result to `embeddings.zarr`. Logs progress every 30 s. |
| `thresholding.py` | Loads `embeddings.zarr`, builds a 22-feature pairwise feature matrix (embedding distances + lexical features), then evaluates a cosine baseline and trains a **Logistic Regression** classifier. Reports accuracy, precision, recall, F1, and the strongest coefficients. |
| `catboost_thresh.py` | Same feature pipeline as `thresholding.py` but adds a **CatBoost** classifier (500 iterations, depth 8). Compares cosine baseline → logistic regression → CatBoost, prints feature importances, and optionally saves misclassified test pairs to `catboost_test_errors.csv`. |

---

## Running on the NUS SOC Cluster (Slurm)

I would recommend you to use this link to figure it out:
https://www.comp.nus.edu.sg/~cs3210/student-guide/accessing/

But the basic sequence of events are:
1. Create a SOC account from https://mysoc.nus.edu.sg/~newacct
2. Turn on 
3. Get the Forticlient VPN and connect to it
4. ssh into the cluster (the password is the password you created for the SOC account)
```bash
ssh <your e0111.. login>@xlogin.comp.nus.edu.sg
```

Once on the cluster, clone the repo and install dependencies:

```bash
git clone https://github.com/reallyjustalan/Quora-Question-Pairs.git
cd Quora-Question-Pairs
uv sync
```

Pull the embeddings with DVC (place the provided config files first):

```bash
dvc pull
```

Submit the embedding job (H200 GPU, 64 GB RAM, 2 h wall time):

```bash
sbatch slurmscript.sh
# Output: embed_<jobid>.log / embed_<jobid>.err
```

To run the classifiers interactively or as a batch job, adapt `slurmscript.sh` — replace `embed_quora.py` with `thresholding.py` or `catboost_thresh.py` as needed. The classifier scripts are CPU-only and do not require a GPU partition.

---

## Project Structure

```
.
├── embed_quora.py          # Embedding pipeline
├── thresholding.py         # Logistic regression classifier
├── catboost_thresh.py      # CatBoost classifier
├── embeddings.zarr.dvc     # DVC pointer to the Zarr store
├── slurmscript.sh          # Slurm job script (embedding)
├── pyproject.toml
└── uv.lock
```
