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
| `embed_quora.py` | Downloads the Quora dataset via `kagglehub`, embeds every unique question with Qwen3-Embedding-4B (SDPA, batches of 128), and writes the result to `embeddings.zarr`. Logs progress every 30 s. |
| `catboost_thresh.py` | Loads `embeddings.zarr`, builds a 22-feature pairwise feature matrix (embedding distances + lexical features), then compares a cosine baseline and a **Logistic Regression** against a **CatBoost** classifier (500 iterations, depth 8). Prints feature importances and optionally saves misclassified test pairs to `catboost_test_errors.csv`. |

---

## Running on the NUS SOC Cluster (Slurm)
*Follow this to do what I did!*

> **What is Slurm?**
> Slurm is a job scheduler used by HPC (high-performance computing) clusters. Because many users share the same machines, you don't run scripts directly — you submit a *job* describing what resources you need (GPUs, RAM, time limit) and Slurm queues it up and runs it when those resources are free. `slurmscript.sh` is that job description: the `#SBATCH` lines at the top are directives to Slurm (not regular shell comments), and the last line is the actual command to execute.
>
> Useful Slurm commands:
> | Command | What it does |
> |---------|-------------|
> | `sbatch slurmscript.sh` | Submit a job to the queue |
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
uv sync
```

Pull the embeddings with DVC (place the provided config files first - you may copy files over using `cp`):

```bash
uv run dvc pull
```

Submit the embedding job (H200 GPU, 64 GB RAM, 2 h wall time):

```bash
sbatch slurmscript.sh
# Output: embed_<jobid>.log / embed_<jobid>.err
```

To run the classifiers interactively or as a batch job, adapt `slurmscript.sh` — replace `embed_quora.py` with `catboost_thresh.py` as needed. The classifier script is CPU-only and does not require a GPU partition.

---

## Project Structure

```
.
├── embed_quora.py          # Embedding pipeline
├── catboost_thresh.py      # CatBoost classifier (+ logistic regression baseline)
├── embeddings.zarr.dvc     # DVC pointer to the Zarr store
├── slurmscript.sh          # Slurm job script (embedding)
├── pyproject.toml
└── uv.lock
```
