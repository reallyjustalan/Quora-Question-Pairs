#!/bin/bash
#SBATCH --job-name=ml_cpu
#SBATCH --output=cpu_%j.log
#SBATCH --error=cpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

# Usage: sbatch slurm_cpu.sh <script.py> [args...]
# Example: sbatch slurm_cpu.sh catboost_thresh.py --threshold 0.5

if [ -z "$1" ]; then
    echo "Error: No script provided."
    echo "Usage: sbatch slurm_cpu.sh <script.py> [args...]"
    exit 1
fi

SCRIPT="$1"
shift
SCRIPT_ARGS="$@"

cd ~/Quora-Question-Pairs

echo "Running: uv run python $SCRIPT $SCRIPT_ARGS"
uv run python "$SCRIPT" $SCRIPT_ARGS
