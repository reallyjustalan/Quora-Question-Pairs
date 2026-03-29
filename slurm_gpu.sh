#!/bin/bash
#SBATCH --job-name=ml_gpu
#SBATCH --output=gpu_%j.log
#SBATCH --error=gpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h200-141:1

# Usage: sbatch slurm_gpu.sh <script.py> [args...]
# Example: sbatch slurm_gpu.sh embed_quora.py --batch-size 64

if [ -z "$1" ]; then
    echo "Error: No script provided."
    echo "Usage: sbatch slurm_gpu.sh <script.py> [args...]"
    exit 1
fi

SCRIPT="$1"
shift
SCRIPT_ARGS="$@"

cd ~/Quora-Question-Pairs

echo "Running: uv run python $SCRIPT $SCRIPT_ARGS"
uv run python "$SCRIPT" $SCRIPT_ARGS
