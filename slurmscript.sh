#!/bin/bash
#SBATCH --job-name=quora_embed
#SBATCH --output=embed_%j.log
#SBATCH --error=embed_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --gpus=h200-141:4

cd ~/Quora-Question-Pairs

uv run python embed_quora.py