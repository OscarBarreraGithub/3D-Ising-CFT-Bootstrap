#!/bin/bash
#SBATCH --job-name=consolidate_cache
#SBATCH --account=randall_lab
#SBATCH --partition=shared
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/consolidate_cache_%j.log
#SBATCH --error=logs/consolidate_cache_%j.log

# Consolidate 520K individual .npy block cache files into a single .npz archive.
# This is a one-time operation. After it completes, Stage A/B jobs will load
# the consolidated archive (~10-30s) instead of 520K individual files (~60+ min).

set -euo pipefail
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate ising_bootstrap

PROJECT_DIR="/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap"
cd "$PROJECT_DIR"

mkdir -p logs

echo "=== Cache Consolidation ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

python jobs/consolidate_cache.py

echo "=== Done: $(date) ==="
