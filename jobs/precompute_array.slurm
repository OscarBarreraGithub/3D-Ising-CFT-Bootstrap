#!/bin/bash
#SBATCH --job-name=ising_precompute
#SBATCH --account=iaifi_lab
#SBATCH --partition=shared
#SBATCH --array=0-9
#SBATCH --time=18:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/precompute_%A_%a.log
#SBATCH --error=logs/precompute_%A_%a.log

# Sharded precompute: splits ~520K operators across 10 parallel jobs.
# Each job handles every 10th operator (~52K each, high-spin ops take ~60s each).
# Observed throughput: ~5,800 operators/hour per shard (8 workers).
# Resumable: skips already-cached operators if resubmitted.

set -e

echo "=== Ising Bootstrap: Block Precomputation (shard ${SLURM_ARRAY_TASK_ID}/10) ==="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo ""

# --- Prevent thread oversubscription ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# --- Conda activation ---
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    source "$HOME/.conda/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda" >&2
    exit 1
fi
conda activate ising_bootstrap

echo "Python: $(which python)"
echo ""

# --- Run precomputation for this shard ---
python -m ising_bootstrap.scans.stage_a \
    --precompute-only \
    --workers ${SLURM_CPUS_PER_TASK} \
    --shard-id ${SLURM_ARRAY_TASK_ID} \
    --num-shards 10 \
    --verbose

echo ""
echo "Shard ${SLURM_ARRAY_TASK_ID} complete: $(date)"
echo "Cache files: $(ls data/cached_blocks/ext_*.npy 2>/dev/null | wc -l)"
