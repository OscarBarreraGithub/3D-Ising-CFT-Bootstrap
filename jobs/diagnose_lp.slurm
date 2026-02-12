#!/bin/bash
#SBATCH --job-name=lp_diagnose
#SBATCH --account=randall_lab
#SBATCH --partition=sapphire
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/diagnose_lp_%j.log
#SBATCH --error=logs/diagnose_lp_%j.log

set -e

echo "=== LP Conditioning Diagnostic ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

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

python scripts/diagnose_lp.py

echo ""
echo "Done: $(date)"
