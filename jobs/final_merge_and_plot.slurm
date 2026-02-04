#!/bin/bash
#SBATCH --job-name=ising_final
#SBATCH --account=iaifi_lab
#SBATCH --partition=shared
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/final_merge_and_plot_%j.log
#SBATCH --error=logs/final_merge_and_plot_%j.log

# Final pipeline step: merge Stage B results and generate Figure 6.
# Runs after Stage B array job completes.

set -e

echo "=== Final: Merge Stage B + Plot Figure 6 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

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

cd "$SLURM_SUBMIT_DIR"

# Step 1: Merge Stage B CSVs
echo "--- Merging Stage B results ---"
bash jobs/merge_stage_b.sh
echo ""

# Step 2: Generate Figure 6
echo "--- Generating Figure 6 ---"
mkdir -p figures
python -m ising_bootstrap.plot.fig6 \
    --data data/epsprime_bound.csv \
    --output figures/fig6_reproduction.png \
    --dpi 300

echo ""
echo "=== Pipeline complete! ==="
echo "Figure saved to: figures/fig6_reproduction.png"
echo "                 figures/fig6_reproduction.pdf"
echo "End: $(date)"
