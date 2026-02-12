#!/bin/bash
#SBATCH --job-name=ising_final
#SBATCH --account=randall_lab
#SBATCH --partition=sapphire
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/final_merge_and_plot_%j.log
#SBATCH --error=logs/final_merge_and_plot_%j.log

# Final pipeline step: merge Stage B results and generate Figure 6.
# Runs after Stage B array job completes.

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=== Final: Merge Stage B + Plot Figure 6 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

source ~/.bashrc
conda activate ising_bootstrap

PROJECT_DIR="/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap"
cd "$PROJECT_DIR"

# Load notification config if available
[ -f .env.notifications ] && source .env.notifications

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

# Send completion notification
if [ -f scripts/notification.py ]; then
    python scripts/notification.py \
        --title "Pipeline Complete - Figure 6 Ready!" \
        --message "The 3D-Ising-CFT-Bootstrap pipeline has completed successfully. Figure 6 has been generated and saved. Compare with reference figure el_showk_2012_fig6.png to validate results." \
        --severity info \
        --context "output_png=figures/fig6_reproduction.png,output_pdf=figures/fig6_reproduction.pdf" || true
fi
