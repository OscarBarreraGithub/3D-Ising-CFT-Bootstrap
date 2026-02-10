#!/bin/bash
#SBATCH --job-name=ising_merge_a_submit_b
#SBATCH --account=randall_lab
#SBATCH --partition=shared
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/merge_a_submit_b_%j.log
#SBATCH --error=logs/merge_a_submit_b_%j.log

# Merge Stage A results and submit Stage B + final plot jobs.
# Runs after all Stage A array tasks complete.

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=== Merge Stage A + Submit Stage B ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

source ~/.bashrc
conda activate ising_bootstrap

PROJECT_DIR="/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap"
cd "$PROJECT_DIR"

# --- Step 1: Merge Stage A CSVs ---
echo "--- Merging Stage A results ---"
bash jobs/merge_stage_a.sh
echo ""

# --- Step 2: Validate merged results ---
n_rows=$(tail -n +2 data/eps_bound.csv | wc -l | tr -d ' ')
echo "Stage A merged: ${n_rows} data points"

if [ "$n_rows" -lt 10 ]; then
    echo "ERROR: Too few Stage A results (${n_rows} < 10). Aborting pipeline."
    exit 1
fi

# Quick sanity check: are all values 0.5 (the known bad result)?
n_bad=$(tail -n +2 data/eps_bound.csv | awk -F',' '$2 < 0.51 {count++} END {print count+0}')
if [ "$n_bad" -eq "$n_rows" ]; then
    echo "ERROR: All Stage A results are ≤ 0.5 (unitarity floor). Solver is still broken."
    exit 1
fi
echo "Sanity check passed: ${n_bad}/${n_rows} points at unitarity floor"

# --- Step 3: Submit Stage B array job ---
echo ""
echo "--- Submitting Stage B ---"
STAGE_B_JOB=$(sbatch --parsable jobs/stage_b_sdpb.slurm)
echo "Stage B job: ${STAGE_B_JOB}"

# --- Step 4: Submit final merge + plot (after Stage B) ---
PLOT_JOB=$(sbatch --parsable --dependency=afterok:${STAGE_B_JOB} jobs/final_merge_and_plot.slurm)
echo "Final plot job: ${PLOT_JOB} (depends on Stage B ${STAGE_B_JOB})"

echo ""
echo "Pipeline continuation submitted."
echo "  Stage B: ${STAGE_B_JOB}"
echo "  Plot:    ${PLOT_JOB}"
echo "End: $(date)"
