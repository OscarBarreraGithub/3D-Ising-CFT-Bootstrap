#!/bin/bash
#SBATCH --job-name=ising_merge_a_submit_b
#SBATCH --account=randall_lab
#SBATCH --partition=sapphire
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

SDPB_TIMEOUT="${SDPB_TIMEOUT:-18000}"  # 5 hours
STAGE_B_TOLERANCE="${STAGE_B_TOLERANCE:-1e-3}"
EPS_SNAP_TOLERANCE="${EPS_SNAP_TOLERANCE:-1e-3}"
STAGE_B_ARRAY="${STAGE_B_ARRAY:-0-50}"
STAGE_B_CPUS="${STAGE_B_CPUS:-16}"  # Sapphire production config
STAGE_B_MEM="${STAGE_B_MEM:-128G}"
STAGE_B_TIME="${STAGE_B_TIME:-36:00:00}"  # Sapphire production config

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

eval "$(python - <<'PY'
import csv
import math

rows = 0
valid = 0
nonfinite = 0
near_lower = 0
near_upper = 0

with open("data/eps_bound.csv", "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows += 1
        try:
            value = float(row["delta_eps_max"])
        except (ValueError, KeyError):
            nonfinite += 1
            continue

        if not math.isfinite(value):
            nonfinite += 1
            continue

        valid += 1
        if value < 0.51:
            near_lower += 1
        if value > 2.49:
            near_upper += 1

print(f"VALID_ROWS={valid}")
print(f"NONFINITE_ROWS={nonfinite}")
print(f"LOWER_ROWS={near_lower}")
print(f"UPPER_ROWS={near_upper}")
PY
)"

echo "Validation summary: valid=${VALID_ROWS}, nonfinite=${NONFINITE_ROWS}, lower=${LOWER_ROWS}, upper=${UPPER_ROWS}"

if [ "${VALID_ROWS}" -lt 10 ]; then
    echo "ERROR: Too few valid Stage A rows (${VALID_ROWS} < 10). Aborting pipeline."
    exit 1
fi

if [ "${NONFINITE_ROWS}" -gt 0 ]; then
    echo "ERROR: Found ${NONFINITE_ROWS} non-finite Stage A values. Aborting pipeline."
    exit 1
fi

if [ "${LOWER_ROWS}" -eq "${VALID_ROWS}" ]; then
    echo "ERROR: All valid Stage A results are near the unitarity floor (~0.5). Solver is still broken."
    exit 1
fi

if [ "${UPPER_ROWS}" -eq "${VALID_ROWS}" ]; then
    echo "ERROR: All valid Stage A results are near the upper scan bound (~2.5). Solver is still broken."
    exit 1
fi

echo "Stage A sanity checks passed."

# --- Step 3: Submit Stage B array job ---
echo ""
echo "--- Submitting Stage B ---"
echo "Stage B envelope: array=${STAGE_B_ARRAY}, cpus=${STAGE_B_CPUS}, mem=${STAGE_B_MEM}, time=${STAGE_B_TIME}, timeout=${SDPB_TIMEOUT}s"
STAGE_B_JOB=$(sbatch --parsable \
    --array="${STAGE_B_ARRAY}" \
    --cpus-per-task="${STAGE_B_CPUS}" \
    --mem="${STAGE_B_MEM}" \
    --time="${STAGE_B_TIME}" \
    --export=ALL,SDPB_TIMEOUT="${SDPB_TIMEOUT}",STAGE_B_TOLERANCE="${STAGE_B_TOLERANCE}",EPS_SNAP_TOLERANCE="${EPS_SNAP_TOLERANCE}" \
    jobs/stage_b_sdpb.slurm)
echo "Stage B job: ${STAGE_B_JOB}"

# --- Step 4: Submit final merge + plot (after Stage B) ---
PLOT_JOB=$(sbatch --parsable --dependency=afterok:${STAGE_B_JOB} jobs/final_merge_and_plot.slurm)
echo "Final plot job: ${PLOT_JOB} (depends on Stage B ${STAGE_B_JOB})"

echo ""
echo "Pipeline continuation submitted."
echo "  Stage B: ${STAGE_B_JOB}"
echo "  Plot:    ${PLOT_JOB}"
echo "End: $(date)"
