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

# Load notification config if available
[ -f .env.notifications ] && source .env.notifications

# Load validation config if available
[ -f .env.validation ] && source .env.validation

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
    ERROR_MSG="Found ${NONFINITE_ROWS} non-finite Stage A values (NaN/inf). This indicates SDPB timeout or solver failures."
    echo "ERROR: ${ERROR_MSG}"

    # Send failure notification
    if [ -f scripts/notification.py ]; then
        python scripts/notification.py \
            --title "Stage A Validation FAILED" \
            --message "${ERROR_MSG}" \
            --severity critical \
            --event-type stage_a_validation \
            --context "valid_rows=${VALID_ROWS},nonfinite_rows=${NONFINITE_ROWS},pattern=nan_values" || true
    fi

    exit 1
fi

if [ "${LOWER_ROWS}" -eq "${VALID_ROWS}" ]; then
    ERROR_MSG="All valid Stage A results are near the unitarity floor (~0.5). This indicates LP solver conditioning bug (scipy/HiGHS). Ensure SDPB backend is being used."
    echo "ERROR: ${ERROR_MSG}"

    # Send failure notification
    if [ -f scripts/notification.py ]; then
        python scripts/notification.py \
            --title "Stage A Validation FAILED" \
            --message "${ERROR_MSG}" \
            --severity critical \
            --event-type stage_a_validation \
            --context "valid_rows=${VALID_ROWS},pattern=all_lower_bound" || true
    fi

    exit 1
fi

if [ "${UPPER_ROWS}" -eq "${VALID_ROWS}" ]; then
    ERROR_MSG="All valid Stage A results are near the upper scan bound (~2.5). This indicates LP solver is broken or constraints are incorrect."
    echo "ERROR: ${ERROR_MSG}"

    # Send failure notification
    if [ -f scripts/notification.py ]; then
        python scripts/notification.py \
            --title "Stage A Validation FAILED" \
            --message "${ERROR_MSG}" \
            --severity critical \
            --event-type stage_a_validation \
            --context "valid_rows=${VALID_ROWS},pattern=all_upper_bound" || true
    fi

    exit 1
fi

echo "Stage A sanity checks passed."

# Send success notification
if [ -f scripts/notification.py ]; then
    python scripts/notification.py \
        --title "Stage A Complete - Validation Passed" \
        --message "Stage A (upper bound on Δε) completed successfully. ${VALID_ROWS} valid results. Proceeding to Stage B submission." \
        --severity info \
        --event-type stage_a_complete \
        --context "valid_rows=${VALID_ROWS},total_rows=${n_rows}" || true
fi

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

# Send Stage B submission notification
if [ -f scripts/notification.py ]; then
    ARRAY_SIZE=$(echo "${STAGE_B_ARRAY}" | awk -F'-' '{if (NF==2) print $2-$1+1; else print 1}')
    python scripts/notification.py \
        --title "Stage B Submitted" \
        --message "Stage B (upper bound on Δε') has been submitted. Job will compute the second constraint for each point from Stage A." \
        --severity info \
        --event-type stage_b_submit \
        --context "job_id=${STAGE_B_JOB},array_size=${ARRAY_SIZE},estimated_runtime=28-35 hours" || true
fi

# Submit progressive validation daemon for Stage B (if enabled)
if [ "${ENABLE_PROGRESSIVE_VALIDATION:-1}" = "1" ] && [ -f jobs/progressive_validation.slurm ]; then
    echo "--- Launching progressive validation daemon for Stage B ---"
    VALIDATION_B_JOB=$(sbatch --parsable \
        --dependency=after:${STAGE_B_JOB} \
        --kill-on-invalid-dep=yes \
        --export=ALL,STAGE=b,JOB_ID_TO_MONITOR="${STAGE_B_JOB}",POLL_INTERVAL="${POLL_INTERVAL:-60}" \
        jobs/progressive_validation.slurm)
    echo "Validation daemon (Stage B): ${VALIDATION_B_JOB} (monitoring ${STAGE_B_JOB})"
fi

# --- Step 4: Submit final merge + plot (after Stage B) ---
PLOT_JOB=$(sbatch --parsable --dependency=afterok:${STAGE_B_JOB} jobs/final_merge_and_plot.slurm)
echo "Final plot job: ${PLOT_JOB} (depends on Stage B ${STAGE_B_JOB})"

echo ""
echo "Pipeline continuation submitted."
echo "  Stage B: ${STAGE_B_JOB}"
echo "  Plot:    ${PLOT_JOB}"
echo "End: $(date)"
