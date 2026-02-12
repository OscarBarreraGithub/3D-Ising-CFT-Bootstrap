#!/bin/bash
#SBATCH --job-name=ising_validation
#SBATCH --account=randall_lab
#SBATCH --partition=shared
#SBATCH --time=48:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/progressive_validation_%j.log
#SBATCH --error=logs/progressive_validation_%j.log

# =============================================================
# Progressive Validation Daemon for 3D-Ising-CFT-Bootstrap
# =============================================================
#
# This job monitors Stage A or B results as they complete and
# detects anomaly patterns early (all NaN, all ~0.5, all ~2.5).
#
# Triggered by merge_stage_a_and_submit_b.slurm with:
#   --dependency=after:STAGE_A_JOB_ID
#
# Environment variables (set by parent script):
#   STAGE: "a" or "b"
#   JOB_ID_TO_MONITOR: Parent Stage A/B job ID
#   POLL_INTERVAL: Seconds between checks (default: 60)
#   CANCEL_ON_CRITICAL: Auto-cancel on critical anomaly (default: 0)
#
# =============================================================

set -euo pipefail
export PYTHONUNBUFFERED=1

# --- Environment ---
source ~/.bashrc
conda activate ising_bootstrap

PROJECT_DIR="/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap"
cd "$PROJECT_DIR"

mkdir -p logs

# Load validation config if available
[ -f .env.validation ] && source .env.validation

# Load notification config if available (for alerts)
[ -f .env.notifications ] && source .env.notifications

# --- Required Parameters ---
STAGE="${STAGE:-a}"
JOB_ID_TO_MONITOR="${JOB_ID_TO_MONITOR:-unknown}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
CANCEL_ON_CRITICAL="${CANCEL_ON_CRITICAL:-0}"

# Anomaly thresholds
ANOMALY_THRESHOLD_WARNING="${ANOMALY_THRESHOLD_WARNING:-10}"
ANOMALY_THRESHOLD_CRITICAL="${ANOMALY_THRESHOLD_CRITICAL:-20}"

echo "=== Progressive Validation Daemon ==="
echo "Stage: ${STAGE}"
echo "Monitoring job: ${JOB_ID_TO_MONITOR}"
echo "Poll interval: ${POLL_INTERVAL}s"
echo "Cancel on critical: ${CANCEL_ON_CRITICAL}"
echo "Anomaly thresholds: warning=${ANOMALY_THRESHOLD_WARNING}, critical=${ANOMALY_THRESHOLD_CRITICAL}"
echo "Start: $(date)"
echo ""

# --- Run Validator ---
python scripts/progressive_validator.py \
    --stage "${STAGE}" \
    --job-id "${JOB_ID_TO_MONITOR}" \
    --poll-interval "${POLL_INTERVAL}" \
    --total-expected 51 \
    --anomaly-threshold-warning "${ANOMALY_THRESHOLD_WARNING}" \
    --anomaly-threshold-critical "${ANOMALY_THRESHOLD_CRITICAL}" \
    $([ "${CANCEL_ON_CRITICAL}" = "1" ] && echo "--cancel-on-critical" || echo "") \
    --output "logs/validation_state_stage${STAGE}_${JOB_ID_TO_MONITOR}.json"

EXIT_CODE=$?

echo ""
echo "End: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
