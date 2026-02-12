#!/bin/bash
# =============================================================
# Full pipeline launcher: Stage A -> Stage B -> Plot
#
# Optional legacy gates can run before Stage A when RUN_TEST_GATES=1.
# Each step only runs if the previous succeeded (--dependency=afterok).
#
# Usage:
#   cd /n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap
#   bash jobs/run_pipeline.sh
#
# Monitor:
#   sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS
#   tail -f logs/test_gates_*.log
#   tail -f logs/stage_a_sdpb_*.log
# =============================================================

set -e

cd "$(dirname "$0")/.."
echo "=== 3D Ising CFT Bootstrap Pipeline ==="
echo "Working directory: $(pwd)"
echo "Started: $(date)"
echo ""

# Load notification config if available
[ -f .env.notifications ] && source .env.notifications

# Load validation config if available
[ -f .env.validation ] && source .env.validation

# Runtime envelope (override at submit shell with env vars).
SDPB_TIMEOUT="${SDPB_TIMEOUT:-18000}"  # 5 hours (measured SDPB solve time)
STAGE_A_TOLERANCE="${STAGE_A_TOLERANCE:-1e-4}"
STAGE_B_TOLERANCE="${STAGE_B_TOLERANCE:-1e-3}"
EPS_SNAP_TOLERANCE="${EPS_SNAP_TOLERANCE:-1e-3}"

STAGE_A_ARRAY="${STAGE_A_ARRAY:-0-50}"
STAGE_B_ARRAY="${STAGE_B_ARRAY:-0-50}"

STAGE_A_CPUS="${STAGE_A_CPUS:-16}"  # Sapphire production config
STAGE_A_MEM="${STAGE_A_MEM:-128G}"
STAGE_A_TIME="${STAGE_A_TIME:-36:00:00}"  # Safety margin for 28-35h runtime

STAGE_B_CPUS="${STAGE_B_CPUS:-16}"  # Sapphire production config
STAGE_B_MEM="${STAGE_B_MEM:-128G}"
STAGE_B_TIME="${STAGE_B_TIME:-36:00:00}"  # Safety margin for 28-35h runtime
RUN_TEST_GATES="${RUN_TEST_GATES:-0}"

echo "Runtime envelope:"
echo "  SDPB timeout: ${SDPB_TIMEOUT}s"
echo "  Stage A tolerance: ${STAGE_A_TOLERANCE}"
echo "  Stage B tolerance: ${STAGE_B_TOLERANCE}"
echo "  Epsilon snap tolerance: ${EPS_SNAP_TOLERANCE}"
echo "  Stage A array: ${STAGE_A_ARRAY}"
echo "  Stage B array: ${STAGE_B_ARRAY}"
echo "  Stage A resources: ${STAGE_A_CPUS} CPUs, ${STAGE_A_MEM}, ${STAGE_A_TIME}"
echo "  Stage B resources: ${STAGE_B_CPUS} CPUs, ${STAGE_B_MEM}, ${STAGE_B_TIME}"
echo "  Run legacy test gates: ${RUN_TEST_GATES}"
echo ""

# Ensure log and data directories exist
mkdir -p logs data figures

# Clean up any stale per-task CSVs from previous runs
echo "Cleaning stale per-task CSVs..."
rm -f data/eps_bound_[0-9]*.csv data/epsprime_bound_[0-9]*.csv
echo ""

# --- Step 1 (optional): legacy test gates ---
STAGE_A_DEP_ARGS=()
GATE_JOB=""
if [ "${RUN_TEST_GATES}" = "1" ]; then
    echo "--- Step 1: Submitting test gates ---"
    GATE_JOB=$(sbatch --parsable jobs/test_gates.slurm)
    STAGE_A_DEP_ARGS=(--dependency=afterok:${GATE_JOB})
    echo "  Test gates job: ${GATE_JOB}"
else
    echo "--- Step 1: Skipping legacy test gates (RUN_TEST_GATES=${RUN_TEST_GATES}) ---"
fi

# --- Step 2: Stage A (SDPB array) ---
echo "--- Step 2: Submitting Stage A ---"
STAGE_A_JOB=$(sbatch --parsable \
    "${STAGE_A_DEP_ARGS[@]}" \
    --array="${STAGE_A_ARRAY}" \
    --cpus-per-task="${STAGE_A_CPUS}" \
    --mem="${STAGE_A_MEM}" \
    --time="${STAGE_A_TIME}" \
    --export=ALL,SDPB_TIMEOUT="${SDPB_TIMEOUT}",STAGE_A_TOLERANCE="${STAGE_A_TOLERANCE}" \
    jobs/stage_a_sdpb.slurm)
echo "  Stage A job: ${STAGE_A_JOB} (array ${STAGE_A_ARRAY})"

# Submit progressive validation daemon for Stage A (if enabled)
if [ "${ENABLE_PROGRESSIVE_VALIDATION:-1}" = "1" ] && [ -f jobs/progressive_validation.slurm ]; then
    echo "  Launching progressive validation daemon for Stage A..."
    VALIDATION_A_JOB=$(sbatch --parsable \
        --dependency=after:${STAGE_A_JOB} \
        --kill-on-invalid-dep=yes \
        --export=ALL,STAGE=a,JOB_ID_TO_MONITOR="${STAGE_A_JOB}",POLL_INTERVAL="${POLL_INTERVAL:-60}" \
        jobs/progressive_validation.slurm)
    echo "  Validation daemon (Stage A): ${VALIDATION_A_JOB} (monitoring ${STAGE_A_JOB})"
fi

# --- Step 3: Merge Stage A + Submit Stage B + Final Plot ---
echo "--- Step 3: Submitting merge + Stage B launcher (depends on Stage A) ---"
MERGE_JOB=$(sbatch --parsable \
    --dependency=afterok:${STAGE_A_JOB} \
    --export=ALL,SDPB_TIMEOUT="${SDPB_TIMEOUT}",STAGE_B_TOLERANCE="${STAGE_B_TOLERANCE}",EPS_SNAP_TOLERANCE="${EPS_SNAP_TOLERANCE}",STAGE_B_ARRAY="${STAGE_B_ARRAY}",STAGE_B_CPUS="${STAGE_B_CPUS}",STAGE_B_MEM="${STAGE_B_MEM}",STAGE_B_TIME="${STAGE_B_TIME}" \
    jobs/merge_stage_a_and_submit_b.slurm)
echo "  Merge+submit job: ${MERGE_JOB}"

# Note: Stage B and final plot are submitted from within merge_stage_a_and_submit_b.slurm
# using sbatch --dependency, so they chain automatically.

echo ""
echo "=== Pipeline submitted ==="
echo ""
echo "Job chain:"
if [ -n "${GATE_JOB}" ]; then
    echo "  1. Test gates:    ${GATE_JOB}"
    echo "  2. Stage A:       ${STAGE_A_JOB} (array ${STAGE_A_ARRAY})"
    echo "  3. Merge + sub B: ${MERGE_JOB}"
    echo "  4. Stage B:       (submitted by step 3)"
    echo "  5. Final plot:    (submitted by step 3)"
else
    echo "  1. Stage A:       ${STAGE_A_JOB} (array ${STAGE_A_ARRAY})"
    echo "  2. Merge + sub B: ${MERGE_JOB}"
    echo "  3. Stage B:       (submitted by step 2)"
    echo "  4. Final plot:    (submitted by step 2)"
fi
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
if [ -n "${GATE_JOB}" ]; then
    echo "  sacct -j ${GATE_JOB},${STAGE_A_JOB},${MERGE_JOB} --format=JobID,JobName,State,Elapsed"
else
    echo "  sacct -j ${STAGE_A_JOB},${MERGE_JOB} --format=JobID,JobName,State,Elapsed"
fi
echo ""
echo "Logs:"
if [ -n "${GATE_JOB}" ]; then
    echo "  tail -f logs/test_gates_${GATE_JOB}.log"
fi
echo "  tail -f logs/stage_a_sdpb_${STAGE_A_JOB}_0.log"
