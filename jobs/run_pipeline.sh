#!/bin/bash
# =============================================================
# Full pipeline launcher: Test Gates -> Stage A -> Stage B -> Plot
#
# Each step only runs if the previous succeeded (--dependency=afterok).
# Gate failures abort the entire pipeline.
#
# Usage:
#   cd /n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap
#   bash jobs/run_pipeline.sh
#
# Monitor:
#   sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS
#   tail -f logs/test_gates_*.log
#   tail -f logs/stage_a_ext_*.log
# =============================================================

set -e

cd "$(dirname "$0")/.."
echo "=== 3D Ising CFT Bootstrap Pipeline ==="
echo "Working directory: $(pwd)"
echo "Started: $(date)"
echo ""

# Ensure log and data directories exist
mkdir -p logs data figures

# Clean up any stale per-task CSVs from previous runs
echo "Cleaning stale per-task CSVs..."
rm -f data/eps_bound_[0-9]*.csv data/epsprime_bound_[0-9]*.csv
echo ""

# --- Step 1: Test Gates ---
echo "--- Step 1: Submitting test gates ---"
GATE_JOB=$(sbatch --parsable jobs/test_gates.slurm)
echo "  Test gates job: ${GATE_JOB}"

# --- Step 2: Stage A (51-point array, extended solver) ---
echo "--- Step 2: Submitting Stage A (depends on gates) ---"
STAGE_A_JOB=$(sbatch --parsable --dependency=afterok:${GATE_JOB} jobs/stage_a_extended.slurm)
echo "  Stage A job: ${STAGE_A_JOB} (array 0-50)"

# --- Step 3: Merge Stage A + Submit Stage B + Final Plot ---
echo "--- Step 3: Submitting merge + Stage B launcher (depends on Stage A) ---"
MERGE_JOB=$(sbatch --parsable --dependency=afterok:${STAGE_A_JOB} jobs/merge_stage_a_and_submit_b.slurm)
echo "  Merge+submit job: ${MERGE_JOB}"

# Note: Stage B and final plot are submitted from within merge_stage_a_and_submit_b.slurm
# using sbatch --dependency, so they chain automatically.

echo ""
echo "=== Pipeline submitted ==="
echo ""
echo "Job chain:"
echo "  1. Test gates:    ${GATE_JOB}"
echo "  2. Stage A:       ${STAGE_A_JOB} (51 array tasks)"
echo "  3. Merge + sub B: ${MERGE_JOB}"
echo "  4. Stage B:       (submitted by step 3)"
echo "  5. Final plot:    (submitted by step 3)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j ${GATE_JOB},${STAGE_A_JOB},${MERGE_JOB} --format=JobID,JobName,State,Elapsed"
echo ""
echo "Logs:"
echo "  tail -f logs/test_gates_${GATE_JOB}.log"
echo "  tail -f logs/stage_a_ext_${STAGE_A_JOB}_0.log"
