#!/bin/bash
#SBATCH --job-name=test_error_detection
#SBATCH --account=randall_lab
#SBATCH --partition=sapphire
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/test_error_detection_%j.log
#SBATCH --error=logs/test_error_detection_%j.log

# TEST: Verify error detection works with insufficient memory
# This should trigger OOM and the new error handling should detect it

set -euo pipefail
export PYTHONUNBUFFERED=1

# --- Environment ---
source ~/.bashrc
conda activate ising_bootstrap

PROJECT_DIR="/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap"
cd "$PROJECT_DIR"

mkdir -p logs data

# Use Δσ = 0.518 (near the kink, should be challenging)
SIGMA=0.518

echo "=========================================="
echo "TEST 1: Error Detection with LOW Memory"
echo "=========================================="
echo "Purpose: Verify SDPB failures are properly detected"
echo "Expected: Job should fail fast with clear OOM error message"
echo "Memory: 16G (intentionally insufficient)"
echo "Delta_sigma: ${SIGMA}"
echo "Start: $(date)"
echo ""

# --- Run scan ---
python -m ising_bootstrap.scans.stage_a \
    --sigma-min "${SIGMA}" \
    --sigma-max "${SIGMA}" \
    --sigma-step 1.0 \
    --tolerance 1e-4 \
    --backend sdpb \
    --sdpb-image tools/sdpb-3.1.0.sif \
    --sdpb-precision 1024 \
    --sdpb-cores "${SLURM_CPUS_PER_TASK}" \
    --output "data/test_error_detection.csv" \
    --verbose

EXIT_CODE=$?

echo ""
echo "=== Test Complete: $(date) ==="
echo "Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "✓ EXPECTED: Job failed as expected (insufficient memory)"
    echo "Check above for error messages containing:"
    echo "  - 'SDPB solver failed 3 consecutive times'"
    echo "  - 'OOM kill' or 'signal 9'"
    echo ""
    echo "If you see these messages, error detection is WORKING!"
fi

exit ${EXIT_CODE}
