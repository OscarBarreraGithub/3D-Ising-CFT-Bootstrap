#!/bin/bash
#SBATCH --job-name=test_sufficient_memory
#SBATCH --account=randall_lab
#SBATCH --partition=sapphire
#SBATCH --time=36:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/test_sufficient_memory_%j.log
#SBATCH --error=logs/test_sufficient_memory_%j.log

# TEST: Verify Stage A works with sufficient memory
# This should succeed and produce valid results

set -euo pipefail
export PYTHONUNBUFFERED=1

# --- Environment ---
source ~/.bashrc
conda activate ising_bootstrap

PROJECT_DIR="/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap"
cd "$PROJECT_DIR"

mkdir -p logs data

SDPB_TIMEOUT="${SDPB_TIMEOUT:-1800}"
STAGE_A_TOLERANCE="${STAGE_A_TOLERANCE:-1e-4}"
SIGMA="${SIGMA:-0.518}"
OUTPUT_CSV="${OUTPUT_CSV:-data/test_sufficient_memory.csv}"

# Use Δσ = 0.518 (near the kink, should be challenging)
echo "=========================================="
echo "TEST 2: Stage A with SUFFICIENT Memory"
echo "=========================================="
echo "Purpose: Verify Stage A succeeds with adequate resources"
echo "Expected: Should complete successfully with Δε_max ≈ 1.41"
echo "Memory: 128G (should be sufficient)"
echo "Delta_sigma: ${SIGMA}"
echo "Timeout: ${SDPB_TIMEOUT}s"
echo "Tolerance: ${STAGE_A_TOLERANCE}"
echo "Output: ${OUTPUT_CSV}"
echo "Start: $(date)"
echo ""

# --- Run scan ---
set +e
python -m ising_bootstrap.scans.stage_a \
    --sigma-min "${SIGMA}" \
    --sigma-max "${SIGMA}" \
    --sigma-step 1.0 \
    --tolerance "${STAGE_A_TOLERANCE}" \
    --backend sdpb \
    --sdpb-image tools/sdpb-3.1.0.sif \
    --sdpb-precision 1024 \
    --sdpb-cores "${SLURM_CPUS_PER_TASK}" \
    --sdpb-timeout "${SDPB_TIMEOUT}" \
    --output "${OUTPUT_CSV}" \
    --verbose

EXIT_CODE=$?
set -e

echo ""
echo "=== Test Complete: $(date) ==="
echo "Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ SUCCESS: Job completed"
    echo ""
    echo "Result:"
    cat "${OUTPUT_CSV}"
    echo ""

    # Extract Δε_max value
    EPS_MAX=$(tail -n 1 "${OUTPUT_CSV}" | cut -d',' -f2)
    echo "Δε_max = ${EPS_MAX}"
    echo ""
    echo "Expected: ~1.41 (from arXiv:1203.6064 Fig 6)"
    echo ""

    # Check if result is reasonable (not the 2.5 upper bound bug)
    if python - "${EPS_MAX}" <<'PY'
import math
import sys

value = float(sys.argv[1])
if math.isfinite(value) and value < 2.0:
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
        echo "✓ Result looks REASONABLE (not the 2.5 bug)"
    else
        echo "✗ WARNING: Result looks suspicious (may still have issues)"
    fi
else
    echo "✗ FAILED: Job failed unexpectedly"
    echo "Check logs above for errors"
fi

exit ${EXIT_CODE}
