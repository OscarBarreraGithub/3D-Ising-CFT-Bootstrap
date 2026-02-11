#!/bin/bash
# Submit a single-point Stage A SDPB characterization run at Delta_sigma=0.518.
#
# This is the Phase 1 helper for tuning timeout/cores/memory before full arrays.
#
# Usage examples:
#   bash jobs/submit_stage_a_runtime_envelope.sh
#   TIMEOUT=3600 bash jobs/submit_stage_a_runtime_envelope.sh
#   TIMEOUT=3600 CPUS=16 MEM=160G WALLTIME=08:00:00 bash jobs/submit_stage_a_runtime_envelope.sh
#   TOLERANCE=1e-3 OUTPUT_CSV=data/test_relaxed.csv bash jobs/submit_stage_a_runtime_envelope.sh

set -euo pipefail

TIMEOUT="${TIMEOUT:-1800}"
CPUS="${CPUS:-8}"
MEM="${MEM:-128G}"
WALLTIME="${WALLTIME:-06:00:00}"
TOLERANCE="${TOLERANCE:-1e-4}"
SIGMA="${SIGMA:-0.518}"
OUTPUT_CSV="${OUTPUT_CSV:-data/test_sufficient_memory.csv}"
PARTITION="${PARTITION:-}"

SBATCH_ARGS=(
    --parsable
    --cpus-per-task="${CPUS}"
    --mem="${MEM}"
    --time="${WALLTIME}"
)

if [ -n "${PARTITION}" ]; then
    SBATCH_ARGS+=(--partition="${PARTITION}")
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" \
    --export=ALL,SDPB_TIMEOUT="${TIMEOUT}",STAGE_A_TOLERANCE="${TOLERANCE}",SIGMA="${SIGMA}",OUTPUT_CSV="${OUTPUT_CSV}" \
    jobs/test_sufficient_memory.slurm)

echo "Submitted Stage A runtime-envelope job: ${JOB_ID}"
echo "  Delta_sigma: ${SIGMA}"
echo "  Timeout: ${TIMEOUT}s"
echo "  CPUs: ${CPUS}"
echo "  Memory: ${MEM}"
echo "  Walltime: ${WALLTIME}"
echo "  Tolerance: ${TOLERANCE}"
echo "  Output CSV: ${OUTPUT_CSV}"
echo ""
echo "Monitor:"
echo "  sacct -j ${JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem"
echo "  tail -200 logs/test_sufficient_memory_${JOB_ID}.log"
echo "  cat ${OUTPUT_CSV}"
