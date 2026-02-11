#!/bin/bash
# Submit one-point Stage B SDPB smoke test with resource and timeout overrides.
#
# Usage examples:
#   bash jobs/submit_stage_b_smoke.sh
#   TIMEOUT=3600 CPUS=16 MEM=160G bash jobs/submit_stage_b_smoke.sh
#   SIGMA=0.518 EPS_BOUND_CSV=data/eps_bound.csv OUTPUT_CSV=data/stage_b_smoke.csv bash jobs/submit_stage_b_smoke.sh

set -euo pipefail

TIMEOUT="${TIMEOUT:-1800}"
CPUS="${CPUS:-8}"
MEM="${MEM:-128G}"
WALLTIME="${WALLTIME:-06:00:00}"
SIGMA="${SIGMA:-0.518}"
EPS_BOUND_CSV="${EPS_BOUND_CSV:-data/eps_bound.csv}"
STAGE_B_TOLERANCE="${STAGE_B_TOLERANCE:-1e-3}"
EPS_SNAP_TOLERANCE="${EPS_SNAP_TOLERANCE:-1e-3}"
OUTPUT_CSV="${OUTPUT_CSV:-data/stage_b_smoke.csv}"
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
    --export=ALL,SIGMA="${SIGMA}",EPS_BOUND_CSV="${EPS_BOUND_CSV}",SDPB_TIMEOUT="${TIMEOUT}",STAGE_B_TOLERANCE="${STAGE_B_TOLERANCE}",EPS_SNAP_TOLERANCE="${EPS_SNAP_TOLERANCE}",OUTPUT_CSV="${OUTPUT_CSV}" \
    jobs/stage_b_smoke_sdpb.slurm)

echo "Submitted Stage B smoke job: ${JOB_ID}"
echo "  Delta_sigma: ${SIGMA}"
echo "  Stage A map: ${EPS_BOUND_CSV}"
echo "  Timeout: ${TIMEOUT}s"
echo "  CPUs: ${CPUS}"
echo "  Memory: ${MEM}"
echo "  Walltime: ${WALLTIME}"
echo "  Stage B tolerance: ${STAGE_B_TOLERANCE}"
echo "  Epsilon snap tolerance: ${EPS_SNAP_TOLERANCE}"
echo "  Output CSV: ${OUTPUT_CSV}"
echo ""
echo "Monitor:"
echo "  sacct -j ${JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem"
echo "  tail -200 logs/stage_b_smoke_${JOB_ID}.log"
echo "  cat ${OUTPUT_CSV}"
