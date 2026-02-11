#!/bin/bash
# =============================================================
# Overnight Full Pipeline Automation
# =============================================================
#
# This script:
# 1. Iteratively tests SDPB timeouts to find the right value
# 2. Once successful, launches the full Stage A + Stage B pipeline
# 3. Generates the final Figure 6 plot
# 4. Has robust error handling to kill jobs if things go wrong
#
# Usage:
#   cd /n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap
#   bash jobs/overnight_full_pipeline.sh
#
# Output:
#   - Status log: logs/overnight_pipeline_TIMESTAMP.log
#   - Final plot: figures/fig6_reproduction.png (if successful)
#
# =============================================================

set -euo pipefail

PROJECT_DIR="/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap"
cd "$PROJECT_DIR"

# Create timestamped log
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/overnight_pipeline_${TIMESTAMP}.log"
mkdir -p logs

# Redirect all output to both console and log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "============================================="
echo "OVERNIGHT FULL PIPELINE AUTOMATION"
echo "============================================="
echo "Started: $(date)"
echo "Log file: $LOG_FILE"
echo ""

# Track all submitted job IDs for emergency cleanup
ALL_JOBS=()
CLEANUP_NEEDED=false

# Emergency cleanup function
cleanup_all_jobs() {
    if [ "$CLEANUP_NEEDED" = true ] && [ ${#ALL_JOBS[@]} -gt 0 ]; then
        echo ""
        echo "=== EMERGENCY: Cancelling all jobs ==="
        for job_id in "${ALL_JOBS[@]}"; do
            echo "Cancelling job: $job_id"
            scancel "$job_id" 2>/dev/null || true
        done
        echo "All jobs cancelled."
        echo ""
    fi
}

# Set trap for cleanup on exit
trap cleanup_all_jobs EXIT

# Function to wait for a job to complete
wait_for_job() {
    local job_id=$1
    local job_name=$2
    local max_wait_seconds=${3:-28800}  # Default 8 hours

    echo "Waiting for job $job_id ($job_name) to complete..."
    echo "  Max wait: $max_wait_seconds seconds"

    local elapsed=0
    local sleep_interval=30

    while [ $elapsed -lt $max_wait_seconds ]; do
        # Check job state
        local state=$(sacct -j "$job_id" -n -o State -P | head -1 | tr -d ' ')

        case "$state" in
            COMPLETED)
                echo "  Job $job_id completed successfully"
                return 0
                ;;
            FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                echo "  ERROR: Job $job_id failed with state: $state"
                return 1
                ;;
            PENDING|RUNNING)
                # Still running, keep waiting
                sleep $sleep_interval
                elapsed=$((elapsed + sleep_interval))

                # Print progress every 5 minutes
                if [ $((elapsed % 300)) -eq 0 ]; then
                    echo "  Still waiting... (${elapsed}s elapsed, state: $state)"
                fi
                ;;
            "")
                # Job not yet in sacct, check squeue
                if squeue -j "$job_id" &>/dev/null; then
                    sleep $sleep_interval
                    elapsed=$((elapsed + sleep_interval))
                else
                    echo "  ERROR: Job $job_id not found in queue"
                    return 1
                fi
                ;;
            *)
                echo "  WARNING: Unknown job state: $state"
                sleep $sleep_interval
                elapsed=$((elapsed + sleep_interval))
                ;;
        esac
    done

    echo "  ERROR: Job $job_id exceeded maximum wait time"
    return 1
}

# Function to check if Stage A envelope test was successful
check_envelope_success() {
    local csv_file="$1"

    if [ ! -f "$csv_file" ]; then
        echo "  ERROR: CSV file not found: $csv_file"
        return 1
    fi

    # Extract the delta_eps_max value (should be second column, second row)
    local value=$(tail -n 1 "$csv_file" | cut -d',' -f2)

    if [ -z "$value" ]; then
        echo "  ERROR: Could not extract value from CSV"
        return 1
    fi

    # Check if value is a valid number
    if ! python3 -c "import math; val=float('$value'); exit(0 if math.isfinite(val) else 1)" 2>/dev/null; then
        echo "  FAILED: Value is not finite (NaN or inf): $value"
        return 1
    fi

    # Check if value is in expected range (not 2.5, not 0.5)
    if python3 -c "val=float('$value'); exit(0 if 1.0 < val < 2.0 else 1)" 2>/dev/null; then
        echo "  SUCCESS: Found valid Δε_max = $value"
        return 0
    else
        echo "  FAILED: Value out of expected range: $value (expected 1.0 < val < 2.0)"
        return 1
    fi
}

# =============================================================
# PHASE 1: Find the correct SDPB timeout
# =============================================================

echo "============================================="
echo "PHASE 1: SDPB Timeout Characterization"
echo "============================================="
echo ""

# Test configurations: (timeout|cpus|mem|walltime|description)
# NOTE: Using | delimiter instead of : to avoid conflict with HH:MM:SS walltime format
TEST_CONFIGS=(
    "1800|8|128G|06:00:00|Baseline (30 min timeout, 8 cores)"
    "3600|8|128G|08:00:00|Extended timeout (60 min, 8 cores)"
    "3600|16|160G|10:00:00|More cores (60 min, 16 cores)"
    "5400|16|160G|12:00:00|Long timeout (90 min, 16 cores)"
)

SUCCESSFUL_CONFIG=""

for config in "${TEST_CONFIGS[@]}"; do
    IFS='|' read -r TIMEOUT CPUS MEM WALLTIME DESC <<< "$config"

    echo "--- Testing: $DESC ---"
    echo "  Timeout: ${TIMEOUT}s"
    echo "  CPUs: $CPUS"
    echo "  Memory: $MEM"
    echo "  Walltime: $WALLTIME"
    echo ""

    # Clean up previous test output
    rm -f data/test_sufficient_memory.csv

    # Submit envelope test
    echo "Submitting envelope test..."
    JOB_ID=$(TIMEOUT=$TIMEOUT CPUS=$CPUS MEM=$MEM WALLTIME=$WALLTIME \
        bash jobs/submit_stage_a_runtime_envelope.sh | grep "^Submitted" | awk '{print $NF}')

    if [ -z "$JOB_ID" ]; then
        echo "ERROR: Failed to submit envelope test"
        CLEANUP_NEEDED=true
        exit 1
    fi

    ALL_JOBS+=("$JOB_ID")
    echo "Job ID: $JOB_ID"
    echo ""

    # Wait for job to complete (max 12 hours)
    if wait_for_job "$JOB_ID" "envelope_test" 43200; then
        echo "Job completed, checking results..."

        # Check the log for errors
        LOG_PATH="logs/test_sufficient_memory_${JOB_ID}.log"
        if [ -f "$LOG_PATH" ]; then
            if grep -q "ERROR:" "$LOG_PATH"; then
                echo "  Found errors in log:"
                grep "ERROR:" "$LOG_PATH" | tail -5
                echo "  This configuration failed, trying next..."
                echo ""
                continue
            fi
        fi

        # Check if results are successful
        if check_envelope_success "data/test_sufficient_memory.csv"; then
            SUCCESSFUL_CONFIG="$config"
            echo ""
            echo "✓ Found working configuration!"
            echo "  Configuration: $DESC"
            echo "  Timeout: ${TIMEOUT}s, CPUs: $CPUS, Memory: $MEM"
            echo ""
            break
        else
            echo "  Configuration failed validation, trying next..."
            echo ""
        fi
    else
        echo "  Job failed or timed out, trying next configuration..."
        echo ""
    fi
done

# Check if we found a working configuration
if [ -z "$SUCCESSFUL_CONFIG" ]; then
    echo "============================================="
    echo "FATAL ERROR: No working configuration found"
    echo "============================================="
    echo ""
    echo "Tried all timeout configurations but none succeeded."
    echo "Please review the logs and adjust the script."
    echo ""
    CLEANUP_NEEDED=true
    exit 1
fi

# Extract successful configuration
IFS='|' read -r PROD_TIMEOUT PROD_CPUS PROD_MEM PROD_WALLTIME PROD_DESC <<< "$SUCCESSFUL_CONFIG"

echo "============================================="
echo "PHASE 1 COMPLETE: Timeout Characterized"
echo "============================================="
echo "Production configuration:"
echo "  Timeout: ${PROD_TIMEOUT}s"
echo "  CPUs: $PROD_CPUS"
echo "  Memory: $PROD_MEM"
echo "  Walltime: $PROD_WALLTIME"
echo ""

# =============================================================
# PHASE 2: Full Pipeline Execution
# =============================================================

echo "============================================="
echo "PHASE 2: Launching Full Pipeline"
echo "============================================="
echo ""

# Set environment variables for production run
export SDPB_TIMEOUT=$PROD_TIMEOUT
export STAGE_A_TOLERANCE=1e-4
export STAGE_B_TOLERANCE=1e-3
export EPS_SNAP_TOLERANCE=1e-3
export STAGE_A_CPUS=$PROD_CPUS
export STAGE_A_MEM=$PROD_MEM
export STAGE_A_TIME=$PROD_WALLTIME
export STAGE_B_CPUS=$PROD_CPUS
export STAGE_B_MEM=$PROD_MEM
export STAGE_B_TIME=$PROD_WALLTIME

echo "Pipeline configuration:"
echo "  SDPB timeout: ${SDPB_TIMEOUT}s"
echo "  Stage A tolerance: $STAGE_A_TOLERANCE"
echo "  Stage B tolerance: $STAGE_B_TOLERANCE"
echo "  Epsilon snap tolerance: $EPS_SNAP_TOLERANCE"
echo "  Resources: $PROD_CPUS CPUs, $PROD_MEM, $PROD_WALLTIME"
echo ""

# Clean up old outputs
echo "Cleaning up previous outputs..."
rm -f data/eps_bound_*.csv data/epsprime_bound_*.csv
rm -f data/eps_bound.csv data/epsprime_bound.csv
echo ""

# Launch pipeline
echo "Launching pipeline..."
echo ""

# Stage A job array
echo "--- Submitting Stage A (51 points) ---"
STAGE_A_JOB=$(sbatch --parsable \
    --array=0-50 \
    --cpus-per-task="$PROD_CPUS" \
    --mem="$PROD_MEM" \
    --time="$PROD_WALLTIME" \
    --export=ALL,SDPB_TIMEOUT="$SDPB_TIMEOUT",STAGE_A_TOLERANCE="$STAGE_A_TOLERANCE" \
    jobs/stage_a_sdpb.slurm)

if [ -z "$STAGE_A_JOB" ]; then
    echo "ERROR: Failed to submit Stage A job"
    CLEANUP_NEEDED=true
    exit 1
fi

ALL_JOBS+=("$STAGE_A_JOB")
echo "Stage A job: $STAGE_A_JOB (array 0-50)"
echo ""

# Merge Stage A + Submit Stage B
echo "--- Submitting merge + Stage B launcher ---"
MERGE_JOB=$(sbatch --parsable \
    --dependency=afterok:${STAGE_A_JOB} \
    --export=ALL,SDPB_TIMEOUT="$SDPB_TIMEOUT",STAGE_B_TOLERANCE="$STAGE_B_TOLERANCE",EPS_SNAP_TOLERANCE="$EPS_SNAP_TOLERANCE",STAGE_B_ARRAY="0-50",STAGE_B_CPUS="$PROD_CPUS",STAGE_B_MEM="$PROD_MEM",STAGE_B_TIME="$PROD_WALLTIME" \
    jobs/merge_stage_a_and_submit_b.slurm)

if [ -z "$MERGE_JOB" ]; then
    echo "ERROR: Failed to submit merge job"
    CLEANUP_NEEDED=true
    exit 1
fi

ALL_JOBS+=("$MERGE_JOB")
echo "Merge + Stage B launcher: $MERGE_JOB"
echo ""

echo "Pipeline submitted successfully!"
echo ""
echo "Job chain:"
echo "  1. Stage A:       $STAGE_A_JOB (array 0-50)"
echo "  2. Merge + sub B: $MERGE_JOB"
echo "  3. Stage B:       (auto-submitted by step 2)"
echo "  4. Final plot:    (auto-submitted by step 2)"
echo ""

# =============================================================
# PHASE 3: Monitor Pipeline and Handle Failures
# =============================================================

echo "============================================="
echo "PHASE 3: Monitoring Pipeline"
echo "============================================="
echo ""

# Wait for merge job (which means Stage A completed)
echo "Waiting for Stage A to complete..."
if ! wait_for_job "$STAGE_A_JOB" "Stage_A" 86400; then  # 24 hours max
    echo ""
    echo "============================================="
    echo "FATAL ERROR: Stage A failed"
    echo "============================================="
    echo ""
    echo "Check logs:"
    echo "  tail -100 logs/stage_a_sdpb_${STAGE_A_JOB}_*.log"
    echo "  sacct -j $STAGE_A_JOB --format=JobID,State,ExitCode,Elapsed,MaxRSS"
    echo ""
    CLEANUP_NEEDED=true
    exit 1
fi

echo "Stage A completed successfully!"
echo ""

# Wait for merge job
echo "Waiting for merge + Stage B submission..."
if ! wait_for_job "$MERGE_JOB" "Merge" 3600; then  # 1 hour max
    echo ""
    echo "============================================="
    echo "FATAL ERROR: Merge job failed"
    echo "============================================="
    echo ""
    echo "Check log:"
    echo "  tail -100 logs/merge_a_submit_b_${MERGE_JOB}.log"
    echo ""
    CLEANUP_NEEDED=true
    exit 1
fi

echo "Merge completed successfully!"
echo ""

# Extract Stage B and plot job IDs from merge log
MERGE_LOG="logs/merge_a_submit_b_${MERGE_JOB}.log"
if [ -f "$MERGE_LOG" ]; then
    STAGE_B_JOB=$(grep "^Stage B job:" "$MERGE_LOG" | awk '{print $NF}' || echo "")
    PLOT_JOB=$(grep "^Final plot job:" "$MERGE_LOG" | awk '{print $NF}' || echo "")

    if [ -n "$STAGE_B_JOB" ]; then
        ALL_JOBS+=("$STAGE_B_JOB")
        echo "Stage B job ID: $STAGE_B_JOB"
    fi

    if [ -n "$PLOT_JOB" ]; then
        ALL_JOBS+=("$PLOT_JOB")
        echo "Plot job ID: $PLOT_JOB"
    fi
    echo ""
fi

# Wait for Stage B
if [ -n "$STAGE_B_JOB" ]; then
    echo "Waiting for Stage B to complete..."
    if ! wait_for_job "$STAGE_B_JOB" "Stage_B" 86400; then  # 24 hours max
        echo ""
        echo "============================================="
        echo "FATAL ERROR: Stage B failed"
        echo "============================================="
        echo ""
        echo "Check logs:"
        echo "  tail -100 logs/stage_b_sdpb_${STAGE_B_JOB}_*.log"
        echo "  sacct -j $STAGE_B_JOB --format=JobID,State,ExitCode,Elapsed,MaxRSS"
        echo ""
        CLEANUP_NEEDED=true
        exit 1
    fi
    echo "Stage B completed successfully!"
    echo ""
else
    echo "WARNING: Could not find Stage B job ID in merge log"
    echo "Pipeline may have failed at merge stage"
fi

# Wait for plot generation
if [ -n "$PLOT_JOB" ]; then
    echo "Waiting for final plot generation..."
    if ! wait_for_job "$PLOT_JOB" "Plot" 600; then  # 10 minutes max
        echo ""
        echo "WARNING: Plot generation failed"
        echo "Check log:"
        echo "  tail -100 logs/final_merge_and_plot_${PLOT_JOB}.log"
        echo ""
        # Don't exit - plot failure is not critical
    else
        echo "Plot generated successfully!"
        echo ""
    fi
else
    echo "WARNING: Could not find plot job ID in merge log"
fi

# =============================================================
# PHASE 4: Final Validation and Summary
# =============================================================

echo "============================================="
echo "PHASE 4: Final Validation"
echo "============================================="
echo ""

# Check final outputs
VALIDATION_FAILED=false

echo "Checking outputs..."
echo ""

if [ -f "data/eps_bound.csv" ]; then
    n_rows=$(tail -n +2 data/eps_bound.csv | wc -l | tr -d ' ')
    echo "✓ Stage A output: $n_rows data points"
else
    echo "✗ Stage A output missing: data/eps_bound.csv"
    VALIDATION_FAILED=true
fi

if [ -f "data/epsprime_bound.csv" ]; then
    n_rows=$(tail -n +2 data/epsprime_bound.csv | wc -l | tr -d ' ')
    echo "✓ Stage B output: $n_rows data points"
else
    echo "✗ Stage B output missing: data/epsprime_bound.csv"
    VALIDATION_FAILED=true
fi

if [ -f "figures/fig6_reproduction.png" ]; then
    echo "✓ Figure 6 plot: figures/fig6_reproduction.png"
else
    echo "✗ Figure 6 plot missing: figures/fig6_reproduction.png"
    VALIDATION_FAILED=true
fi

if [ -f "figures/fig6_reproduction.pdf" ]; then
    echo "✓ Figure 6 PDF: figures/fig6_reproduction.pdf"
fi

echo ""

# =============================================================
# Final Summary
# =============================================================

echo "============================================="
echo "OVERNIGHT PIPELINE COMPLETE"
echo "============================================="
echo "Finished: $(date)"
echo ""

if [ "$VALIDATION_FAILED" = true ]; then
    echo "⚠ WARNING: Some outputs are missing"
    echo "Review the logs to identify issues"
    echo ""
else
    echo "✓ All outputs generated successfully!"
    echo ""
fi

echo "Configuration used:"
echo "  SDPB timeout: ${PROD_TIMEOUT}s"
echo "  CPUs: $PROD_CPUS"
echo "  Memory: $PROD_MEM"
echo "  Walltime: $PROD_WALLTIME"
echo ""

echo "Outputs:"
echo "  Stage A: data/eps_bound.csv"
echo "  Stage B: data/epsprime_bound.csv"
echo "  Plot:    figures/fig6_reproduction.png"
echo "  Plot:    figures/fig6_reproduction.pdf"
echo ""

echo "Full log: $LOG_FILE"
echo ""

if [ "$VALIDATION_FAILED" = false ]; then
    echo "🎉 SUCCESS! Check figures/fig6_reproduction.png for your result!"
else
    echo "⚠ PARTIAL SUCCESS: Pipeline completed but some outputs missing"
fi

echo ""
echo "============================================="

# Don't cleanup jobs on successful completion
CLEANUP_NEEDED=false
