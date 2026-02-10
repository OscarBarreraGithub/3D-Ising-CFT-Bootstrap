#!/bin/bash
#
# Real-time monitoring dashboard for 3D Ising bootstrap pipeline
# Usage: ./monitor_jobs.sh [JOB_ID] [--refresh SECONDS] [--once]
#

set -e

# Color codes
RUNNING_COLOR="\033[0;36m"    # Cyan
COMPLETED_COLOR="\033[0;32m"  # Green
FAILED_COLOR="\033[0;31m"     # Red
WARNING_COLOR="\033[0;33m"    # Yellow
RESET_COLOR="\033[0m"

# Default values
REFRESH_INTERVAL=10
ONCE_MODE=0
JOB_ID=""

# Project directories
PROJECT_DIR="/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap"
DATA_DIR="$PROJECT_DIR/data"
LOG_DIR="$PROJECT_DIR/logs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --refresh)
            REFRESH_INTERVAL="$2"
            shift 2
            ;;
        --once)
            ONCE_MODE=1
            shift
            ;;
        --help)
            echo "Usage: $0 [JOB_ID] [--refresh SECONDS] [--once]"
            echo ""
            echo "Options:"
            echo "  JOB_ID         Job ID to monitor (default: most recent)"
            echo "  --refresh N    Auto-refresh interval in seconds (default: 10)"
            echo "  --once         Single run, no loop"
            echo "  --help         Show this help"
            exit 0
            ;;
        *)
            JOB_ID="$1"
            shift
            ;;
    esac
done

# Function to get most recent job ID if not provided
get_recent_job_id() {
    squeue -u $USER -h -o "%i" | head -1
}

# Function to detect stage from job
detect_stage() {
    local job_id=$1
    # Check for log files to determine stage
    if ls "$LOG_DIR/stage_a_sdpb_${job_id}_"*.log &>/dev/null; then
        echo "a"
    elif ls "$LOG_DIR/stage_b_sdpb_${job_id}_"*.log &>/dev/null; then
        echo "b"
    else
        echo "a"  # Default to stage a
    fi
}

# Function to get SLURM job status
get_job_status() {
    local job_id=$1

    # Check if job exists in queue
    if ! squeue -j "$job_id" &>/dev/null; then
        echo "NOT_IN_QUEUE"
        return
    fi

    # Count running/pending tasks
    local running=$(squeue -j "$job_id" -h -t RUNNING | wc -l)
    local pending=$(squeue -j "$job_id" -h -t PENDING | wc -l)
    local completing=$(squeue -j "$job_id" -h -t COMPLETING | wc -l)

    echo "RUNNING:$running PENDING:$pending COMPLETING:$completing"
}

# Function to count valid result files
count_results() {
    local stage=$1
    local prefix=""

    if [[ "$stage" == "a" ]]; then
        prefix="eps_bound"
    else
        prefix="epsprime_bound"
    fi

    local total=0
    local valid=0
    local empty=0

    for file in "$DATA_DIR/${prefix}_"*.csv; do
        if [[ -f "$file" ]]; then
            ((total++))
            local size=$(stat -c%s "$file" 2>/dev/null || echo 0)
            if [[ $size -gt 100 ]]; then
                ((valid++))
            else
                ((empty++))
            fi
        fi
    done

    echo "TOTAL:$total VALID:$valid EMPTY:$empty"
}

# Function to scan for errors in logs
scan_errors() {
    local job_id=$1
    local stage=$2

    local oom_count=0
    local mpi_count=0

    # Count OOM kills
    oom_count=$(grep -l "signal 9\|Killed\|oom" "$LOG_DIR/stage_${stage}_sdpb_${job_id}_"*.log 2>/dev/null | wc -l)

    # Count MPI errors
    mpi_count=$(grep -l "mpirun.*error\|not enough slots" "$LOG_DIR/stage_${stage}_sdpb_${job_id}_"*.log 2>/dev/null | wc -l)

    echo "OOM:$oom_count MPI:$mpi_count"
}

# Function to draw progress bar
draw_progress_bar() {
    local current=$1
    local total=$2
    local width=30

    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    echo -n "["
    for ((i=0; i<filled; i++)); do echo -n "█"; done
    for ((i=0; i<empty; i++)); do echo -n "░"; done
    echo -n "] $current/$total ($percent%)"
}

# Function to display dashboard
display_dashboard() {
    local job_id=$1
    local stage=$2

    # Clear screen (only if not in once mode)
    if [[ $ONCE_MODE -eq 0 ]]; then
        clear
    fi

    # Header
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│ 3D Ising Bootstrap Pipeline Monitor                     │"
    echo "│ Job: $job_id  Stage: $(echo $stage | tr '[:lower:]' '[:upper:]')  Time: $(date '+%Y-%m-%d %H:%M:%S')                    │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo ""

    # SLURM status
    local job_status=$(get_job_status "$job_id")
    echo "SLURM STATUS:"

    if [[ "$job_status" == "NOT_IN_QUEUE" ]]; then
        echo -e "  ${WARNING_COLOR}Job not found in queue (completed or cancelled?)${RESET_COLOR}"
    else
        local running=$(echo "$job_status" | grep -oP 'RUNNING:\K\d+')
        local pending=$(echo "$job_status" | grep -oP 'PENDING:\K\d+')
        local completing=$(echo "$job_status" | grep -oP 'COMPLETING:\K\d+')

        echo -e "  ${RUNNING_COLOR}Running:${RESET_COLOR}    $running tasks"
        echo -e "  ${WARNING_COLOR}Pending:${RESET_COLOR}    $pending tasks"
        if [[ $completing -gt 0 ]]; then
            echo -e "  ${COMPLETED_COLOR}Completing:${RESET_COLOR} $completing tasks"
        fi
    fi
    echo ""

    # Result file status
    local result_status=$(count_results "$stage")
    local total_files=$(echo "$result_status" | grep -oP 'TOTAL:\K\d+')
    local valid_files=$(echo "$result_status" | grep -oP 'VALID:\K\d+')
    local empty_files=$(echo "$result_status" | grep -oP 'EMPTY:\K\d+')

    echo "STAGE $(echo $stage | tr '[:lower:]' '[:upper:]') PROGRESS: $(draw_progress_bar $valid_files 51)"
    echo -e "  ${COMPLETED_COLOR}Valid results:${RESET_COLOR}  $valid_files files"

    if [[ $empty_files -gt 0 ]]; then
        echo -e "  ${FAILED_COLOR}Empty results:${RESET_COLOR}  $empty_files files ⚠"
    fi

    local missing=$((51 - total_files))
    if [[ $missing -gt 0 ]]; then
        echo -e "  ${WARNING_COLOR}Missing:${RESET_COLOR}        $missing files"
    fi
    echo ""

    # Error detection
    local errors=$(scan_errors "$job_id" "$stage")
    local oom_errors=$(echo "$errors" | grep -oP 'OOM:\K\d+')
    local mpi_errors=$(echo "$errors" | grep -oP 'MPI:\K\d+')

    echo "ERROR DETECTION:"
    if [[ $oom_errors -gt 0 ]]; then
        echo -e "  ${FAILED_COLOR}⚠ OOM kills:${RESET_COLOR}      $oom_errors tasks"
    else
        echo -e "  ${COMPLETED_COLOR}✓ No OOM kills${RESET_COLOR}"
    fi

    if [[ $mpi_errors -gt 0 ]]; then
        echo -e "  ${FAILED_COLOR}⚠ MPI errors:${RESET_COLOR}     $mpi_errors tasks"
    else
        echo -e "  ${COMPLETED_COLOR}✓ No MPI errors${RESET_COLOR}"
    fi
    echo ""

    # Quick actions
    echo "QUICK ACTIONS:"
    echo "  View log:       tail -f $LOG_DIR/stage_${stage}_sdpb_${job_id}_0.log"
    echo "  Analyze logs:   python scripts/analyze_logs.py --job $job_id --stage $stage"
    echo "  Validate:       python scripts/validate_results.py --stage $stage"
    echo "  Cancel job:     scancel $job_id"
    echo ""

    # Auto-refresh message
    if [[ $ONCE_MODE -eq 0 ]]; then
        echo "Auto-refreshing in ${REFRESH_INTERVAL}s... (Ctrl+C to stop)"
    fi
}

# Main loop
main() {
    # Get job ID if not provided
    if [[ -z "$JOB_ID" ]]; then
        JOB_ID=$(get_recent_job_id)
        if [[ -z "$JOB_ID" ]]; then
            echo "Error: No job ID provided and no jobs found in queue"
            exit 1
        fi
        echo "Monitoring most recent job: $JOB_ID"
        sleep 2
    fi

    # Detect stage
    STAGE=$(detect_stage "$JOB_ID")

    # Run loop
    while true; do
        display_dashboard "$JOB_ID" "$STAGE"

        if [[ $ONCE_MODE -eq 1 ]]; then
            break
        fi

        sleep "$REFRESH_INTERVAL"
    done
}

main
