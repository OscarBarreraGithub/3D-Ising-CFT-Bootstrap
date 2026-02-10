#!/bin/bash
# Usage: ./jobs/check_usage.sh <JOB_ID>
# Example: ./jobs/check_usage.sh 59675873
#
# Shows per-task resource usage for a completed SLURM array job:
#   - Wall time vs requested time
#   - Peak memory (MaxRSS) vs requested memory
#   - CPU efficiency (CPU time / wall time / n_cpus)

set -euo pipefail

JOB_ID="${1:?Usage: $0 <JOB_ID>}"

echo "=== Resource Usage for Job ${JOB_ID} ==="
echo ""

# Summary across all array tasks
sacct -j "${JOB_ID}" \
    --format="JobID%20,JobName%22,State%12,Elapsed,TotalCPU,MaxRSS,ReqMem,NCPUS,ExitCode" \
    --units=G \
    --parsable2 \
    > /tmp/sacct_raw_${JOB_ID}.txt

# Print header + only the ".batch" steps (actual resource usage, not the job-shell wrapper)
echo "--- Per-task breakdown (batch steps) ---"
printf "%-8s %-12s %10s %12s %12s %8s %8s\n" \
    "TaskID" "State" "WallTime" "CPUTime" "MaxRSS" "ReqMem" "CPUs"
echo "----------------------------------------------------------------------"

sacct -j "${JOB_ID}" \
    --format="JobID%20,State%12,Elapsed%10,TotalCPU%12,MaxRSS%12,ReqMem%8,NCPUS%6" \
    --units=M \
    --noheader \
    --parsable2 \
| grep "\.batch" \
| while IFS='|' read -r jobid state elapsed totalcpu maxrss reqmem ncpus; do
    # Extract array task ID from jobid like "59675873_5.batch"
    task_id=$(echo "$jobid" | sed 's/.*_\([0-9]*\)\.batch/\1/')
    printf "%-8s %-12s %10s %12s %12s %8s %8s\n" \
        "$task_id" "$state" "$elapsed" "$totalcpu" "$maxrss" "$reqmem" "$ncpus"
done

echo ""
echo "--- Aggregate statistics ---"

# Compute stats from sacct
sacct -j "${JOB_ID}" \
    --format="Elapsed,TotalCPU,MaxRSS,NCPUS" \
    --units=M \
    --noheader \
    --parsable2 \
| grep -v "\.extern" \
| grep "\.batch" \
| awk -F'|' '
function to_seconds(t) {
    # Handle D-HH:MM:SS or HH:MM:SS or MM:SS
    n = split(t, parts, ":")
    if (n == 3) {
        # Check for D- prefix
        d = 0
        if (index(parts[1], "-") > 0) {
            split(parts[1], dp, "-")
            d = dp[1]
            parts[1] = dp[2]
        }
        return d*86400 + parts[1]*3600 + parts[2]*60 + parts[3]
    }
    return 0
}
function to_mb(s) {
    # Strip trailing M, G, K etc
    gsub(/[^0-9.]/, "", s)
    return s + 0
}
BEGIN {
    max_wall=0; min_wall=999999; sum_wall=0
    max_mem=0; min_mem=999999; sum_mem=0
    max_cpu=0; sum_cpu=0
    n=0
}
{
    wall = to_seconds($1)
    cpu = to_seconds($2)
    mem = to_mb($3)
    ncpus = $4 + 0

    if (wall > max_wall) max_wall = wall
    if (wall < min_wall) min_wall = wall
    sum_wall += wall

    if (mem > max_mem) max_mem = mem
    if (mem > 0 && mem < min_mem) min_mem = mem
    sum_mem += mem

    if (cpu > max_cpu) max_cpu = cpu
    sum_cpu += cpu

    cpus = ncpus
    n++
}
END {
    if (n == 0) { print "No completed tasks found."; exit }

    avg_wall = sum_wall / n
    avg_mem = sum_mem / n

    printf "Tasks completed:    %d\n", n
    printf "CPUs per task:      %d\n", cpus
    printf "\n"
    printf "Wall time:\n"
    printf "  Min:  %02d:%02d:%02d\n", min_wall/3600, (min_wall%3600)/60, min_wall%60
    printf "  Max:  %02d:%02d:%02d\n", max_wall/3600, (max_wall%3600)/60, max_wall%60
    printf "  Avg:  %02d:%02d:%02d\n", avg_wall/3600, (avg_wall%3600)/60, avg_wall%60
    printf "\n"
    printf "Peak memory (MaxRSS):\n"
    printf "  Min:  %.0f MB\n", min_mem
    printf "  Max:  %.0f MB\n", max_mem
    printf "  Avg:  %.0f MB\n", avg_mem
    printf "\n"
    printf "--- Recommendations ---\n"
    # Recommend 1.5x max wall time, rounded up to nearest hour
    rec_hours = int((max_wall * 1.5) / 3600) + 1
    printf "  --time=%02d:00:00    (1.5x max wall time)\n", rec_hours
    # Recommend 1.3x max memory, rounded up to nearest GB
    rec_mem = int((max_mem * 1.3) / 1024) + 1
    printf "  --mem=%dG           (1.3x max peak memory)\n", rec_mem
}
'

echo ""
echo "--- Failed/timed-out tasks ---"
failed=$(sacct -j "${JOB_ID}" --format="JobID%20,State%15,ExitCode" --noheader \
    | grep -v "\.extern" | grep -v "\.batch" \
    | grep -vE "COMPLETED|RUNNING|PENDING" \
    | head -20)
if [ -z "$failed" ]; then
    echo "  None"
else
    echo "$failed"
fi
