#!/bin/bash
# Merge per-task Stage A CSV files into a single sorted output.
#
# Usage: bash jobs/merge_stage_a.sh
#
# Input:  data/eps_bound_0.csv, data/eps_bound_1.csv, ..., data/eps_bound_50.csv
# Output: data/eps_bound.csv (sorted by delta_sigma)

set -e

OUTPUT="data/eps_bound.csv"
TEMP="${OUTPUT}.tmp"

echo "=== Merging Stage A results ==="

# Write header
echo "delta_sigma,delta_eps_max" > "${TEMP}"

# Count available files
n_files=0
n_missing=0
for i in $(seq 0 50); do
    f="data/eps_bound_${i}.csv"
    if [ -f "$f" ]; then
        # Append data rows (skip header)
        tail -n +2 "$f" >> "${TEMP}"
        n_files=$((n_files + 1))
    else
        echo "  WARNING: Missing data/eps_bound_${i}.csv"
        n_missing=$((n_missing + 1))
    fi
done

# Sort by delta_sigma (column 1, numeric)
head -1 "${TEMP}" > "${OUTPUT}"
tail -n +2 "${TEMP}" | sort -t',' -k1 -n >> "${OUTPUT}"
rm "${TEMP}"

n_rows=$(tail -n +2 "${OUTPUT}" | wc -l | tr -d ' ')

echo "  Files merged: ${n_files}/51"
echo "  Missing: ${n_missing}"
echo "  Data rows: ${n_rows}"
echo "  Output: ${OUTPUT}"

if [ "$n_missing" -gt 0 ]; then
    echo ""
    echo "  To re-submit missing tasks:"
    missing=""
    for i in $(seq 0 50); do
        [ ! -f "data/eps_bound_${i}.csv" ] && missing="${missing},${i}"
    done
    missing="${missing:1}"  # remove leading comma
    echo "    sbatch --array=${missing} jobs/stage_a.slurm"
fi

echo ""
echo "Preview:"
head -5 "${OUTPUT}"
