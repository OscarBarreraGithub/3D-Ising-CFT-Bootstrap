#!/bin/bash
# Merge pilot Stage A CSV files into one sorted output.
#
# Usage:
#   bash jobs/merge_stage_a_pilot.sh
#
# Input pattern:
#   data/eps_bound_pilot_*.csv
# Output:
#   data/eps_bound_pilot.csv

set -euo pipefail

OUTPUT="data/eps_bound_pilot.csv"
TEMP="${OUTPUT}.tmp"

echo "=== Merging Stage A pilot results ==="
echo "delta_sigma,delta_eps_max" > "${TEMP}"

n_files=0
for f in data/eps_bound_pilot_*.csv; do
    if [ -f "$f" ]; then
        tail -n +2 "$f" >> "${TEMP}"
        n_files=$((n_files + 1))
    fi
done

if [ "$n_files" -eq 0 ]; then
    echo "ERROR: No pilot files found matching data/eps_bound_pilot_*.csv"
    rm -f "${TEMP}"
    exit 1
fi

head -1 "${TEMP}" > "${OUTPUT}"
tail -n +2 "${TEMP}" | sort -t',' -k1 -n >> "${OUTPUT}"
rm "${TEMP}"

n_rows=$(tail -n +2 "${OUTPUT}" | wc -l | tr -d ' ')
echo "  Files merged: ${n_files}"
echo "  Data rows: ${n_rows}"
echo "  Output: ${OUTPUT}"
echo ""
head -n 10 "${OUTPUT}"
