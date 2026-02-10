#!/bin/bash
#SBATCH --job-name=ising_merge_a
#SBATCH --account=randall_lab
#SBATCH --partition=shared
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/merge_stage_a_%j.log
#SBATCH --error=logs/merge_stage_a_%j.log

# Wrapper SLURM job for merging Stage A results.
# Used in dependency chains so merge runs automatically after Stage A array completes.

set -e

echo "=== Merge Stage A ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

cd "$SLURM_SUBMIT_DIR"
bash jobs/merge_stage_a.sh

echo ""
echo "Merge complete: $(date)"
