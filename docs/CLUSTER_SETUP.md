# Cluster Setup Guide

This document explains how to set up and run the 3D Ising bootstrap pipeline on a computing cluster.

---

## Overview

The computation is embarrassingly parallel over Δσ grid points. Each Stage A/B binary search at a given Δσ is independent, making this ideal for cluster parallelization.

**Estimated compute time (full run):**
- Stage A: ~2-4 hours on a laptop, ~30 min with 8 cores
- Stage B: ~4-8 hours on a laptop, ~1 hour with 8 cores

---

## Prerequisites

### Required Software
- Python 3.11 (pinned for reproducibility)
- pip or conda

### Required Python Packages
Defined in `pyproject.toml`, pinned versions in `requirements.lock`.

---

## Installation on Cluster

### Option 1: Conda Environment (Recommended)

```bash
# The environment.yml references pyproject.toml
conda env create -f environment.yml
conda activate ising_bootstrap
# Done - package is installed in editable mode with all dependencies
```

### Option 2: Reproducible pip Install (Lock File)

Use this for guaranteed reproducibility across machines:

```bash
# Load Python module (cluster-specific)
module load python/3.11.0-fasrc01  # FASRC example

# Create venv
python -m venv .venv
source .venv/bin/activate

# Install exact pinned versions
pip install -r requirements.lock

# Install package (no deps - already installed from lock)
pip install -e . --no-deps
```

### Option 3: Standard pip Install

```bash
module load python/3.11.0-fasrc01

python -m venv .venv
source .venv/bin/activate

# Install package with dependencies
pip install -e .
```

### Option 4: User Installation (No Admin Rights)

```bash
pip install --user -r requirements.lock
pip install --user -e . --no-deps
```

---

## Directory Structure for Cluster

```
$SCRATCH/ising_bootstrap/
├── code/                    # Clone of this repository
│   └── ...
├── data/                    # Shared output directory
│   ├── eps_bound.csv
│   ├── epsprime_bound.csv
│   └── cached_blocks/
├── jobs/                    # Job scripts
│   ├── stage_a.slurm
│   └── stage_b.slurm
├── logs/                    # Job output logs
└── results/                 # Final aggregated results
```

---

## SLURM Job Scripts

### Stage A: Δε Bound

Create `jobs/stage_a.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=ising_stage_a
#SBATCH --output=logs/stage_a_%A_%a.out
#SBATCH --error=logs/stage_a_%A_%a.err
#SBATCH --array=0-50              # 51 Δσ values (0.50 to 0.60, step 0.002)
#SBATCH --time=01:00:00           # 1 hour per task
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Load environment
source /path/to/venv/bin/activate
# OR: conda activate ising_bootstrap

# Calculate Δσ for this array task
SIGMA_MIN=0.50
SIGMA_STEP=0.002
SIGMA=$(python -c "print(${SIGMA_MIN} + ${SLURM_ARRAY_TASK_ID} * ${SIGMA_STEP})")

# Run single Δσ point
python -m ising_bootstrap.scans.stage_a \
    --sigma-min $SIGMA \
    --sigma-max $SIGMA \
    --sigma-step 1.0 \
    --output data/eps_bound_${SLURM_ARRAY_TASK_ID}.csv

echo "Completed Δσ = $SIGMA"
```

### Stage B: Δε' Bound

Create `jobs/stage_b.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=ising_stage_b
#SBATCH --output=logs/stage_b_%A_%a.out
#SBATCH --error=logs/stage_b_%A_%a.err
#SBATCH --array=0-50
#SBATCH --time=02:00:00           # 2 hours per task (Stage B is slower)
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

source /path/to/venv/bin/activate

SIGMA_MIN=0.50
SIGMA_STEP=0.002
SIGMA=$(python -c "print(${SIGMA_MIN} + ${SLURM_ARRAY_TASK_ID} * ${SIGMA_STEP})")

python -m ising_bootstrap.scans.stage_b \
    --eps-bound data/eps_bound.csv \
    --sigma-min $SIGMA \
    --sigma-max $SIGMA \
    --sigma-step 1.0 \
    --output data/epsprime_bound_${SLURM_ARRAY_TASK_ID}.csv

echo "Completed Δσ = $SIGMA"
```

### Merge Results Script

Create `jobs/merge_results.sh`:

```bash
#!/bin/bash

# Merge Stage A results
echo "delta_sigma,delta_eps_max" > data/eps_bound.csv
for f in data/eps_bound_*.csv; do
    tail -n +2 "$f" >> data/eps_bound.csv
done
sort -t',' -k1 -n data/eps_bound.csv -o data/eps_bound.csv

# Merge Stage B results
echo "delta_sigma,delta_eps,delta_eps_prime_max" > data/epsprime_bound.csv
for f in data/epsprime_bound_*.csv; do
    tail -n +2 "$f" >> data/epsprime_bound.csv
done
sort -t',' -k1 -n data/epsprime_bound.csv -o data/epsprime_bound.csv

echo "Merged results written to data/eps_bound.csv and data/epsprime_bound.csv"
```

---

## Running on Cluster

### Step 1: Submit Stage A Jobs

```bash
# Create logs directory
mkdir -p logs

# Submit array job
sbatch jobs/stage_a.slurm

# Monitor progress
squeue -u $USER
```

### Step 2: Merge Stage A Results

After all Stage A jobs complete:

```bash
bash jobs/merge_results.sh
```

### Step 3: Submit Stage B Jobs

```bash
sbatch jobs/stage_b.slurm
```

### Step 4: Merge Final Results and Plot

```bash
bash jobs/merge_results.sh

# Generate figure (can be done locally)
python -m ising_bootstrap.plot.fig6 \
    --data data/epsprime_bound.csv \
    --output figures/fig6_reproduction.png
```

---

## PBS/Torque Job Scripts

If using PBS instead of SLURM:

```bash
#!/bin/bash
#PBS -N ising_stage_a
#PBS -o logs/stage_a.out
#PBS -e logs/stage_a.err
#PBS -t 0-50
#PBS -l walltime=01:00:00
#PBS -l mem=4gb

cd $PBS_O_WORKDIR
source /path/to/venv/bin/activate

SIGMA_MIN=0.50
SIGMA_STEP=0.002
SIGMA=$(python -c "print(${SIGMA_MIN} + ${PBS_ARRAYID} * ${SIGMA_STEP})")

python -m ising_bootstrap.scans.stage_a \
    --sigma-min $SIGMA --sigma-max $SIGMA --sigma-step 1.0 \
    --output data/eps_bound_${PBS_ARRAYID}.csv
```

---

## Caching Block Derivatives

For maximum efficiency, pre-compute block derivatives once and share across jobs:

```bash
# Pre-compute cache (do this once, before main runs)
python -m ising_bootstrap.blocks.precompute_cache \
    --output data/cached_blocks/

# Jobs will automatically use cached values if --cache-dir is specified
python -m ising_bootstrap.scans.stage_a \
    --cache-dir data/cached_blocks/ \
    ...
```

---

## Memory and Performance Notes

### Memory Usage
- Full discretization (T1-T5): ~500MB-1GB per process
- Reduced discretization (T1-T2): ~100MB per process

### Scaling
- Each Δσ point is independent → perfect parallel scaling
- 51 grid points × 8 cores = ~7x speedup (with overhead)

### I/O Considerations
- Block cache is read-only after precomputation → no I/O contention
- Output CSVs are small (<1KB each) → minimal I/O
- Consider using local scratch (`$TMPDIR`) for intermediate files

---

## Troubleshooting

### Job Fails with OOM
- Increase `--mem` in job script
- Use `--reduced` discretization
- Check for memory leaks in block computation

### Numerical Precision Issues
- Increase mpmath precision: `--precision 80`
- Check LP solver tolerance: `--lp-tolerance 1e-8`

### Missing Results
- Check job logs in `logs/` directory
- Verify array task IDs completed
- Re-run failed tasks: `sbatch --array=5,17,32 jobs/stage_a.slurm`

### Cache Corruption
- Delete and regenerate: `rm -rf data/cached_blocks/ && python -m ising_bootstrap.blocks.precompute_cache`
