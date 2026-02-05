# FASRC Cluster Deployment Guide

Step-by-step instructions for running the 3D Ising CFT Bootstrap on Harvard's FASRC Cannon cluster.

## Prerequisites

- FASRC account with access to `iaifi_lab`
- SSH access: `ssh <username>@login.rc.fas.harvard.edu`
- Miniforge3 installed at `$HOME/miniforge3` (or system conda)

## 1. One-Time Setup

### 1a. SSH to cluster and set up working directory

```bash
ssh <username>@login.rc.fas.harvard.edu

# Set up working directory on netscratch
WORKDIR="/n/netscratch/schwartz_lab/Everyone/${USER}"
mkdir -p $WORKDIR
cd $WORKDIR
```

### 1b. Clone the repository

```bash
git clone https://github.com/OscarBarreraGithub/3D-Ising-CFT-Bootstrap.git ising_bootstrap
cd ising_bootstrap
```

### 1c. Create conda environment

Request an interactive node first (don't build on login nodes):

```bash
salloc -p test --account=iaifi_lab -c 4 -t 00:30:00 --mem=8G
```

Then create the environment:

```bash
# Source conda
source $HOME/miniforge3/etc/profile.d/conda.sh

# Create environment
mamba create -n ising_bootstrap -c conda-forge python=3.11 numpy scipy mpmath matplotlib -y

# Activate and install the package
conda activate ising_bootstrap
pip install -e .

# Verify
python -c "from ising_bootstrap.config import N_MAX; print(f'n_max = {N_MAX}')"
# Should print: n_max = 10

# Run a quick test
pytest tests/ -x -q --timeout=60
```

### 1d. Create log directory

```bash
mkdir -p logs
```

## 2. Quick Sanity Check

Before submitting batch jobs, verify the pipeline works interactively:

```bash
# Still on the salloc node from setup
conda activate ising_bootstrap

# Run a 3-point reduced scan (~2-5 min)
python -m ising_bootstrap.scans.stage_a \
    --sigma-min 0.51 --sigma-max 0.53 --sigma-step 0.01 \
    --reduced --verbose --output data/eps_bound_test.csv
```

**Expected output:**
- 3 rows in `data/eps_bound_test.csv`
- Each `delta_eps_max` in range [0.5, 2.5]
- Completes in under 5 minutes

```bash
# Check results
cat data/eps_bound_test.csv
```

## 3. Production Stage A Scan

The production run has two phases: block precomputation (one-time, expensive) and the scan itself (parallelized via SLURM array job).

### 3a. Precompute block derivatives

This computes the extended H arrays for all ~520K operators in the spectrum. With 8 workers in a single job, this takes ~14-16 hours. With 10 shards (`precompute_array.slurm`), each shard takes ~5-9 hours depending on cache state. Observed throughput: ~5,800 operators/hour per shard.

```bash
sbatch jobs/precompute.slurm
```

Monitor progress:

```bash
squeue -u $USER
tail -f logs/precompute_*.log
```

**Expected**: ~520K blocks computed, saved to `data/cached_blocks/ext_*.npy`

### 3b. Run Stage A array job

Once precomputation is done, run the scan across all 51 grid points in parallel:

```bash
sbatch jobs/stage_a.slurm
```

This submits 51 independent tasks (array indices 0-50), each computing one Δσ point. Each task takes ~5-30 minutes depending on the grid point.

Monitor:

```bash
squeue -u $USER
# Check individual task output
tail -f logs/stage_a_*_17.log  # Task 17 = Δσ ≈ 0.534
```

### 3c. Merge results

After all 51 tasks complete:

```bash
bash jobs/merge_stage_a.sh
```

This creates `data/eps_bound.csv` with 51 rows sorted by Δσ.

### 3d. Validate

```bash
# Check row count (should be 52: 1 header + 51 data)
wc -l data/eps_bound.csv

# Preview the data
head -20 data/eps_bound.csv

# Check near Ising point (Δσ ≈ 0.518)
grep "0.518" data/eps_bound.csv
# Expected: delta_eps_max ≈ 1.41
```

## 4. Resource Estimates

| Job | Partition | Time | Memory | CPUs | Notes |
|-----|-----------|------|--------|------|-------|
| Precompute | shared | 10h | 8G | 8 | 10 shards; ~5,800 ops/hr/shard |
| Stage A (per task) | shared | 1h | 4G | 1 | 51 array tasks |
| Quick test | test | 20m | 4G | 1 | Interactive salloc |

Total cluster time for Stage A: ~50 core-hours (but wall-clock ~1 hour with 51 parallel tasks).

## 5. File Layout on Cluster

```
$WORKDIR/ising_bootstrap/
├── src/ising_bootstrap/          # Source code
├── tests/                        # Test suite (277 tests)
├── data/
│   ├── cached_blocks/            # ~520K .npy files (~1 GB)
│   │   ├── ext_d0.50000000_l0.npy
│   │   ├── ext_d0.50002000_l0.npy
│   │   └── ...
│   ├── eps_bound_0.csv           # Per-task output
│   ├── eps_bound_1.csv
│   ├── ...
│   └── eps_bound.csv             # Merged final output
├── jobs/                         # SLURM scripts
│   ├── precompute.slurm
│   ├── stage_a.slurm
│   └── merge_stage_a.sh
└── logs/                         # Job logs
    ├── precompute_12345.log
    ├── stage_a_12346_0.log
    └── ...
```

## 6. Troubleshooting

### "ModuleNotFoundError: No module named 'ising_bootstrap'"

The package isn't installed. Re-run:
```bash
conda activate ising_bootstrap
pip install -e .
```

### Jobs stuck in PENDING

Check your account's allocation:
```bash
sacct --account=iaifi_lab --starttime=2024-01-01 --format=JobID,Partition,State,Elapsed
sshare -u $USER
```

Try the `test` partition for quick debugging:
```bash
salloc -p test --account=iaifi_lab -c 1 -t 00:20:00 --mem=4G
```

### Memory errors during precompute

The precompute job uses ~2-4 GB with 8 workers. If you see OOM errors:
```bash
# Reduce workers or increase memory
sbatch --mem=16G jobs/precompute.slurm
```

### Precompute interrupted — how to resume

The precompute function has `skip_existing=True` by default. Just re-submit the same job — it will skip already-cached blocks:
```bash
sbatch jobs/precompute.slurm
```

### Thread oversubscription warnings

All job scripts set `OMP_NUM_THREADS=1` to prevent numpy/scipy from spawning extra threads. If you see warnings about thread contention, ensure these are set:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

### Missing task outputs

Check which array tasks failed:
```bash
# List expected vs actual output files
for i in $(seq 0 50); do
    [ ! -f data/eps_bound_${i}.csv ] && echo "Missing: task $i"
done
```

Re-submit just the missing tasks:
```bash
sbatch --array=3,17,42 jobs/stage_a.slurm
```

## 7. Next Steps

After Stage A completes and validates:

1. **Stage B scan** — compute Δε' bounds using Stage A output
   ```bash
   sbatch jobs/stage_b.slurm  # (to be created)
   ```

2. **Plotting** — generate Figure 6 reproduction
   ```bash
   python -m ising_bootstrap.plot.fig6 --data data/epsprime_bound.csv
   ```
