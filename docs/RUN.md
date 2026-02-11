# Running the 3D Ising Bootstrap Pipeline

How to run the full pipeline to reproduce Figure 6 of arXiv:1203.6064.

---

## Prerequisites

```bash
# Conda environment
conda activate ising_bootstrap
python -c "from ising_bootstrap import config; print(f'n_max = {config.N_MAX}')"
# Should print: n_max = 10
```

SDPB backend (required for production n_max=10 runs):
- Singularity container at `tools/sdpb-3.1.0.sif`
- Pull with: `singularity pull tools/sdpb-3.1.0.sif docker://bootstrapcollaboration/sdpb:3.1.0`

---

## Pipeline Overview

| Stage | Description | Backend | Output |
|-------|-------------|---------|--------|
| Precompute | Block derivatives for ~520K operators | CPU only | `data/cached_blocks/ext_*.npy` |
| Stage A | Upper bound on Delta_epsilon(Delta_sigma) | SDPB | `data/eps_bound.csv` |
| Stage B | Upper bound on Delta_epsilon'(Delta_sigma) | SDPB | `data/epsprime_bound.csv` |
| Plot | Generate Figure 6 | matplotlib | `figures/fig6_reproduction.png` |

**Important:** The scipy/HiGHS LP backend fails at n_max=10 due to float64 conditioning
(condition number ~4e16). Production runs **must** use `--backend sdpb`. See
`docs/LP_CONDITIONING_BUG.md` for details.

---

## SLURM Pipeline (FASRC Cannon)

### Stage A with SDPB

```bash
# Submit 51-task array job (one per Delta_sigma point)
sbatch jobs/stage_a_sdpb.slurm
```

Each task runs binary search for Delta_epsilon bound at one Delta_sigma value,
using SDPB with 1024-bit precision and 8 MPI cores.

### Merge Stage A results

```bash
bash jobs/merge_stage_a.sh
# Creates data/eps_bound.csv
```

### Stage B with SDPB

```bash
sbatch jobs/stage_b_sdpb.slurm
```

### Merge and plot

```bash
bash jobs/merge_stage_b.sh
python -m ising_bootstrap.plot.fig6 \
    --data data/epsprime_bound.csv \
    --output figures/fig6_reproduction.png
```

---

## Local / Interactive Runs

### Quick test (scipy backend, low n_max)

```bash
python -m ising_bootstrap.scans.stage_a \
    --sigma-min 0.51 --sigma-max 0.53 --sigma-step 0.005 \
    --reduced --output data/eps_bound_test.csv --verbose
```

### Single-point SDPB test (on a compute node)

```bash
salloc -p test --account=iaifi_lab -c 4 -t 01:00:00 --mem=8G

python -m ising_bootstrap.scans.stage_a \
    --sigma-min 0.518 --sigma-max 0.518 --sigma-step 1.0 \
    --backend sdpb --sdpb-cores 4 --sdpb-timeout 1800 --tolerance 1e-4 \
    --output data/eps_bound_test.csv --verbose
```

---

## CLI Options

Both `stage_a` and `stage_b` accept:

| Flag | Default | Description |
|------|---------|-------------|
| `--sigma-min` | 0.50 | Start of Delta_sigma grid |
| `--sigma-max` | 0.60 | End of Delta_sigma grid |
| `--sigma-step` | 0.002 | Grid spacing |
| `--tolerance` | 1e-4 | Binary search tolerance |
| `--backend` | sdpb | LP backend: `scipy` or `sdpb` |
| `--sdpb-image` | tools/sdpb-3.1.0.sif | Path to SDPB Singularity image |
| `--sdpb-precision` | 1024 | SDPB arithmetic precision in bits |
| `--sdpb-cores` | 4 | MPI cores for SDPB |
| `--sdpb-timeout` | 600 | SDPB timeout in seconds |
| `--reduced` | false | Use T1-T2 only (faster, less accurate) |
| `--output` | - | Output CSV path |
| `--verbose` | false | Print progress |

Stage B additionally requires `--eps-bound <path>` pointing to Stage A output.
Stage B also supports `--eps-snap-tolerance` (default `1e-3`) to anchor `Δε`
to the scalar grid used in Stage B.

---

## Monitoring

```bash
squeue -u $USER                              # Check queue
tail -f logs/stage_a_sdpb_<JOBID>_<TASK>.log # Watch a task
./jobs/check_usage.sh <JOBID>                # Post-job resource usage
```

---

## Validation Targets

| Check | Expected |
|-------|----------|
| Delta_epsilon_max at Delta_sigma ~ 0.518 | ~1.41 |
| Delta_epsilon'_max at Delta_sigma ~ 0.518 | ~3.84 |
| Spike feature below Ising Delta_sigma | Present |
