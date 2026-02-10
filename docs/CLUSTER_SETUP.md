# FASRC Cluster Setup

Step-by-step instructions for running the 3D Ising CFT Bootstrap on Harvard's FASRC Cannon cluster.

## 1. Environment Setup

```bash
ssh <username>@login.rc.fas.harvard.edu
cd /n/holylabs/schwartz_lab/Lab/<username>/3D-Ising-CFT-Bootstrap

# Create conda environment (on a compute node, not login)
salloc -p test --account=iaifi_lab -c 4 -t 00:30:00 --mem=8G
conda activate ising_bootstrap   # or create with: mamba create -n ising_bootstrap ...
pip install -e .
pytest tests/ -x -q --timeout=60
```

## 2. SDPB Installation

SDPB is the arbitrary-precision solver required for production runs (n_max=10).
It runs via a Singularity container.

```bash
mkdir -p tools
singularity pull tools/sdpb-3.1.0.sif docker://bootstrapcollaboration/sdpb:3.1.0

# Verify
singularity exec tools/sdpb-3.1.0.sif pmp2sdp --help
singularity exec tools/sdpb-3.1.0.sif sdpb --help
```

The `.sif` file is ~300 MB. It contains `pmp2sdp`, `sdpb`, and `mpirun`.

## 3. Directory Layout

```
3D-Ising-CFT-Bootstrap/
├── src/ising_bootstrap/     # Source code
├── tests/                   # Test suite (308+ tests)
├── tools/
│   └── sdpb-3.1.0.sif      # SDPB Singularity container (gitignored)
├── jobs/
│   ├── stage_a_sdpb.slurm  # Stage A with SDPB backend
│   ├── stage_b_sdpb.slurm  # Stage B with SDPB backend
│   ├── check_usage.sh      # Post-job resource analysis
│   ├── merge_stage_a.sh    # Merge per-task CSVs
│   └── merge_stage_b.sh
├── data/                    # Output (gitignored)
│   ├── cached_blocks/       # ~520K .npy block derivative files
│   ├── eps_bound*.csv       # Stage A output
│   └── epsprime_bound*.csv  # Stage B output
├── logs/                    # SLURM logs (gitignored)
└── figures/                 # Plots (gitignored)
```

## 4. SLURM Job Scripts

| Script | Tasks | CPUs | Mem | Time | Description |
|--------|-------|------|-----|------|-------------|
| `stage_a_sdpb.slurm` | 0-50 | 8 | 16G | 8h | Stage A with SDPB, one Delta_sigma per task |
| `stage_b_sdpb.slurm` | 0-50 | 8 | 16G | 8h | Stage B with SDPB, requires Stage A CSV |
| `precompute_array.slurm` | 0-9 | 8 | 8G | 18h | Block derivative precomputation (one-time) |

Submit:
```bash
sbatch jobs/stage_a_sdpb.slurm
# After completion:
bash jobs/merge_stage_a.sh
sbatch jobs/stage_b_sdpb.slurm
bash jobs/merge_stage_b.sh
```

## 5. Resource Usage

After a job completes, check actual resource consumption:
```bash
./jobs/check_usage.sh <JOB_ID>
```

This shows per-task wall time, peak memory, CPU efficiency, and recommendations
for tuning SLURM resource requests.

## 6. Troubleshooting

### "SDPB Singularity image not found"
Pull the container: `singularity pull tools/sdpb-3.1.0.sif docker://bootstrapcollaboration/sdpb:3.1.0`

### "ModuleNotFoundError: No module named 'ising_bootstrap'"
Install the package: `conda activate ising_bootstrap && pip install -e .`

### Missing task outputs
```bash
for i in $(seq 0 50); do
    [ ! -f data/eps_bound_${i}.csv ] && echo "Missing: task $i"
done
# Resubmit just the missing ones:
sbatch --array=3,17,42 jobs/stage_a_sdpb.slurm
```

### Jobs pending
Check fairshare: `sshare -u $USER`
Try test partition: `salloc -p test --account=iaifi_lab -c 4 -t 01:00:00 --mem=8G`
