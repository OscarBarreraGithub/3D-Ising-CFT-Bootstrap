# Running the 3D Ising Bootstrap Pipeline

This document explains how to run the full pipeline to reproduce Figure 6 of arXiv:1203.6064.

---

## Prerequisites

### Python Environment
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Verify Installation
```bash
python -c "from ising_bootstrap import config; print(f'n_max = {config.N_MAX}')"
# Should print: n_max = 10
```

---

## Quick Test (5-10 minutes)

Run a fast test with reduced discretization to verify the pipeline works:

```bash
# Stage A (reduced, ~2 min)
python -m ising_bootstrap.scans.stage_a \
    --sigma-min 0.51 --sigma-max 0.53 --sigma-step 0.005 \
    --reduced \
    --output data/eps_bound_test.csv

# Stage B (reduced, ~3 min)
python -m ising_bootstrap.scans.stage_b \
    --eps-bound data/eps_bound_test.csv \
    --reduced \
    --output data/epsprime_bound_test.csv

# Quick plot
python -m ising_bootstrap.plot.fig6 \
    --data data/epsprime_bound_test.csv \
    --output figures/fig6_test.png
```

---

## Full Production Run

### Overview

| Stage | Description | Estimated Time | Output |
|-------|-------------|----------------|--------|
| A | Compute Δε_max(Δσ) | 2-4 hours | data/eps_bound.csv |
| B | Compute Δε'_max(Δσ) | 4-8 hours | data/epsprime_bound.csv |
| Plot | Generate figure | < 1 minute | figures/fig6_reproduction.png |

Times are estimates for a modern laptop (M1/M2 Mac or recent Intel i7/i9).

### Stage A: Δε Bound

This computes the upper bound on the first Z2-even scalar dimension Δε as a function of Δσ.

```bash
python -m ising_bootstrap.scans.stage_a \
    --sigma-min 0.50 \
    --sigma-max 0.60 \
    --sigma-step 0.002 \
    --tolerance 1e-4 \
    --output data/eps_bound.csv \
    --verbose
```

**Options:**
- `--sigma-min`, `--sigma-max`: Range of Δσ values (paper uses 0.50 to 0.60)
- `--sigma-step`: Grid spacing (0.002 gives 51 points)
- `--tolerance`: Binary search tolerance for Δε (1e-4 is usually sufficient)
- `--reduced`: Use reduced discretization (T1-T2 only) for faster but less accurate run
- `--verbose`: Print progress information

**Output format** (data/eps_bound.csv):
```csv
delta_sigma,delta_eps_max
0.500,1.0234
0.502,1.0312
...
```

### Stage B: Δε' Bound

This computes the upper bound on Δε' given that Δε is set to its maximal allowed value.

```bash
python -m ising_bootstrap.scans.stage_b \
    --eps-bound data/eps_bound.csv \
    --sigma-min 0.50 \
    --sigma-max 0.60 \
    --tolerance 1e-3 \
    --output data/epsprime_bound.csv \
    --verbose
```

**Options:**
- `--eps-bound`: Path to Stage A output (required)
- `--tolerance`: Binary search tolerance for Δε' (1e-3 is usually sufficient)
- Other options same as Stage A

**Output format** (data/epsprime_bound.csv):
```csv
delta_sigma,delta_eps,delta_eps_prime_max
0.500,1.0234,2.567
0.502,1.0312,2.589
...
```

### Plot Generation

```bash
python -m ising_bootstrap.plot.fig6 \
    --data data/epsprime_bound.csv \
    --output figures/fig6_reproduction.png \
    --dpi 300
```

**Options:**
- `--data`: Path to Stage B output
- `--output`: Output file path (supports .png and .pdf)
- `--dpi`: Resolution for PNG output (default 300)
- `--show`: Display plot interactively before saving

---

## Validation Checks

### Expected Results

At the Ising point (Δσ ≈ 0.5182):
- **Δε_max** ≈ 1.41 (from Stage A)
- **Δε'_max** ≈ 3.84 (from Stage B)

### Sanity Check Script

```bash
python -m ising_bootstrap.validate \
    --eps-bound data/eps_bound.csv \
    --epsprime-bound data/epsprime_bound.csv
```

This will print:
```
=== Validation Results ===

Stage A (Δε bound):
  At Δσ = 0.518: Δε_max = 1.412  (expected ~1.41) ✓

Stage B (Δε' bound):
  At Δσ = 0.518: Δε'_max = 3.842  (expected ~3.84) ✓

Qualitative checks:
  [✓] Spike feature present below Ising Δσ
  [✓] Δε' bound increases for Δσ > 0.52
  [✓] Δε' bound diverges near Δσ = 0.50
```

---

## Parallelization

Grid points are independent, so the scans can be parallelized.

### Using Python multiprocessing

```bash
python -m ising_bootstrap.scans.stage_a \
    --sigma-min 0.50 --sigma-max 0.60 --sigma-step 0.002 \
    --parallel --workers 4 \
    --output data/eps_bound.csv
```

### Manual splitting

Run different Δσ ranges on different machines:
```bash
# Machine 1
python -m ising_bootstrap.scans.stage_a \
    --sigma-min 0.50 --sigma-max 0.55 --sigma-step 0.002 \
    --output data/eps_bound_1.csv

# Machine 2
python -m ising_bootstrap.scans.stage_a \
    --sigma-min 0.55 --sigma-max 0.60 --sigma-step 0.002 \
    --output data/eps_bound_2.csv

# Merge results
cat data/eps_bound_1.csv > data/eps_bound.csv
tail -n +2 data/eps_bound_2.csv >> data/eps_bound.csv
```

---

## Troubleshooting

### LP Solver Issues

**Problem**: LP returns "infeasible" when it should be feasible

**Solutions**:
1. Increase LP tolerance: `--lp-tolerance 1e-6`
2. Check constraint scaling
3. Verify discretization matches Table 2 exactly

### Numerical Precision Issues

**Problem**: Results vary significantly between runs

**Solutions**:
1. Increase mpmath precision: `--precision 80`
2. Use smaller binary search tolerance
3. Check for overflow in block computation

### Memory Issues

**Problem**: Out of memory with full discretization

**Solutions**:
1. Use `--reduced` flag for testing
2. Process in batches
3. Use sparse matrix representation for constraints

### Slow Performance

**Problem**: Stage A/B taking much longer than expected

**Solutions**:
1. Use `--reduced` for initial testing
2. Enable caching: `--cache-dir data/cached_blocks/`
3. Use parallelization
4. Profile with `--profile` flag to identify bottleneck

---

## Output Files Reference

```
data/
├── cached_blocks/           # Precomputed block derivatives
│   ├── block_0.5_0.npy     # G derivatives for (Δ,l)=(0.5,0)
│   └── ...
├── eps_bound.csv            # Stage A output
├── epsprime_bound.csv       # Stage B output
└── eps_bound_test.csv       # Test run output

figures/
├── fig6_reproduction.png    # Main result
└── fig6_test.png            # Test run figure
```

---

## Configuration Options

All parameters can also be set in a config file:

```yaml
# config.yaml
dimension: 3
n_max: 10
sigma_range: [0.50, 0.60]
sigma_step: 0.002
eps_tolerance: 1e-4
epsprime_tolerance: 1e-3
use_reduced_discretization: false
cache_dir: "data/cached_blocks/"
lp_solver: "highs-ds"
lp_tolerance: 1e-7
mpmath_precision: 50
```

Then run:
```bash
python -m ising_bootstrap.scans.stage_a --config config.yaml
```

---

## Reproducing Paper Results Exactly

To match the paper's Fig. 6 as closely as possible:

1. **Use full discretization** (T1-T5 from Table 2)
2. **Use n_max=10** (default)
3. **Use fine Δσ grid**: step of 0.001 or smaller near the spike
4. **Use tight tolerances**: 1e-4 for Δε, 1e-3 for Δε'

Note: The paper used CPLEX; we use HiGHS. Results should be nearly identical but may differ at the 4th decimal place due to different LP implementations.
