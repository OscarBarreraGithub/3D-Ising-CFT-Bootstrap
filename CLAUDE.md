# 3D Ising CFT Bootstrap

## Project Overview

Reproducing Figure 6 from arXiv:1203.6064 ("Solving the 3D Ising Model with the Conformal Bootstrap") using conformal bootstrap and linear programming.

## Development Setup

- **Conda env**: `conda activate ising_bootstrap`
- **Install**: `pip install -e .[dev]`
- **Test**: `pytest tests/`

## Key Files

| Path | Purpose |
|------|---------|
| `src/ising_bootstrap/config.py` | Physical constants (D=3, n_max=10) |
| `src/ising_bootstrap/blocks/` | Conformal block computation at z=z̄=1/2 |
| `src/ising_bootstrap/spectrum/` | Table 2 discretization (T1-T5) |
| `src/ising_bootstrap/lp/solver.py` | LP feasibility (scipy + SDPB backends) |
| `src/ising_bootstrap/lp/sdpb.py` | SDPB integration (PMP writer, subprocess runner) |
| `src/ising_bootstrap/scans/` | Stage A and B bootstrap scans |
| `src/ising_bootstrap/plot/fig6.py` | Figure 6 generation |
| `tools/sdpb-3.1.0.sif` | SDPB Singularity container (gitignored) |

## LP Solver

Production runs use **SDPB** (arbitrary-precision SDP solver) via `--backend sdpb`.
The scipy/HiGHS backend fails at n_max=10 due to float64 conditioning (condition number ~4e16).
See `docs/LP_CONDITIONING_BUG.md` for the full diagnosis and fix.

## Critical Parameters

- `n_max = 10` → 66 index pairs (m,n) with m odd, m+2n ≤ 21
- Crossing point: z = z̄ = 1/2 (u = v = 1/4)
- Expected results at Δσ ≈ 0.5182: Δε ≈ 1.41, Δε' ≈ 3.84

## Code Standards

- Use `mpmath` for extended precision (50+ decimal digits minimum)
- Follow Table 2 discretization from paper **exactly**
- Cache block derivatives to `data/cached_blocks/`
- Normalization: Dolan-Osborn convention (critical for matching paper results)

## Common Pitfalls

1. **Block normalization**: Must use Dolan-Osborn convention
2. **Derivative indexing**: m + 2n ≤ 21, not total order
3. **Discretization**: Table 2 must be followed literally
4. **LP conditioning**: scipy fails at n_max=10 — use SDPB backend
