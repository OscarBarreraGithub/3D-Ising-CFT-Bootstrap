# 3D Ising CFT Bootstrap

## Project Overview

Reproducing Figure 6 from arXiv:1203.6064 ("Solving the 3D Ising Model with the Conformal Bootstrap") using conformal bootstrap and linear programming.

## Development Setup

- **Python version**: 3.11 (required)
- **Install**: `pip install -e .[dev]`
- **Test**: `pytest tests/`
- **Conda**: `conda env create -f environment.yml && conda activate ising_bootstrap`

## Parallel Worktree Workflow

When working on a feature:

1. **Create worktree**: `git worktree add ../ising-{feature} -b feature/{name}`
2. **Setup environment**: `cd ../ising-{feature} && pip install -e .`
3. **Work** in that directory with independent Claude session
4. **When done**: Signal "ready for review" - coordinator will manage review process
5. **After approval**: Merge from main repo and remove worktree

## Code Standards

- Use `mpmath` for extended precision (50+ decimal digits minimum)
- Follow Table 2 discretization from paper **exactly**
- Cache block derivatives to `data/cached_blocks/`
- Normalization: Dolan-Osborn convention (critical for matching paper results)

## Context Management

- Auto-compact triggers at 80% context capacity
- Use `/context` to check current usage
- Key formulas are in `README.md` sections 4-5
- Put persistent instructions here (survives compaction)

## Key Files

| Path | Purpose |
|------|---------|
| `src/ising_bootstrap/config.py` | All physical constants (D=3, n_max=10, etc.) |
| `src/ising_bootstrap/blocks/` | Conformal block computation at z=z̄=1/2 |
| `src/ising_bootstrap/spectrum/` | Table 2 discretization |
| `src/ising_bootstrap/lp/` | Linear programming feasibility solver |
| `src/ising_bootstrap/scans/` | Stage A and B bootstrap scans |
| `README.md` | Full physics specification and algorithm details |

## Critical Parameters

- `n_max = 10` → 66 index pairs (m,n) with m odd, m+2n ≤ 21
- Crossing point: z = z̄ = 1/2 (u = v = 1/4)
- Expected results at Δσ ≈ 0.5182: Δε ≈ 1.41, Δε' ≈ 3.84

## Post-Implementation Checklist

After completing a milestone or significant feature (tests passing, implementation confirmed working):

1. **Run the session-documenter agent** to update `docs/PROGRESS.md`, `docs/TODO.md`, and `docs/SESSION_LOG.md`
2. Do this proactively — don't wait for the user to ask

## Common Pitfalls

1. **Block normalization**: Must use Dolan-Osborn convention
2. **Derivative indexing**: m + 2n ≤ 21, not total order
3. **Discretization**: Table 2 must be followed literally
4. **LP conditioning**: Constraints span many orders of magnitude - scaling required
