# Project Plan Summary

## Quick Reference for Claude Instances

This document provides a high-level summary for any Claude instance working on this project.

---

## Project Goal

Reproduce **Figure 6** of arXiv:1203.6064 ("Solving the 3D Ising Model with the Conformal Bootstrap").

**Output**: Upper bound on Δε' vs Δσ showing:
- X-axis: Δσ ∈ [0.50, 0.60]
- Y-axis: Δε' (second Z2-even scalar dimension)
- Key feature: Sharp spike below Ising point (Δσ ≈ 0.5182)
- Validation: Δε' ≈ 3.84 at Ising point with n_max=10

---

## Key Decision Made

**Conformal Block Strategy: Option B (Specialized Evaluator)**

Rationale:
- Paper provides explicit formulas for z=z̄=1/2 (Eqs. 4.10, 4.11, 4.9, C.1)
- Full control over normalization
- No C++ dependencies (pure Python + mpmath)
- See [docs/implementation_choices.md](implementation_choices.md) for details

---

## Implementation Milestones

| # | Milestone | Key Deliverable | Status |
|---|-----------|-----------------|--------|
| 0 | Scaffolding | pyproject.toml, package structure | ✓ |
| 1 | Block Engine | G_{Δ,l}(z) at z=z̄=1/2 with derivatives | ✓ |
| 2 | Discretization | Table 2 (T1-T5), index set (66 terms) | ✓ |
| 3 | LP Builder | Constraint matrix, HiGHS wrapper | ✓ |
| 4 | Stage A | Δε_max(Δσ) scan with binary search | ✓ |
| 5 | Stage B | Δε'_max(Δσ) scan with two-gap assumption | ✓ |
| 6 | Plotting | Figure matching paper's Fig. 6 | ✓ |

See [docs/TODO.md](TODO.md) for detailed task lists.

---

## Critical Formulas (from paper)

### Crossing Equation (Eq. 5.3)
```
u^{Δσ} - v^{Δσ} = Σ p_{Δ,l} [v^{Δσ} G_{Δ,l}(u,v) - u^{Δσ} G_{Δ,l}(v,u)]
```
where p_{Δ,l} = λ² ≥ 0.

### Linear Functional (Appendix D)
```
α[F] = Σ_{m,n} λ_{m,n} ∂_a^m ∂_b^n F|_{a=1,b=0}
```
Index set: m odd, m ≥ 1, n ≥ 0, m + 2n ≤ 21 (for n_max=10)
Total: 66 terms.

### Coordinates (Eq. 4.15)
```
z = (a + √b)/2,  z̄ = (a - √b)/2
```
Crossing-symmetric point: a=1, b=0 → z=z̄=1/2 → u=v=1/4

### Spin-0 Block at z=z̄ (Eq. 4.10)
```
G_{Δ,0}(z) = (z²/(1-z))^{Δ/2} × ₃F₂(Δ/2, Δ/2, Δ/2-α; (Δ+1)/2, Δ-α; z²/(4(z-1)))
```
where α = D/2 - 1 = 0.5 for D=3.

### Spin Recursion (Eq. 4.9)
At z=z̄, becomes non-derivative recursion relating G_{Δ,l} to G_{Δ,l-2}, G_{Δ+1,l-1}, G_{Δ+2,l-2}.

### Transverse Derivatives (Eq. C.1)
Casimir recursion expressing h_{m,n} in terms of h_{m',n'} with n' < n.

---

## Discretization (Table 2)

| Table | δ (step) | Δ_max | L_max |
|-------|----------|-------|-------|
| T1 | 2×10⁻⁵ | 3 | 0 |
| T2 | 5×10⁻⁴ | 8 | 6 |
| T3 | 2×10⁻³ | 22 | 20 |
| T4 | 0.02 | 100 | 50 |
| T5 | 1 | 500 | 100 |

Dimension range for each (table, l):
- Δ_min(l) = l + 1 - (1/2)δ_{l,0}
- Δ_max = Δ_max^table + 2(L_max^table - l)

Even spins only: l = 0, 2, 4, ...

---

## Algorithm Overview

### Stage A (Δε bound)
```
For each Δσ in grid:
    Binary search Δε_trial:
        Build spectrum with gap: no scalar with Δ < Δε_trial
        if LP_feasible(Δσ, spectrum):
            # Can exclude this gap → push higher
            lo = Δε_trial
        else:
            # Cannot exclude → gap too large
            hi = Δε_trial
    Record Δε_max(Δσ) = lo
```

### Stage B (Δε' bound)
```
For each Δσ in grid:
    Set Δε = Δε_max(Δσ) from Stage A
    Binary search Δε'_trial:
        Build spectrum with two gaps:
            - No scalar with Δ < Δε
            - No scalar with Δε < Δ < Δε'_trial
        if LP_feasible(Δσ, spectrum):
            lo = Δε'_trial
        else:
            hi = Δε'_trial
    Record Δε'_max(Δσ)
```

### LP Feasibility
Looking for α with:
1. α[F_id] = 1 (normalization)
2. α[F_{Δ,l}] ≥ 0 for all (Δ,l) in spectrum

If such α exists → spectrum EXCLUDED (contradiction with crossing)
If no such α → spectrum ALLOWED (cannot be ruled out)

---

## File Structure

```
3D Ising CFT Bootstrap/
├── README.md                     # Original specification
├── pyproject.toml               # Package config [TO CREATE]
├── docs/
│   ├── TODO.md                  # Detailed milestones ✓
│   ├── implementation_choices.md # Block strategy ✓
│   ├── RUN.md                   # Run instructions ✓
│   └── PLAN_SUMMARY.md          # This file ✓
├── src/ising_bootstrap/         # Main package
│   ├── config.py                # Constants
│   ├── blocks/                  # Block computation
│   ├── spectrum/                # Discretization
│   ├── lp/                      # LP solver
│   ├── scans/                   # Stage A & B (both implemented)
│   └── plot/                    # Plotting (fig6.py)
├── tests/                       # 302 tests (277 Stage A + 25 Stage B)
├── data/                        # Output data
└── figures/                     # Generated figures [TO CREATE]
```

---

## Dependencies

```
numpy>=1.24
scipy>=1.10       # HiGHS LP solver
mpmath>=1.3       # Extended-precision ₃F₂
matplotlib>=3.7
pytest>=7.0       # Testing
```

---

## Validation Targets

| Check | Expected | Source |
|-------|----------|--------|
| Index set count | 66 | m odd, m+2n≤21 |
| Δε_max(0.518) | ≈ 1.41 | Fig. 3 of paper |
| Δε'_max(0.518) | ≈ 3.84 | Sec. 5.2 of paper |
| Spike feature | Present below Ising | Fig. 6 qualitative |

---

## Key Documentation Files

1. **[docs/TODO.md](TODO.md)** - Detailed implementation checklist with all tasks
2. **[docs/implementation_choices.md](implementation_choices.md)** - Block strategy decision rationale
3. **[docs/RUN.md](RUN.md)** - How to run the pipeline
4. **[README.md](../README.md)** - Original physics specification

---

## Notes for Claude Instances

1. **Always check TODO.md** for current implementation status
2. **Match paper's formulas exactly** - normalization is critical
3. **Test incrementally** - validate each milestone before proceeding
4. **Use reduced discretization** (T1-T2 only) for fast testing
5. **Cache block derivatives** - they don't depend on Δσ
