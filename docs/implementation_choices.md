# Implementation Choices: Conformal Block Evaluation Strategy

## Decision Summary

**Choice**: Implement a specialized conformal block evaluator (Option B) rather than using an existing library (Option A).

---

## Background: What We Need

To reproduce Figure 6 of arXiv:1203.6064, we need to compute:

1. **Conformal blocks** G_{Δ,l}(z,z̄) for D=3 dimensions
2. **Derivatives** ∂_a^m ∂_b^n G evaluated at the crossing-symmetric point z=z̄=1/2
3. **High derivative orders**: up to m+2n ≤ 21 (for n_max=10)
4. **Many operator dimensions**: ~10⁵ discretized (Δ,l) points from Table 2

The key insight is that we ONLY need evaluations at z=z̄=1/2, not at arbitrary (z,z̄).

---

## Option A: Existing Libraries

### Available Libraries

| Library | Language | D=3 Support | Derivatives | Notes |
|---------|----------|-------------|-------------|-------|
| scalar_blocks | C++ | Yes | Yes | Fastest, maintained by bootstrap collaboration |
| blocks_3d | C++ | Yes | Yes | General 3D, 2-3x slower than scalar_blocks |
| PyCFTBoot | Python | Yes | Via SDPB | Wrapper, depends on external tools |
| blocks.wl | Mathematica | Yes | Yes | Original paper used this |
| JuliBootS | Julia | Yes | Yes | Pedagogical, good for learning |

### Pros of Option A
- Battle-tested implementations
- Optimized performance
- Community support

### Cons of Option A
- **C++ compilation required** for scalar_blocks/blocks_3d
- **Complex dependencies** (MPFR, Boost, etc.)
- **Normalization conventions unclear** - must reverse-engineer to match paper
- **Overkill** - these libraries compute blocks at arbitrary (z,z̄), but we only need z=z̄=1/2
- **Black box** - harder to debug if results don't match

---

## Option B: Specialized Evaluator

### Key Observation

The paper (Sections 4 and Appendices B,C) provides **explicit closed-form expressions** for everything we need at z=z̄:

1. **Spin 0 blocks** (Eq. 4.10):
   ```
   G_{Δ,0}(z) = (z²/(1-z))^{Δ/2} × ₃F₂(Δ/2, Δ/2, Δ/2-α; (Δ+1)/2, Δ-α; z²/(4(z-1)))
   ```

2. **Spin 1 blocks** (Eq. 4.11):
   ```
   G_{Δ,1}(z) = (2-z)/(2z) × (z²/(1-z))^{(Δ+1)/2} × ₃F₂(...)
   ```

3. **Higher spins via recursion** (Eq. 4.9) - at z=z̄, this becomes a **non-derivative** algebraic recursion:
   ```
   (l+D-3)(2Δ+2-D) G_{Δ,l} = ... terms involving G_{Δ,l-2}, G_{Δ+1,l-1}, G_{Δ+2,l-2}
   ```

4. **z-derivatives** via ₃F₂ differential equation (Eq. 4.12)

5. **Transverse (b) derivatives** via Casimir recursion (Eq. C.1)

### Pros of Option B
- **Full control over normalization** - can match paper exactly
- **No C++ dependencies** - pure Python with mpmath
- **Transparent implementation** - easy to debug and validate
- **Optimal for our use case** - no wasted computation at unused (z,z̄) points
- **Extended precision** - mpmath provides arbitrary-precision ₃F₂

### Cons of Option B
- More code to write (~500-800 lines)
- Must implement and test each recursion carefully
- Potential for bugs in mathematical formulas

---

## Decision Rationale

We choose **Option B** for the following reasons:

### 1. Paper Provides All Formulas
The paper explicitly gives every formula needed for z=z̄=1/2 evaluation. This is not a research problem; it's a direct implementation of known mathematics.

### 2. Normalization is Critical
The README emphasizes (Section 10.1): "Block normalization mismatch" is a common failure mode. By implementing from the paper's equations, we use the exact same normalization convention.

### 3. Restricted Domain
We only need z=z̄=1/2. General-purpose libraries compute blocks everywhere, which is overkill and may introduce unnecessary complexity.

### 4. Dependency Simplicity
Pure Python with mpmath vs. C++ compilation with MPFR/Boost/etc. The former is far easier to set up and maintain.

### 5. Debugging Transparency
When (not if) something goes wrong, it's much easier to debug our own code than to reverse-engineer a compiled library.

---

## Implementation Details

### Dependencies
```
numpy>=1.24      # Array operations, LP input
scipy>=1.10      # LP solver (HiGHS)
mpmath>=1.3      # Extended-precision ₃F₂
matplotlib>=3.7  # Plotting
```

### Precision Strategy

1. **Block computation**: Use mpmath with 50+ decimal digits
2. **LP constraints**: Convert to float64 (sufficient for HiGHS)
3. **Critical values**: Validate against known results before full run

### Normalization Convention

Following Appendix A of the paper:
- G_{Δ,l} = F_{λ₁,λ₂} where λ₁ = (Δ+l)/2, λ₂ = (Δ-l)/2
- This is the Dolan-Osborn normalization from their Ref. [15]
- NOT the older convention from Refs. [13,43] which has an extra prefactor

### Caching Strategy

Block derivatives h_{m,n}(Δ,l) depend only on (Δ,l,D), NOT on Δσ.

The Δσ-dependence enters only through the prefactors v^{Δσ} and u^{Δσ} in the crossing function:
```
F^{Δσ}_{Δ,l}(u,v) = v^{Δσ} G_{Δ,l}(u,v) - u^{Δσ} G_{Δ,l}(v,u)
```

Therefore:
1. Precompute h_{m,n}(Δ,l) for all (Δ,l) in Table 2 discretization
2. Cache to disk in `data/cached_blocks/`
3. For each Δσ, load cached blocks and apply prefactors

This is the same strategy described in README Section 5.2.

---

## Validation Plan

### Step 1: Spin-0 Block
Compare G_{Δ,0}(1/2) against independent calculation (e.g., Mathematica) for Δ = 1.0, 1.5, 2.0.

### Step 2: Spin Recursion
Verify G_{Δ,2}(1/2) from recursion matches direct formula (if available) or cross-check with different recursion paths.

### Step 3: Identity Derivatives
Compute ∂_a^m(v^{Δσ} - u^{Δσ})|_{a=1,b=0} analytically and compare to numerical implementation.

### Step 4: Single Point Feasibility
Run LP for known (Δσ, Δε) point and verify feasibility/infeasibility matches expectation.

### Step 5: Ising Point Check
At Δσ ≈ 0.5182, verify:
- Δε_max ≈ 1.41 (Stage A)
- Δε'_max ≈ 3.84 (Stage B)

---

## LP Solver Choice

### Decision: SciPy HiGHS

The paper used CPLEX (commercial). For an open-source implementation, we use **SciPy's linprog with HiGHS backend**.

### Rationale
1. **Built into SciPy** - no additional installation
2. **HiGHS dual simplex** - inspired by same principles as CPLEX
3. **Automatic scaling** - handles constraints spanning many orders of magnitude
4. **Good numerical stability** - adjustable tolerances

### Alternative Solvers (if needed)
- Google OR-Tools (GLOP) - good backup option
- CVXPY - convenient but limited precision
- GLPK - older, reliable for validation

### Usage Pattern
```python
from scipy.optimize import linprog

result = linprog(
    c=np.zeros(n_vars),      # No objective (feasibility)
    A_ub=-A,                  # Positivity: A @ λ ≥ 0
    b_ub=np.zeros(n_rows),
    A_eq=A_eq,                # Normalization
    b_eq=np.array([1.0]),
    method='highs-ds',        # Dual simplex
    options={'presolve': True, 'disp': False}
)

is_feasible = result.success and result.status == 0
# feasible → functional α exists → spectrum excluded
# infeasible → spectrum cannot be excluded → is allowed
```

---

## References

1. El-Showk et al., "Solving the 3D Ising Model with the Conformal Bootstrap," arXiv:1203.6064
2. Dolan & Osborn, "Conformal Partial Waves: Further Mathematical Results," arXiv:1108.6194
3. mpmath documentation: https://mpmath.org/doc/current/functions/hypergeometric.html
4. SciPy HiGHS documentation: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html
