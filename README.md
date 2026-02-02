
# Reproducing Figure 6 of arXiv:1203.6064 (3D Ising bootstrap, bound on Δε′ vs Δσ)

This repository’s only scientific goal is to **reproduce Figure 6** of:

- S. El-Showk et al., *Solving the 3D Ising Model with the Conformal Bootstrap*, arXiv:1203.6064.

Figure 6 is described in Sec. 5.2 and implemented via the linear-functional / linear-programming approach in Appendix D.

The plot shows an **upper bound** on the second Z2-even scalar dimension (called ε′ or ε0 in the paper’s notation) as a function of the external scalar dimension Δσ, **under the condition that** Δε is set to the **maximal value allowed** by the bootstrap bound of Fig. 3.

---

## 0. Deliverables (what “done” means)

1. A script that generates a CSV (or JSON) dataset:
   - Input grid: Δσ ∈ [0.50, 0.60] (use the same domain as the paper’s Fig. 6).
   - Output: for each Δσ, the computed bootstrap upper bound `Δεprime_max(Δσ)` **with n_max = 10**.

2. A plotting script producing a PNG/PDF visually matching Fig. 6:
   - x-axis: Δσ
   - y-axis: bound on Δε′
   - include a vertical marker at the Ising Δσ ≈ 0.5182 (paper uses a thick red line).

3. A “sanity check” report printed to stdout:
   - near Δσ ≈ 0.5182, the curve should yield **Δε′ ≈ 3.84** at this truncation.
   - you should observe the qualitative “rapid growth” of the bound just below the Ising Δσ, as in the paper.

---

## 1. Physics / mathematical context (minimal but complete)

### 1.1 Which correlator?
We bootstrap the identical-scalar 4-point function of the **Z2-odd Ising spin field** σ:

\[
\langle \sigma(x_1)\sigma(x_2)\sigma(x_3)\sigma(x_4)\rangle.
\]

Its conformal block expansion includes only **Z2-even operators** in the σ×σ OPE (identity, ε, ε′, stress tensor, etc.) and only **even spins** by Bose symmetry.

### 1.2 Crossing equation in the form used by the paper
Define cross ratios in the usual way:
- \(u = z\bar z\),
- \(v = (1-z)(1-\bar z)\).

The paper rewrites crossing for this correlator into a single scalar equation (their Eq. (5.3)).

Operationally, for each primary \(\mathcal O\) with dimension Δ and spin ℓ, define a “crossing difference” function
\[
F^{\Delta_\sigma}_{\Delta,\ell}(u,v)
\equiv v^{\Delta_\sigma}\,G_{\Delta,\ell}(u,v)
 - u^{\Delta_\sigma}\,G_{\Delta,\ell}(v,u),
\]
where \(G_{\Delta,\ell}\) is the conformal block in the normalization consistent with the paper.

Then the crossing equation is:
\[
F^{\Delta_\sigma}_{\text{id}}(u,v) + \sum_{\mathcal O\neq \text{id}}
\lambda_{\sigma\sigma\mathcal O}^2\,F^{\Delta_\sigma}_{\Delta,\ell}(u,v)=0,
\]
with \(\lambda^2 \ge 0\) by unitarity/reflection positivity.

### 1.3 Unitarity bounds in D = 3
For the exchanged primaries in 3D:
- scalars (ℓ=0): \(\Delta \ge \tfrac12\)
- spins ℓ ≥ 1: \(\Delta \ge \ell + 1\)

Only even ℓ appear in σ×σ for identical σ.

---

## 2. What Figure 6 is bounding (precise statement)

Let ε be the **lowest-dimension Z2-even scalar** in σ×σ, and ε′ the **next** Z2-even scalar.

Figure 3 in the paper provides, for each Δσ, an **upper bound** on Δε (call it \(\Delta_\varepsilon^{\max}(\Delta_\sigma)\)).

Figure 6 then computes, for each Δσ, an **upper bound** on Δε′ under the condition
\[
\Delta_\varepsilon = \Delta_\varepsilon^{\max}(\Delta_\sigma).
\]
In other words, it reports a function \(\Delta_{\varepsilon'}^{\max}(\Delta_\sigma)\) evaluated along the “maximal-ε” curve.

Important: operationally, the way you implement “bounding Δε′” in this framework is by assuming a **gap** above ε:
- there is **no** Z2-even scalar in \((\Delta_\varepsilon,\Delta_{\varepsilon'})\),
- i.e. the next scalar satisfies \(\Delta \ge \Delta_{\varepsilon'}\),
and then you find the maximal value of \(\Delta_{\varepsilon'}\) compatible with crossing + unitarity.

This is exactly the “gap between ε and ε′” logic of Sec. 5.2; Fig. 6 is a 1D slice of that story.

---

## 3. Numerical method: linear functional + linear programming (as in Appendix D)

### 3.1 Expand around the crossing-symmetric point
The paper Taylor-expands around \(z=\bar z = 1/2\), which is symmetric under exchanging channels.

They introduce variables (Eq. (4.15)):
\[
z = \tfrac12 + a + b,\qquad
\bar z = \tfrac12 + a - b,
\]
so that the symmetric line \(z=\bar z\) is \(b=0\), and the expansion point is \(a=b=0\).

### 3.2 Define the linear functional
A linear functional \(\alpha\) acts on functions of (u,v) via derivatives in (a,b) at the origin:

\[
\alpha[F] = \sum_{m,n} \lambda_{m,n}\,
\left.\partial_a^m\partial_b^n F\right|_{a=b=0}.
\]

The index set \((m,n)\) is truncated by an integer \(n_{\max}\). In this work we must use
\[
n_{\max}=10.
\]


Because \(F^{\Delta_\sigma}_{\Delta,\ell}(u,v)\) is antisymmetric under \(u\leftrightarrow v\) (equivalently \(a\to -a\) at fixed b), only **odd m** derivatives contribute.

The paper’s derivative-truncation rule is (Appendix D):
- precompute \(\partial_a^m G_{\Delta,\ell}\) up to \(m = 2n_{\max}+1\),
- compute all derivatives in the range
  \[
  m + 2n \le 2n_{\max}+1.
  \]


For n_max = 10, this means \(m+2n \le 21\), with m odd.

### 3.3 LP feasibility condition (dual)
Write crossing schematically as
\[
F_{\text{id}} + \sum p_{\Delta,\ell} F_{\Delta,\ell} = 0,\quad p_{\Delta,\ell}=\lambda^2\ge 0.
\]

To **exclude** a hypothesized spectrum, it suffices to find \(\alpha\) such that:
1. Normalization: \(\alpha[F_{\text{id}}]=1\).
2. Positivity: \(\alpha[F_{\Delta,\ell}] \ge 0\) for every operator (Δ,ℓ) allowed by the hypothesized spectrum constraints.

If such \(\alpha\) exists, applying it to crossing yields
\[
1 + \sum p_{\Delta,\ell}\,\alpha[F_{\Delta,\ell}] = 0,
\]
which is impossible because every term is ≥ 0. Hence the hypothesized spectrum is inconsistent.

**This is a pure feasibility LP in the variables \(\lambda_{m,n}\).**

---

## 4. Discretization of the spectrum (must match Table 2)

The paper does not impose positivity for a continuum of Δ,ℓ directly; it **discretizes** Δ and truncates ℓ, using multiple “tables” T1–T5 to cover different Δ ranges and resolutions.

You should reproduce Table 2 exactly:

| Table | δ (step in Δ) | Δ_max | L_max |
|------:|---------------:|------:|------:|
| T1 | 2×10^-5 | 3 | 0 |
| T2 | 5×10^-4 | 8 | 6 |
| T3 | 2×10^-3 | 22 | 20 |
| T4 | 0.02 | 100 | 50 |
| T5 | 1 | 500 | 100 |



For each table, dimensions are sampled from the unitarity bound
\[
\Delta_{\min}(\ell) = \ell + 1 - \tfrac12\delta_{\ell,0}
\]
up to
\[
\Delta_{\max}^{\text{table}} + 2(L_{\max}^{\text{table}}-\ell),
\]
in steps of δ, and spins are restricted to \(0\le \ell \le L_{\max}\) (even ℓ only in practice).

**Implementation note:** combining tables means you take the union of all sampled (Δ,ℓ) points across T1–T5 and impose the positivity constraint at each point.

---

## 5. What has to be implemented in code (roadmap)

### 5.1 Conformal blocks and derivatives: choose an existing implementation
You do *not* want to hand-derive blocks. Use a known conformal-block engine and extract numerical derivatives at \(z=\bar z=1/2\).

Acceptable options:
- `PyCFTBoot`-style block tables (historically used with SDPB; but you can also just read out derivatives).
- `JuliBootS` block tables.
- Any reliable scalar conformal block library in D=3 that supports:
  - arbitrary (Δ,ℓ),
  - evaluation at generic (z, zbar),
  - derivatives up to total order ~21 (in the (a,b) sense).

**Key caution:** normalization conventions for \(G_{\Delta,\ell}\) differ across libraries. This matters because α[F_id]=1 must use the same normalization as the operator blocks. Before running large scans, validate with one small case by reproducing any published number (e.g. the approximate Δε′ ≈ 3.84 at Ising).

### 5.2 Compute derivative “feature vectors” for each (Δ,ℓ)
For each sampled operator (Δ,ℓ), you need the vector
\[
v_{\Delta,\ell}^{(m,n)} = \left.\partial_a^m\partial_b^n F_{\Delta,\ell}^{\Delta_\sigma}\right|_{0},
\]
over the truncated index set \(m+2n\le 21\), m odd.

You also need the identity contribution:
\[
F_{\text{id}}^{\Delta_\sigma}(u,v) = v^{\Delta_\sigma} - u^{\Delta_\sigma}.
\]
Differentiate it in the same way to get \(v_{\text{id}}^{(m,n)}\).

**Performance trick:** blocks \(G_{\Delta,\ell}\) do **not** depend on Δσ, only on (Δ,ℓ,d). Δσ only enters through prefactors \(u^{\Delta_\sigma}, v^{\Delta_\sigma}\) and the u↔v swap. So cache derivatives of \(G_{\Delta,\ell}\) once, then assemble \(F^{\Delta_\sigma}\) derivatives cheaply for each Δσ.

### 5.3 LP formulation
Let the LP variables be the coefficients \(\lambda_{m,n}\) in α.

- Equality constraint:
  \[
  \sum_{m,n}\lambda_{m,n}\,v_{\text{id}}^{(m,n)} = 1.
  \]
- Inequality constraints (for each sampled (Δ,ℓ) allowed by your spectrum assumptions):
  \[
  \sum_{m,n}\lambda_{m,n}\,v_{\Delta,\ell}^{(m,n)} \ge 0.
  \]

This is a feasibility problem. Use any robust LP solver:
- commercial: CPLEX / Gurobi (closest to the paper; they used CPLEX dual simplex)
- open-source: HiGHS (via SciPy), GLPK, CBC (might be slower)

**Numerical stability:** constraints span many orders of magnitude. You will almost certainly need:
- scaling/normalization of constraint rows,
- rational/decimal high precision (at least float128 or mpfr-like) if available,
- a consistent tolerance for “≥ 0”.

The paper’s results rely on careful numerics; do not expect naive float64 to be robust near the kink/spike.

---

## 6. How to reproduce Fig. 6: the two-stage scan

### Stage A: reproduce Fig. 3 boundary Δε_max(Δσ)
For each Δσ in your scan grid:

1. Define a trial value Δε_trial.
2. Impose the spectrum assumption:
   - **no** scalar (ℓ=0) with \(\Delta < \Delta_{\varepsilon,\text{trial}}\) (besides identity),
   - all even spins ℓ=2,4,… satisfy just unitarity bounds.
3. Run the LP feasibility test:
   - if an α satisfying the constraints exists ⇒ the assumed gap is **excluded**.
4. Binary search in Δε_trial to find the threshold where the gap becomes excluded.
   - The **upper bound** \(\Delta_\varepsilon^{\max}(\Delta_\sigma)\) is the smallest Δε_trial that becomes excluded (within numerical tolerance).
   - This is exactly how “maximal allowed Δε” is defined in Sec. 5.1.

Store Δε_max(Δσ) to disk (CSV).

### Stage B: reproduce Fig. 6 boundary Δε′_max(Δσ) along Δε = Δε_max
For each Δσ:

1. Set \(\Delta_\varepsilon\) to the value found in Stage A:
   \[
   \Delta_\varepsilon \leftarrow \Delta_\varepsilon^{\max}(\Delta_\sigma).
   \]
   This is the “condition” stated in the Fig. 6 caption.

2. Now scan a trial Δε′_trial and impose the spectrum assumption:
   - no scalar with \(\Delta < \Delta_\varepsilon\) (besides identity),
   - **(gap)** no scalar with \(\Delta \in (\Delta_\varepsilon, \Delta_{\varepsilon',\text{trial}})\),
   - other even-spin operators only constrained by unitarity.

3. Run the same LP feasibility test:
   - if a functional α exists ⇒ this hypothesized large gap is excluded.
4. Binary search Δε′_trial to find the exclusion threshold, giving
   \[
   \Delta_{\varepsilon'}^{\max}(\Delta_\sigma).
   \]

#### Subtlety (do not ignore)
The phrase “Δε fixed” is stronger than “no scalars below Δε”.
In a strict mathematical sense, to force an operator to exist *exactly* at Δε you would need additional structure (e.g. extremal functional reconstruction or primal feasibility with a nonzero coefficient). The paper’s practical implementation uses the fact that they are scanning along the **boundary** of the allowed region in (Δσ,Δε), where the extremal solution is expected to have ε at that location.

For reproduction purposes:
- Implement exactly the **gap constraints** above (as the paper’s LP method naturally supports).
- Then validate empirically that, near the boundary, the extremal functional exhibits a near-zero at Δ≈Δε (optional but recommended).

This is the most faithful path to reproducing Fig. 6 without re-deriving their entire “extremal spectrum” machinery.

---

## 7. Parameter choices that must match the paper

- Dimension: D = 3.
- Correlator: identical scalar σ.
- Truncation: \(n_{\max} = 10\).
- Discretization tables: exactly Table 2 (T1–T5).
- Expansion point: \(z=\bar z = 1/2\).

---

## 8. Expected qualitative / quantitative checks

1. **Qualitative:** the Δε′ bound curve should show a pronounced upturn (“rapid growth”) just below the Ising Δσ, matching Fig. 6 discussion.

2. **Quantitative:** around the Ising point (Δσ ≈ 0.5182), the paper reports
   \[
   \Delta_{\varepsilon'}^{\max} \approx 3.84
   \]
   at this level of truncation.

3. **Regression test:** using the same code, you should also be able to reproduce one of the Fig. 5 “gap-assumption” exclusion regions (optional but strongly diagnostic). Sec. 5.2 describes the gap constraints used there.

---

## 9. Suggested repository structure

```

.
├── README.md
├── pyproject.toml / requirements.txt
├── src/
│   ├── blocks/
│   │   ├── block_engine_wrapper.py    # wrapper around external conformal-block library
│   │   ├── derivatives.py             # computes ∂a^m ∂b^n at a=b=0, caches
│   │   └── normalization_tests.py     # small tests to detect convention mismatches
│   ├── spectrum/
│   │   ├── discretization.py          # builds T1–T5 grids exactly as Table 2
│   │   └── assumptions.py             # imposes scalar gaps for (ε) and (ε′)
│   ├── lp/
│   │   ├── build_lp.py                # constructs Ax ≥ b, equality constraint
│   │   └── solve_lp.py                # solver interface, scaling, tolerances
│   ├── scans/
│   │   ├── scan_eps_bound.py          # Stage A (Fig. 3 boundary)
│   │   └── scan_epsprime_bound.py     # Stage B (Fig. 6)
│   └── plot/
│       └── plot_fig6.py
└── data/
├── cached_block_derivatives/      # heavy cache
├── eps_bound.csv                  # Δε_max(Δσ)
└── epsprime_bound.csv             # Δε′_max(Δσ)

````

---

## 10. Implementation details that commonly go wrong

### 10.1 Block normalization mismatch
Different block libraries define \(G_{\Delta,\ell}\) with different prefactors.
Symptoms:
- your bounds are systematically shifted,
- you do not see the spike near Ising,
- the Ising-point value is far from ~3.84.

Mitigation:
- implement unit tests at the level of the crossing equation and identity term,
- compare one computed derivative of \(F_{\Delta,\ell}\) against an independent implementation (even at low order).

### 10.2 Derivative indexing: use the paper’s rule exactly
Do not substitute “total derivative order” for the paper’s constraint \(m+2n \le 2n_{\max}+1\).
This *matters* because transverse derivatives are “more expensive” and counted differently.

### 10.3 Spectrum discretization: Table 2 must be followed literally
If you skip T4/T5, the solver may exploit missing asymptotic constraints and you can get artificially strong bounds (or false feasibility).

### 10.4 LP numerical conditioning
Expect constraint matrices with very poor conditioning.
Practical necessities:
- row/column scaling,
- solver tolerances tuned and logged,
- reproducible seeds/options.

---

## 11. Minimal command sequence (what the user should run)

Example (you will implement the actual entrypoints):

```bash
# Stage A: compute Δε_max(Δσ)
python -m scans.scan_eps_bound \
  --d 3 --nmax 10 \
  --sigma-min 0.50 --sigma-max 0.60 --sigma-step 0.001 \
  --out data/eps_bound.csv

# Stage B: compute Δε′_max(Δσ) using Δε fixed to Stage A result
python -m scans.scan_epsprime_bound \
  --d 3 --nmax 10 \
  --eps-bound data/eps_bound.csv \
  --out data/epsprime_bound.csv

# Plot
python -m plot.plot_fig6 \
  --epsprime-bound data/epsprime_bound.csv \
  --out figures/fig6_reproduction.png
````

---

## 12. Primary references inside the paper (what to read while coding)

* Sec. 5.2 for what Fig. 6 represents and the “gap above ε” logic.
* Appendix D for the LP functional definition, n_max truncation, and Table 2 discretization.
* Eq. (4.15) for the (a,b) coordinates around z= z̄ = 1/2.
* Eq. (5.3) for the explicit crossing equation being linearized.

---

## 13. Bottom line (interpretation guidance)

If the implementation is faithful, you should reproduce the key empirical phenomenon:

* a sharp change in the allowed size of the scalar gap above ε near the Ising Δσ,
* yielding Δε′ ≈ 3.84 near Ising at n_max=10.

Do not interpret this as a proof of the Ising spectrum by itself; it is a **rigorous exclusion bound** within a specific truncation/discretization scheme. Increasing n_max and tightening discretization should monotonically strengthen (or stabilize) the bounds, but the convergence is not trivial and must be checked empirically.
