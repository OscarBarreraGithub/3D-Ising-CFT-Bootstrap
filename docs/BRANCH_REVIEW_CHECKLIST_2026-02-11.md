# Branch Review Checklist (2026-02-11)

Branch: `codex/strict-failfast-stageb-snap-eps`

This checklist is intended to support a single end-to-end review before merge to `main`.

## Commits in Scope

1. `1f6fa47` Enforce strict SDPB failure handling and Stage B epsilon anchoring
2. `8bd441c` Update tests and docs for strict Stage B and SDPB semantics
3. `41657b6` Add SDPB runtime-envelope orchestration and cluster runbook

## What Changed (Conceptual)

1. SDPB outcomes are now interpreted strictly:
- Only explicit feasible/infeasible termination reasons are trusted.
- Inconclusive/unknown reasons are solver failures (`success=False`), not "allowed".

2. Stage A now fail-fast on solver anomalies:
- No conservative fallback for failed solves.
- A single failed solve in binary search raises and fails that point/run.

3. Stage B now fail-fast on solver anomalies:
- No silent continuation when solver reliability is unknown.
- Missing valid Stage A row for requested `Delta sigma` is fatal.

4. Stage B now uses fixed-`Delta epsilon` snap-and-anchor semantics:
- Stage A `Delta epsilon` is snapped to nearest scalar grid point.
- Snap tolerance enforced (`eps_snap_tolerance`, default `1e-3`).
- Anchor row at snapped `Delta epsilon` is explicitly included.

5. Stage A inputs to Stage B are sanitized:
- Non-finite values are dropped with warning.
- Strict missing-key behavior prevents silent skips.

6. SDPB timeout is now explicitly configurable end-to-end:
- Exposed via Stage A/Stage B CLIs (`--sdpb-timeout`).
- Propagated via SLURM scripts and pipeline launcher.

7. Pipeline gates are hardened:
- Stage B launch is blocked when Stage A merged results are invalid/pathological.

## Primary Files to Review

### Runtime Semantics

- `src/ising_bootstrap/lp/sdpb.py:383`
  SDPB termination-reason mapping to `FeasibilityResult`.

- `src/ising_bootstrap/scans/stage_a.py:280`
  Stage A binary-search solve callback fail-fast behavior.

- `src/ising_bootstrap/scans/stage_b.py:122`
  `Delta epsilon` snap helper + tolerance enforcement.

- `src/ising_bootstrap/scans/stage_b.py:199`
  Anchored epsilon row inclusion and two-gap row mask.

- `src/ising_bootstrap/scans/stage_b.py:320`
  Stage A map sanitization and strict missing-row behavior.

### Job Orchestration

- `jobs/stage_a_sdpb.slurm:27`
- `jobs/stage_b_sdpb.slurm:28`
- `jobs/test_sufficient_memory.slurm:26`
- `jobs/merge_stage_a_and_submit_b.slurm:28`
- `jobs/run_pipeline.sh:26`

### Tests

- `tests/test_lp/test_sdpb.py:161`
- `tests/test_scans/test_stage_a.py:193`
- `tests/test_scans/test_stage_b.py:46`
- `tests/test_scans/test_stage_b.py:189`
- `tests/test_scans/test_stage_b.py:337`
- `tests/test_scans/test_stage_b.py:424`

### New Operational Helpers

- `jobs/submit_stage_a_runtime_envelope.sh`
- `jobs/stage_a_pilot_sdpb.slurm`
- `jobs/merge_stage_a_pilot.sh`
- `jobs/stage_b_smoke_sdpb.slurm`
- `jobs/submit_stage_b_smoke.sh`
- `docs/SDPB_RUNTIME_ENVELOPE_2026-02-11.md`

## Required Test Matrix Before Merge

Run these in order.

### A. Fast Python Validation (local/CPU)

```bash
PYTHONPATH=src pytest tests/test_lp/test_sdpb.py -q
PYTHONPATH=src pytest tests/test_scans/test_stage_a.py -q
PYTHONPATH=src pytest tests/test_scans/test_stage_b.py -q
```

Pass criteria:
- All three suites pass.
- No failures in strict-semantics tests.

### B. Stage A Single-Point Runtime Envelope (cluster)

1. Baseline:
```bash
bash jobs/submit_stage_a_runtime_envelope.sh
```

2. If timeout:
```bash
TIMEOUT=3600 bash jobs/submit_stage_a_runtime_envelope.sh
```

3. If timeout persists:
```bash
TIMEOUT=3600 CPUS=16 MEM=160G WALLTIME=08:00:00 bash jobs/submit_stage_a_runtime_envelope.sh
```

4. Last-resort profiling:
```bash
TIMEOUT=3600 TOLERANCE=1e-3 bash jobs/submit_stage_a_runtime_envelope.sh
```

For each run:
```bash
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem
tail -200 logs/test_sufficient_memory_<JOBID>.log
cat data/test_sufficient_memory.csv
```

Pass criteria:
- Job exits 0.
- Output CSV has finite `delta_eps_max`.
- No SDPB timeout/inconclusive/failure in log.

### C. Stage A Pilot (5-point) then Merge

```bash
sbatch --array=0,9,18,27,36 \
  --cpus-per-task=8 \
  --mem=128G \
  --time=12:00:00 \
  --export=ALL,SDPB_TIMEOUT=1800,STAGE_A_TOLERANCE=1e-4 \
  jobs/stage_a_pilot_sdpb.slurm

bash jobs/merge_stage_a_pilot.sh
```

Pass criteria:
- Pilot CSV merges cleanly.
- Finite, non-pathological values across pilot points.

### D. Stage B Smoke (single-point strict behavior)

Requires valid Stage A merged map first.

```bash
bash jobs/submit_stage_b_smoke.sh
```

Pass criteria:
- Job exits 0 for valid input.
- No silent "allowed" behavior on solver anomalies.
- Output row is finite and includes anchored `delta_eps`.

### E. Merge Gate Validation (negative tests)

Validate `jobs/merge_stage_a_and_submit_b.slurm` gate behavior by staged fixtures:

1. Valid numeric data -> Stage B submitted.
2. Non-finite rows present -> gate blocks.
3. All rows near 0.5 -> gate blocks.
4. All rows near 2.5 -> gate blocks.
5. Too few valid rows -> gate blocks.

Pass criteria:
- Invalid cases exit non-zero before Stage B submission.
- Valid case proceeds.

## Expected Signals in Logs

### Good

- Stage A/B logs show configured timeout and tolerance.
- SDPB run lines present and no repeated inconclusive terminations.
- Stage B logs show snapped epsilon message only when needed.

### Bad (must block merge)

- Any non-finite Stage A merged values.
- Uniform Stage A saturation near lower or upper bound.
- Stage B proceeding when Stage A key is missing.
- Stage A/Stage B treating solver failures as valid allowed points.

## Notes / Known Operational Choice

`jobs/run_pipeline.sh` defaults `RUN_TEST_GATES=0` because existing `jobs/test_gates.slurm` targets legacy extended-solver gates. This avoids blocking SDPB runtime characterization with stale gate assumptions. If you want SDPB-native gates later, add a dedicated SDPB gate script and set `RUN_TEST_GATES=1`.

## Merge Criteria (all required)

1. Python tests above pass.
2. Stage A single-point characterization finds a finite envelope.
3. Stage A pilot passes without pathological pattern.
4. Stage B smoke passes with strict behavior.
5. Merge gate negative tests block invalid inputs correctly.
6. No unresolved reviewer comments on runtime semantics.
