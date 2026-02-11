# SDPB Runtime Envelope Bring-up (2026-02-11)

This runbook operationalizes timeout-dominated Stage A failures under strict fail-fast semantics.

## Preconditions

- Strict failure handling is enabled in Stage A/Stage B and SDPB result interpretation.
- `--sdpb-timeout` is exposed in both scan CLIs and wired in SDPB job scripts.
- Stage B fixed-`Delta epsilon` anchoring and snap tolerance are enabled.

## Phase 1: Single-Point Stage A Characterization (`Delta sigma=0.518`)

Run in order and stop at first finite success.

1. Baseline: `timeout=1800`, `cpus=8`, `mem=128G`
```bash
bash jobs/submit_stage_a_runtime_envelope.sh
```

2. If timed out, raise timeout to `3600`
```bash
TIMEOUT=3600 bash jobs/submit_stage_a_runtime_envelope.sh
```

3. If still timed out, raise cores (and memory margin)
```bash
TIMEOUT=3600 CPUS=16 MEM=160G WALLTIME=08:00:00 bash jobs/submit_stage_a_runtime_envelope.sh
```

4. Last-resort profiling mode (relaxed tolerance)
```bash
TIMEOUT=3600 TOLERANCE=1e-3 bash jobs/submit_stage_a_runtime_envelope.sh
```

For each run, inspect:
```bash
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem
tail -200 logs/test_sufficient_memory_<JOBID>.log
cat data/test_sufficient_memory.csv
```

Success criterion:
- `data/test_sufficient_memory.csv` contains one finite `delta_eps_max`.
- No SDPB timeout/inconclusive/failure in log.

## Phase 2: Stage A Pilot then Full Array

Use SDPB resources consistent with the successful single-point envelope.

1. Submit 5-point pilot:
```bash
sbatch --array=0,9,18,27,36 \
  --cpus-per-task=8 \
  --mem=128G \
  --time=12:00:00 \
  --export=ALL,SDPB_TIMEOUT=1800,STAGE_A_TOLERANCE=1e-4 \
  jobs/stage_a_pilot_sdpb.slurm
```

2. Merge pilot outputs:
```bash
bash jobs/merge_stage_a_pilot.sh
```

3. If pilot is healthy, run full Stage A:
```bash
sbatch --array=0-50 \
  --cpus-per-task=8 \
  --mem=128G \
  --time=12:00:00 \
  --export=ALL,SDPB_TIMEOUT=1800,STAGE_A_TOLERANCE=1e-4 \
  jobs/stage_a_sdpb.slurm
```

## Phase 3: Gate Before Stage B

Merge + gate job blocks Stage B when Stage A merged data has:
- non-finite values,
- all values near lower bound (`~0.5`),
- all values near upper bound (`~2.5`),
- too few valid rows.

Run:
```bash
sbatch jobs/merge_stage_a_and_submit_b.slurm
```

## Phase 4: Stage B Bring-up

1. One-point smoke (`Delta sigma=0.518`):
```bash
bash jobs/submit_stage_b_smoke.sh
```

2. If smoke passes, run pilot/full Stage B through `jobs/stage_b_sdpb.slurm` with array and resource overrides.

## End-to-End Pipeline Launcher

`jobs/run_pipeline.sh` now launches SDPB Stage A directly and forwards runtime knobs via env vars:

```bash
SDPB_TIMEOUT=1800 \
STAGE_A_TOLERANCE=1e-4 \
STAGE_B_TOLERANCE=1e-3 \
EPS_SNAP_TOLERANCE=1e-3 \
STAGE_A_CPUS=8 \
STAGE_A_MEM=128G \
STAGE_A_TIME=12:00:00 \
STAGE_B_CPUS=8 \
STAGE_B_MEM=128G \
STAGE_B_TIME=12:00:00 \
RUN_TEST_GATES=0 \
bash jobs/run_pipeline.sh
```
