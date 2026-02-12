# Sapphire Partition Migration Review Checklist

## Purpose

Validate that the 3D Ising CFT Bootstrap pipeline is ready for production runs on the Harvard FAS RC sapphire partition after resolving SDPB timeout issues on the shared partition.

## Background

- **Issue:** Stage A jobs timeout on shared partition (12h walltime limit, but need 28-35h per task)
- **Solution:** Migrate to sapphire partition (7-day walltime, 990GB RAM/node, 112 cores/node, MPI-optimized)
- **Detailed Analysis:** See `docs/OVERNIGHT_TIMEOUT_ANALYSIS_2026-02-12.md` for complete failure analysis and justification

**Key Findings:**
- SDPB solve time: 2-2.2 hours per iteration
- Bisection iterations needed: 12-16 per Δσ point
- Total time per point: 26-35 hours
- Shared partition maximum: 12 hours → **Insufficient by 2-3×**

---

## Checklist for External Reviewer

### 1. Code Review

#### 1.1 Partition Configuration

Verify all 18 SLURM scripts use sapphire partition:

- [ ] All `.slurm` files in `jobs/` directory have `#SBATCH --partition=sapphire` at line 4
- [ ] No scripts remain with `#SBATCH --partition=shared`
- [ ] Verification script exists and is executable: `jobs/verify_partition_migration.sh`
- [ ] Verification script passes when run

**Verification Commands:**
```bash
cd /n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap
bash jobs/verify_partition_migration.sh
# Expected output: "✓ All 18 .slurm files use sapphire partition"
```

**Manual spot-check:**
```bash
# Check a few files manually
sed -n '4p' jobs/stage_a_sdpb.slurm  # Should be: #SBATCH --partition=sapphire
sed -n '4p' jobs/stage_b_sdpb.slurm  # Should be: #SBATCH --partition=sapphire
sed -n '4p' jobs/test_sufficient_memory.slurm  # Should be: #SBATCH --partition=sapphire
```

**Search for missed references:**
```bash
grep -r "partition=shared" --include="*.slurm" jobs/
# Expected: No output (all changed to sapphire)
```

---

#### 1.2 Resource Requests (Sapphire-Appropriate)

Verify resource requests fit sapphire node capacity and job requirements:

- [ ] **Stage A** (`stage_a_sdpb.slurm`): 16 cores, 128GB, 36h walltime
- [ ] **Stage B** (`stage_b_sdpb.slurm`): 16 cores, 128GB, 36h walltime
- [ ] **Test script** (`test_sufficient_memory.slurm`): 16 cores, 128GB, 36h walltime
- [ ] **Pilot script** (`stage_a_pilot_sdpb.slurm`): 16 cores, 128GB, 36h walltime
- [ ] **Smoke test** (`stage_b_smoke_sdpb.slurm`): 16 cores, 128GB, 36h walltime
- [ ] Supporting scripts use sapphire but keep original resource requests (merge jobs, precompute, etc.)

**Verification Commands:**
```bash
# Check critical production scripts
grep -E "mem|cpus-per-task|time" jobs/stage_a_sdpb.slurm | head -5
# Expected: --time=36:00:00, --mem=128G, --cpus-per-task=16

grep -E "mem|cpus-per-task|time" jobs/stage_b_sdpb.slurm | head -5
# Expected: --time=36:00:00, --mem=128G, --cpus-per-task=16

grep -E "mem|cpus-per-task|time" jobs/test_sufficient_memory.slurm | head -5
# Expected: --time=36:00:00, --mem=128G, --cpus-per-task=16
```

**Sapphire node capacity check:**
- Sapphire nodes: 990GB RAM, 112 cores
- Each job: 128GB, 16 cores
- Jobs per node (CPU-limited): 112 / 16 = 7
- Jobs per node (RAM-limited): 990 / 128 = 7.7
- **Actual:** 6-7 jobs fit per node ✓
- **For 51 jobs:** Need 9 nodes (51 / 6 = 8.5, round up to 9)

---

#### 1.3 SDPB Timeout Configuration

Verify SDPB timeout is appropriate and configurable:

- [ ] Default SDPB timeout is 18000s (5 hours) in production scripts
- [ ] Timeout is configurable via `SDPB_TIMEOUT` environment variable
- [ ] Python CLI accepts `--sdpb-timeout` parameter
- [ ] Timeout setting is documented in `SDPB_RUNTIME_ENVELOPE_2026-02-11.md`

**Verification Commands:**
```bash
# Check if SDPB_TIMEOUT is used in scripts
grep -n "SDPB_TIMEOUT" jobs/stage_a_sdpb.slurm
grep -n "SDPB_TIMEOUT" jobs/test_sufficient_memory.slurm

# Check Python CLI help
conda activate ising_bootstrap
python -m ising_bootstrap.scans.stage_a --help | grep -i timeout
```

**Timeout Rationale:**
- Each SDPB solve: 2-2.2 hours observed
- 5-hour timeout allows for solver variability
- If solve exceeds 5h, it's likely stuck (fail fast, don't waste walltime)
- 36-hour walltime allows 7-8 SDPB timeouts before job killed

---

#### 1.4 Wrapper Script Defaults

Verify production-safe defaults in pipeline launcher scripts:

- [ ] `run_pipeline.sh` defaults to sapphire values (16 cores, 36h, 18000s timeout)
- [ ] `merge_stage_a_and_submit_b.slurm` defaults to sapphire values for Stage B
- [ ] Overrides still possible via environment variables for testing

**Verification Commands:**
```bash
# Check run_pipeline.sh defaults (lines 27, 35-41)
grep -A1 "SDPB_TIMEOUT=" jobs/run_pipeline.sh | head -2
# Should show: SDPB_TIMEOUT="${SDPB_TIMEOUT:-18000}"

grep -A1 "STAGE_A_CPUS=" jobs/run_pipeline.sh | head -2
# Should show: STAGE_A_CPUS="${STAGE_A_CPUS:-16}"

grep -A1 "STAGE_A_TIME=" jobs/run_pipeline.sh | head -2
# Should show: STAGE_A_TIME="${STAGE_A_TIME:-36:00:00}"

grep -A1 "STAGE_B_CPUS=" jobs/run_pipeline.sh | head -2
# Should show: STAGE_B_CPUS="${STAGE_B_CPUS:-16}"

grep -A1 "STAGE_B_TIME=" jobs/run_pipeline.sh | head -2
# Should show: STAGE_B_TIME="${STAGE_B_TIME:-36:00:00}"

# Check merge_stage_a_and_submit_b.slurm defaults (lines 28, 32-34)
grep "SDPB_TIMEOUT=\|STAGE_B_CPUS=\|STAGE_B_TIME=" jobs/merge_stage_a_and_submit_b.slurm
# Should show 18000, 16, 36:00:00
```

**Expected result:** All defaults aligned with sapphire partition (16 cores, 36h, 18000s)

**Why This Matters:**
- Wrapper scripts override .slurm file settings by passing explicit `--cpus-per-task`, `--time`, etc. to sbatch
- Without correct defaults, production runs would still use 8 cores/12h despite .slurm files saying sapphire
- This was the P0 issue found in external review (2026-02-12)

---

### 2. Documentation Review

#### 2.1 Overnight Failure Analysis

Verify comprehensive documentation exists:

- [ ] File exists: `docs/OVERNIGHT_TIMEOUT_ANALYSIS_2026-02-12.md`
- [ ] Document includes timeline of Jobs 59973738, 59973739 (2026-02-11/12)
- [ ] Performance metrics documented:
  - 8 cores: 2 iterations in 8h, ~2.2h per solve
  - 16 cores: 4 iterations in 8h, ~2.0h per solve
- [ ] Extrapolation shows 26-35 hours total time needed
- [ ] Document explains why shared partition is inadequate (12h < 26-35h)
- [ ] Sapphire advantages clearly quantified (7-day walltime, 990GB RAM, 6 jobs/node)

**Review Checklist:**
```bash
# Check document exists and is non-trivial
ls -lh docs/OVERNIGHT_TIMEOUT_ANALYSIS_2026-02-12.md
# Expected: ~30-50KB file size

# Verify key sections present
grep -E "^##" docs/OVERNIGHT_TIMEOUT_ANALYSIS_2026-02-12.md
# Expected: Executive Summary, Background, Timeline, Performance Analysis, Root Cause, etc.
```

**Key Questions:**
1. Does the analysis convincingly demonstrate 28-35h requirement? **Y / N**
2. Are performance metrics based on actual log data? **Y / N**
3. Is the sapphire solution well-justified? **Y / N**

---

#### 2.2 Migration Rationale

Verify sapphire partition choice is well-documented:

- [ ] Sapphire advantages table present (RAM, cores, walltime, MPI)
- [ ] Resource efficiency comparison: 9 nodes (sapphire) vs 51 nodes (shared)
- [ ] Expected timeline: 56-70 hours total (Stage A + B)
- [ ] No anticipated regressions (same SDPB solver, just more resources)
- [ ] Design fit explained: SDPB is exactly what sapphire is for (MPI, long, high-memory)

**Sapphire Design Match:**
- ✓ Long-running (28-35h per task)
- ✓ MPI-based (`mpirun -n 16` within SDPB)
- ✓ High-memory (128GB/job, but 990GB/node allows 6× packing)
- ✓ Arbitrary-precision solver (1024-bit mpfr)

---

#### 2.3 Updated Documentation

Verify existing documentation reflects sapphire migration:

- [ ] `CLAUDE.md` updated (Environment section mentions sapphire)
- [ ] `docs/CLUSTER_SETUP.md` updated (sapphire section added or partition table updated)
- [ ] `docs/PROGRESS.md` updated (migration timeline added)
- [ ] Memory instructions updated (if applicable)
- [ ] README or RUN docs mention sapphire (if they reference partition)

**Verification:**
```bash
grep -i sapphire CLAUDE.md docs/CLUSTER_SETUP.md docs/PROGRESS.md
# Expected: Multiple matches showing sapphire usage documented
```

---

### 3. Artifact Cleanup

#### 3.1 Failed Test Files Moved

Verify empty/failed test artifacts are archived:

- [ ] `data/test_config_0.csv` removed from main data directory
- [ ] `data/test_config_1.csv` removed from main data directory
- [ ] Both archived to `logs/archive/2026-02-overnight-timeout/` with descriptive names
- [ ] Archive directory has README explaining contents

**Verification:**
```bash
# Should NOT exist in data/
ls data/test_config_*.csv 2>&1
# Expected: "No such file or directory"

# Should exist in archive with descriptive names
ls logs/archive/2026-02-overnight-timeout/test_config_*.csv
# Expected: test_config_0_job59973738_TIMEOUT.csv, test_config_1_job59973739_TIMEOUT.csv

# Archive README should exist
ls logs/archive/2026-02-overnight-timeout/README.md
# Expected: File exists
```

---

#### 3.2 Log Archive Organization

Verify logs are properly archived and main logs directory is clean:

- [ ] Archive directory created: `logs/archive/2026-02-overnight-timeout/`
- [ ] Archive directory created: `logs/archive/2026-02-10-stage-a-timeout/`
- [ ] Both archive directories have READMEs explaining contents
- [ ] Overnight timeout logs (59973738, 59973739) preserved
- [ ] Main `logs/` directory clean (only current run logs remain)

**Verification:**
```bash
# Check archive structure
ls -lh logs/archive/
# Expected: Two directories (2026-02-overnight-timeout, 2026-02-10-stage-a-timeout)

# Check overnight timeout archive
ls logs/archive/2026-02-overnight-timeout/
# Expected: test_config CSVs, README.md, possibly symlinks to original logs

# Check for README content
head logs/archive/2026-02-overnight-timeout/README.md
# Expected: Explanation of contents and link to OVERNIGHT_TIMEOUT_ANALYSIS doc
```

---

### 4. Pre-Flight Test Plan

Before launching the full 51-task production array, validate sapphire with a single-point test:

#### 4.1 Single-Point Test Configuration

- [ ] Test script ready: `jobs/test_sufficient_memory.slurm` configured for sapphire
- [ ] Test uses production configuration: 16 cores, 128GB, 36h walltime
- [ ] Test point: Δσ=0.518 (the hardest/slowest point, near the Ising critical point)
- [ ] SDPB timeout: 18000s (5 hours, same as production)
- [ ] Output CSV specified for easy validation

**Test Command:**
```bash
cd /n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap

# Submit single-point test
sbatch --partition=sapphire --mem=128G --cpus-per-task=16 --time=36:00:00 \
  --export=ALL,SDPB_TIMEOUT=18000,SIGMA=0.518,OUTPUT_CSV=data/sapphire_test.csv \
  jobs/test_sufficient_memory.slurm

# Monitor submission
squeue -u $USER
# Expected: Job appears in queue with state PD (pending) or R (running)
```

---

#### 4.2 Single-Point Success Criteria

After test completes (24-32 hours expected), verify:

- [ ] Job status: COMPLETED (not TIMEOUT, not OUT_OF_MEMORY, not FAILED)
- [ ] CSV file created: `data/sapphire_test.csv`
- [ ] CSV contains data row (not just header)
- [ ] Δε_max value is finite (not NaN, not inf)
- [ ] Δε_max value in expected range: 1.3 < Δε_max < 1.5 (literature ≈ 1.41 at Δσ≈0.5182)
- [ ] Log shows 12-16 bisection iterations completed
- [ ] No SDPB "inconclusive" errors in log
- [ ] No SDPB "timeout" errors in log (should complete within 5h per solve)

**Verification Commands (after job completes):**
```bash
# Check job status
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
# Expected: State=COMPLETED

# Check CSV
cat data/sapphire_test.csv
# Expected:
# delta_sigma,delta_eps_max
# 0.518,1.41234567...

# Validate value is finite and in range
python3 << 'EOF'
import csv
with open('data/sapphire_test.csv') as f:
    reader = csv.DictReader(f)
    row = next(reader)
    eps_max = float(row['delta_eps_max'])
    assert 1.3 < eps_max < 1.5, f"Unexpected Δε_max = {eps_max}"
    print(f"✓ Valid result: Δε_max = {eps_max:.4f}")
EOF

# Count bisection iterations in log
grep "Writing PMP JSON" logs/test_sufficient_memory_<JOBID>.log | wc -l
# Expected: 12-18 lines (one per iteration)

# Check for errors
grep -i "timeout\|inconclusive\|error" logs/test_sufficient_memory_<JOBID>.log | grep -v "primalError\|dualError"
# Expected: Minimal output (no critical errors)
```

---

#### 4.3 If Single-Point Test Fails

If the test fails, diagnose before proceeding:

**Possible Failures:**
1. **TIMEOUT** → Increase walltime to 48h or 72h, try again
2. **OUT_OF_MEMORY** → Increase mem to 160G or 192G, try again
3. **NaN result** → Check log for SDPB errors, may need longer timeout
4. **Inconclusive SDPB** → Numerical instability, check precision settings

**DO NOT proceed to full 51-task array until single-point test succeeds.**

---

### 5. Full Pipeline Readiness

After single-point test succeeds, verify pipeline is ready:

#### 5.1 Pipeline Scripts Verified

- [ ] Stage A script ready: `jobs/stage_a_sdpb.slurm` (array 0-50, sapphire, 16 cores, 36h)
- [ ] Stage B script ready: `jobs/stage_b_sdpb.slurm` (array 0-50, sapphire, 16 cores, 36h)
- [ ] Merge scripts unchanged (partition-agnostic, lightweight)
- [ ] Plot script unchanged (partition-agnostic, lightweight)
- [ ] All scripts executable and syntactically correct

**Verification:**
```bash
# Check array size
grep "array" jobs/stage_a_sdpb.slurm
# Expected: #SBATCH --array=0-50 (51 tasks)

grep "array" jobs/stage_b_sdpb.slurm
# Expected: #SBATCH --array=0-50 (51 tasks)

# Syntax check (dry run)
bash -n jobs/stage_a_sdpb.slurm
bash -n jobs/stage_b_sdpb.slurm
# Expected: No output (syntax OK)
```

---

#### 5.2 Expected Production Timeline

After single-point validation, full pipeline timeline:

**Stage A (51 Δσ points, parallel):**
- Expected runtime: 28-35 hours (limited by slowest task at Δσ=0.518)
- Resource usage: 51 jobs × 16 cores = 816 cores (4% of sapphire)
- Nodes needed: 9 nodes (51 jobs / 6 jobs per node)

**Stage B (51 Δσ points, parallel):**
- Expected runtime: 28-35 hours (also limited by slowest task)
- Same resource usage: 816 cores, 9 nodes

**Total Pipeline:**
- Stage A: 30 hours (average)
- Merge A: 5 minutes
- Stage B: 30 hours
- Merge B + Plot: 10 minutes
- **Total: ~60 hours (2.5 days)**

**Submission Commands (manual):**
```bash
# Stage A (after single-point success)
STAGE_A_JOB=$(sbatch --parsable --array=0-50 jobs/stage_a_sdpb.slurm)
echo "Stage A job: $STAGE_A_JOB"

# Wait for Stage A to complete, then merge and launch Stage B
# Use jobs/run_pipeline.sh for automated chaining, or manual sbatch
```

---

### 6. Git Repository State

#### 6.1 Commit Hygiene

Verify git repository is clean and well-organized:

- [ ] Migration on clean branch: `migration/sapphire-partition`
- [ ] Branch created from up-to-date `main`
- [ ] All changes committed with meaningful messages
- [ ] No uncommitted `.slurm.bak` files (sed backup artifacts)
- [ ] No untracked files that should be committed
- [ ] `.gitignore` properly excludes logs and data (but includes archive READMEs)

**Verification:**
```bash
git branch
# Expected: * migration/sapphire-partition

git status
# Expected: Clean working tree or only expected untracked files

# Check for .bak files
find jobs/ -name "*.slurm.bak"
# Expected: No output (backups cleaned up)

git log --oneline -10
# Expected: Meaningful commit messages about migration
```

---

#### 6.2 Recommended Commit Structure

Verify commits are atomic and logical:

Expected commit sequence:
1. **"Migrate all 18 SLURM scripts to sapphire partition"**
   - All .slurm file changes (partition, walltime, cores)
   - Added verification script

2. **"Document overnight timeout failures and sapphire rationale"**
   - Created `OVERNIGHT_TIMEOUT_ANALYSIS_2026-02-12.md`

3. **"Add external review checklist for sapphire migration"**
   - Created `SAPPHIRE_MIGRATION_CHECKLIST.md`

4. **"Archive failed overnight test artifacts"**
   - Moved test CSVs to archive
   - Created archive READMEs

5. **"Update documentation for sapphire partition usage"**
   - Updated CLUSTER_SETUP.md, PROGRESS.md, CLAUDE.md
   - Updated memory instructions

6. **"Verify sapphire migration complete"**
   - Ran verification script
   - Updated PROGRESS.md with verification results

**Check:**
```bash
git log --oneline migration/sapphire-partition ^main
# Should show 4-6 commits with clear descriptions
```

---

#### 6.3 No Breaking Changes

Verify backward compatibility preserved:

- [ ] Python code unchanged (partition-agnostic)
- [ ] Environment variables unchanged (SDPB_TIMEOUT still works)
- [ ] CLI unchanged (same `python -m ising_bootstrap.scans.stage_a` interface)
- [ ] Cache files unchanged (still use ext_cache_consolidated.npz)
- [ ] Scripts can still be run individually (not dependent on pipeline automation)

**Test individual script:**
```bash
# Should work (won't actually run without sbatch, but syntax OK)
bash -n jobs/stage_a_sdpb.slurm
# Expected: No errors
```

---

### 7. External Review Sign-Off

#### 7.1 Reviewer Actions

**Mandatory steps for reviewer:**
- [ ] Read `docs/OVERNIGHT_TIMEOUT_ANALYSIS_2026-02-12.md` (15-20 minutes)
- [ ] Verify partition change: Run `bash jobs/verify_partition_migration.sh`
- [ ] Spot-check 3-5 random .slurm files (sed -n '4p' jobs/*.slurm)
- [ ] Check archive structure: `ls logs/archive/*/`
- [ ] Review git commits: `git log migration/sapphire-partition`
- [ ] Confirm understanding of test plan (Section 4)

**Estimated review time:** 30-60 minutes

---

#### 7.2 Reviewer Questions to Answer

**1. Is the overnight failure analysis convincing?**
   - [ ] Yes, the 28-35h requirement is clearly demonstrated from actual data
   - [ ] No, need more data: _______________________

**2. Is the sapphire partition appropriate for this workload?**
   - [ ] Yes, high-memory, long-running MPI jobs fit sapphire's design
   - [ ] No, concern: _______________________

**3. Are the resource requests (16 cores, 128GB, 36h) reasonable?**
   - [ ] Yes, conservative given 28-35h estimate with 10% safety margin
   - [ ] No, recommend different config: cores=____ mem=____ time=____

**4. Is the single-point test plan adequate before full 51-task array?**
   - [ ] Yes, validates partition behavior and avoids wasting 51×36h if something is wrong
   - [ ] No, suggest additional testing: _______________________

**5. Is the documentation sufficient for future maintenance?**
   - [ ] Yes, can reproduce reasoning and undo migration if needed
   - [ ] No, missing: _______________________

**6. Are there any concerns about sapphire resource usage (4% of cluster)?**
   - [ ] No concerns, 816 cores / 20,832 total is reasonable
   - [ ] Yes, concern: _______________________

---

#### 7.3 Final Approval

**Single-Point Test Approval:**
- [ ] **APPROVED** to run single-point test on sapphire
  - Reviewer: _________________ Date: _________
  - Notes: _________________________________________

**Full Production Approval (after single-point success):**
- [ ] **APPROVED** to run full 51-task production pipeline
  - Reviewer: _________________ Date: _________
  - Conditions (if any): ___________________________

**If NOT approved:**
- [ ] **CHANGES REQUESTED**
  - Specific changes needed: _______________________
  - Re-review after changes: Y / N

---

### 8. Post-Migration Validation

After first successful production run on sapphire, validate results:

#### 8.1 Performance Validation

Collect and verify performance metrics:

- [ ] Record actual Stage A walltime (per task and total)
  - Expected: 24-35 hours per task, ~30 hours total (parallel)
  - Actual: ________ hours

- [ ] Verify SDPB iteration counts (should be 12-16 per task)
  - Sample task logs and count "Writing PMP JSON" lines
  - Actual range: ________ iterations

- [ ] Check sapphire resource efficiency via sacct:
  - CPU utilization: `sacct -j <JOBID> --format=JobID,CPUTime,Elapsed`
  - Memory peak: `sacct -j <JOBID> --format=JobID,MaxRSS`
  - Expected: ~90% CPU utilization, ~100-120GB peak memory

- [ ] Compare to shared partition baseline (none available, but no regressions expected)

---

#### 8.2 Scientific Validation

Verify results match expected physics:

- [ ] Stage A CSV has 51 finite values (one per Δσ)
  - No NaN or inf values
  - File: `data/eps_bound.csv`

- [ ] At Δσ ≈ 0.5182 (Ising point): Δε_max ≈ 1.41
  - Literature value (arXiv:1203.6064): Δε ≈ 1.412625(10)
  - Tolerance: ±0.01 acceptable

- [ ] Stage B CSV has 51 finite values (one per Δσ)
  - File: `data/epsprime_bound.csv`

- [ ] At Δσ ≈ 0.5182: Δε'_max ≈ 3.84
  - Literature value (arXiv:1203.6064): Δε' ≈ 3.83(1)
  - Tolerance: ±0.1 acceptable

- [ ] Figure 6 plot shows characteristic kink/spike below Ising point
  - File: `figures/fig6_reproduction.png`
  - Visual inspection: Matches Fig 6 from paper

**Validation Command:**
```bash
# Check CSV has expected number of lines
wc -l data/eps_bound.csv
# Expected: 52 lines (1 header + 51 data rows)

# Check for NaN or inf
grep -E "nan|inf" data/eps_bound.csv
# Expected: No output

# Extract Ising point value (Δσ≈0.518)
grep "0.518" data/eps_bound.csv
# Expected: 0.518,1.41... (value near 1.41)
```

---

#### 8.3 Documentation Update

After successful run, finalize documentation:

- [ ] Update `docs/PROGRESS.md` with sapphire migration completion date
- [ ] Add performance data to `OVERNIGHT_TIMEOUT_ANALYSIS_2026-02-12.md` appendix:
  - Actual walltime observed
  - Actual iteration counts
  - Resource utilization stats

- [ ] Update memory instructions (if not already done):
  - `/n/home09/obarrera/.claude/projects/-n-holylabs-schwartz-lab-Lab-obarrera/memory/MEMORY.md`
  - Add: "Now running on sapphire: 16 cores, 36h walltime, completed in X hours"

- [ ] Merge `migration/sapphire-partition` branch to `main`:
  ```bash
  git checkout main
  git merge --no-ff migration/sapphire-partition
  git push origin main
  ```

---

## Rollback Plan (If Needed)

If sapphire partition proves unsuitable after testing:

### Option A: Revert to Shared Partition

**Steps:**
```bash
git checkout main  # Return to pre-migration state
# OR
for file in jobs/*.slurm; do
    sed -i '4s/^#SBATCH --partition=sapphire$/#SBATCH --partition=shared/' "$file"
done
```

**Limitations:** Still have 12h walltime problem, not a solution

---

### Option B: Reduce Problem Size to Fit Shared Partition

**Approach:** Reduce n_max from 10 to 8
- Fewer derivatives → smaller SDP problem → faster SDPB solves
- May fit within 12h walltime limit
- **Trade-off:** Coarser bounds, less accurate results

**Changes:**
```python
# In src/ising_bootstrap/config.py
N_MAX = 8  # Was 10
```

**Expected impact:**
- SDPB solve time: ~40-50% faster (rough estimate)
- Total time: 12-18 hours (might fit 12h limit with luck)
- Bound quality: Slightly worse (larger allowed regions)

**Documented:** Can try this as fallback

---

### Option C: Request Custom Walltime Exception

**Contact:** Harvard FAS RC support
**Request:** 48-hour walltime exception for randall_lab on shared partition
**Justification:** Scientific computation (conformal bootstrap) requires long SDPB solves

**Pros:** Keeps using shared partition (familiar)
**Cons:** May be denied, not guaranteed, affects other users

---

## Summary Checklist

Before approving migration:

**Code:**
- [ ] ✓ All 18 .slurm files use sapphire
- [ ] ✓ Resources appropriate (16 cores, 128GB, 36h)
- [ ] ✓ Verification script passes

**Documentation:**
- [ ] ✓ Overnight failure analysis complete and convincing
- [ ] ✓ Sapphire advantages quantified
- [ ] ✓ Existing docs updated

**Artifacts:**
- [ ] ✓ Failed test CSVs archived
- [ ] ✓ Logs organized
- [ ] ✓ Archive READMEs present

**Git:**
- [ ] ✓ Clean branch with atomic commits
- [ ] ✓ No breaking changes
- [ ] ✓ Ready to merge after validation

**Testing:**
- [ ] ✓ Single-point test plan clear
- [ ] ✓ Success criteria defined
- [ ] ✓ Failure scenarios documented

---

## Approval Signature

**I have reviewed the sapphire partition migration and:**

☐ **APPROVE** single-point test on sapphire
- Reviewer Name: _______________________
- Date: _______________________
- Notes: _______________________

☐ **APPROVE** full 51-task production run (after single-point success)
- Reviewer Name: _______________________
- Date: _______________________
- Estimated completion: 2.5-3 days from approval

☐ **REQUEST CHANGES** (specify): _______________________

---

**Checklist Version:** 1.0 (2026-02-12)
**Associated Documents:** `OVERNIGHT_TIMEOUT_ANALYSIS_2026-02-12.md`
**Next Review:** After single-point sapphire test completion
