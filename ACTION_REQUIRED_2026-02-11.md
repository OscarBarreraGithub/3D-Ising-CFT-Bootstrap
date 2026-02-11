# ⚠️ ACTION REQUIRED (2026-02-11)

**Date:** 2026-02-11
**Status:** Strict semantics merged locally, documentation updated, ready to push

---

## 🔴 IMMEDIATE ACTIONS (Git)

You need to push the changes to GitHub:

```bash
cd /n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap

# 1. Push merged main branch
git push origin main

# 2. Delete the remote Codex branch
git push origin --delete codex/strict-failfast-stageb-snap-eps

# 3. Verify
git branch -a | grep codex  # Should show nothing
```

---

## 📋 WHAT WAS DONE

### Code Changes ✓
- [x] Merged `codex/strict-failfast-stageb-snap-eps` branch to `main`
- [x] Strict SDPB failure handling (inconclusive outcomes = failure)
- [x] Fail-fast Stage A/B (ANY solver anomaly aborts immediately)
- [x] Stage B epsilon anchoring (snaps to scalar grid with tolerance)
- [x] Hardened merge gates (validates Stage A before Stage B)
- [x] SDPB timeout exposed end-to-end (`--sdpb-timeout` CLI flag)
- [x] All 75 tests passing (17 SDPB + 28 Stage A + 30 Stage B)

### Documentation Updates ✓
- [x] Created `HANDOFF_2026-02-11_STRICT_SEMANTICS_MERGED.md` (comprehensive handoff)
- [x] Created `STATUS_2026-02-11.md` (project status summary)
- [x] Updated `CLAUDE.md` (pipeline status, strict semantics, common pitfalls)
- [x] Updated global memory file (latest status)
- [x] All stale references updated

### Local Git ✓
- [x] Merge commit: `216890d`
- [x] Documentation commit: `4ff6daa`
- [x] Working tree clean
- [x] Ready to push (6 commits ahead of origin/main)

---

## 🎯 NEXT STEPS (After Pushing)

### Step 1: Runtime Envelope Characterization

**Goal:** Find the correct SDPB timeout for production runs

**Command:**
```bash
bash jobs/submit_stage_a_runtime_envelope.sh
```

**Check result:**
```bash
# Wait for job to complete, then:
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem
cat data/test_sufficient_memory.csv
tail -100 logs/test_sufficient_memory_<JOBID>.log
```

**Success:** CSV shows `0.518000,1.41XXXX` (not NaN, not 2.5)

**If timed out:** Increase timeout and retry:
```bash
TIMEOUT=3600 bash jobs/submit_stage_a_runtime_envelope.sh
```

**If still timed out:** Try more cores:
```bash
TIMEOUT=3600 CPUS=16 MEM=160G WALLTIME=08:00:00 \
  bash jobs/submit_stage_a_runtime_envelope.sh
```

---

### Step 2: Pilot Run (After Step 1 Success)

Once you know the correct timeout (e.g., 3600s with 16 cores):

```bash
sbatch --array=0,9,18,27,36 \
  --cpus-per-task=16 \
  --mem=160G \
  --time=12:00:00 \
  --export=ALL,SDPB_TIMEOUT=3600,STAGE_A_TOLERANCE=1e-4 \
  jobs/stage_a_pilot_sdpb.slurm

# Merge pilot results
bash jobs/merge_stage_a_pilot.sh

# Check results
cat data/eps_bound_pilot.csv
```

---

### Step 3: Full Pipeline (After Step 2 Success)

```bash
sbatch --array=0-50 \
  --cpus-per-task=16 \
  --mem=160G \
  --time=12:00:00 \
  --export=ALL,SDPB_TIMEOUT=3600,STAGE_A_TOLERANCE=1e-4 \
  jobs/stage_a_sdpb.slurm
```

---

## 📚 DOCUMENTATION INDEX

All documentation has been updated. Here's where to find everything:

### Quick Start
1. **This file** - Action items and next steps
2. `STATUS_2026-02-11.md` - Project status summary

### Main Handoff
- `HANDOFF_2026-02-11_STRICT_SEMANTICS_MERGED.md` - Complete context and workflow

### Runtime Workflow
- `docs/SDPB_RUNTIME_ENVELOPE_2026-02-11.md` - Step-by-step envelope characterization

### Reference
- `CLAUDE.md` - Pipeline overview, strict semantics, common pitfalls
- `docs/PROGRESS.md` - Full implementation history (milestones 0-7)

### Historical (Superseded)
- `HANDOFF_2026-02-10_ERROR_HANDLING_FIX.md` - Previous handoff (archive)

---

## ✅ WHAT'S READY

- ✅ All code implemented and tested
- ✅ Strict failure semantics prevent silent bugs
- ✅ Block cache consolidated (fast NFS loading)
- ✅ SDPB backend integrated (arbitrary precision)
- ✅ Stage B epsilon anchoring (robust)
- ✅ Merge gates validate data quality
- ✅ Timeout configurable end-to-end
- ✅ Helper scripts for characterization
- ✅ All documentation up to date

---

## ⏳ WHAT'S BLOCKING

**SDPB timeout needs characterization**
- Default: 1800s (30 min)
- Likely need: 3600s (60 min) with 16 cores
- Must verify with envelope job first

---

## 🔧 EXPECTED TIMELINE

**After you push git changes:**

| Step | Time | Cumulative |
|------|------|------------|
| 1. Push to GitHub | 1 min | 1 min |
| 2. Envelope job (baseline) | 30-60 min | ~1 hour |
| 3. Envelope job (if retry) | 1-2 hours | ~3 hours |
| 4. Pilot (5 points) | 2-6 hours | ~9 hours |
| 5. Full Stage A (51 points) | 12-24 hours | ~33 hours |
| 6. Stage B (51 points) | 12-24 hours | ~57 hours |
| 7. Final merge + plot | 5 min | ~57 hours |

**Total:** 2-3 days (mostly queue wait time + SDPB solve time)

---

## 🎓 KEY LEARNINGS

### What We Fixed
1. **LP conditioning** (scipy→SDPB): condition number 4e16 → 1024-bit precision
2. **NFS I/O** (520K files→1 .npz): 60+ min → 10-30s cache loading
3. **SDPB MPI** (`--oversubscribe`): MPI errors → clean execution
4. **Error handling** (strict semantics): silent failures → explicit NaN + abort
5. **Stage B anchoring** (epsilon snap): missing scalars → validated anchor

### What We Learned
- SDPB solves are EXPENSIVE at n_max=10 (470K blocks)
- Strict fail-fast is CORRECT but requires proper timeout
- Timeout must be characterized, not guessed
- More cores likely help (SDPB scales well with MPI)

---

## 🚨 IMPORTANT REMINDERS

### With Strict Semantics
- **ANY timeout → NaN** (no retry, no fallback)
- **Must configure timeout BEFORE full run**
- **Envelope characterization is NOT optional**

### Don't Skip Steps
1. ❌ Don't run full array before envelope
2. ❌ Don't skip pilot before full run
3. ❌ Don't guess timeout values

### Do This Instead
1. ✅ Envelope first (find timeout)
2. ✅ Pilot second (validate timeout)
3. ✅ Full run third (production)

---

## 📞 QUESTIONS?

If something goes wrong:
1. Check `HANDOFF_2026-02-11_STRICT_SEMANTICS_MERGED.md` decision trees
2. Check `CLAUDE.md` "What Failure Looks Like" section
3. Check job logs: `tail -200 logs/test_sufficient_memory_<JOBID>.log`
4. Check `sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS`

---

## ✨ SUMMARY

**Status:** Ready to push and start runtime characterization
**Blocker:** SDPB timeout needs tuning (not a bug, just configuration)
**Next action:** Push git changes, then run envelope job
**Expected outcome:** Will find correct timeout within 1-3 tries

**You're very close! Just need to tune the timeout. 🚀**

---

**Created:** 2026-02-11 by Claude
