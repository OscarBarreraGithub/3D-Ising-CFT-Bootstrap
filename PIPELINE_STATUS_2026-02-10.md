# Pipeline Submission — 2026-02-10

## ✅ Complete Pipeline Submitted with Fixes

All SDPB MPI and memory fixes have been applied and the full pipeline is now running.

### Jobs Submitted

| Order | Job ID | Name | Status | Depends On | What It Does |
|-------|--------|------|--------|------------|--------------|
| 1 | 59716207[0-50] | `stage_a_sdpb` | **PENDING** | — | Stage A: Upper bound on Δε vs Δσ (51 tasks) |
| 2 | 59716230 | `merge_a_submit_b` | **PENDING** | Job 59716207 | Merge Stage A CSVs, validate, auto-submit Stage B + plot |
| 3 | (auto) | `stage_b_sdpb` | Will be submitted by Job 2 | Job 2 | Stage B: Upper bound on Δε' vs Δσ (51 tasks) |
| 4 | (auto) | `final_merge_and_plot` | Will be submitted by Job 2 | Job 3 | Merge Stage B, generate Figure 6 |

### Fixes Applied

**From [docs/SDPB_MPI_FIX.md](docs/SDPB_MPI_FIX.md):**

1. ✅ **SDPB MPI fix**: Added `--oversubscribe` flag to mpirun command ([sdpb.py:225](src/ising_bootstrap/lp/sdpb.py#L225))
2. ✅ **Stage A memory**: Increased from 16G → 48G ([stage_a_sdpb.slurm](jobs/stage_a_sdpb.slurm))
3. ✅ **Stage B memory**: Increased from 8G → 48G ([stage_b_sdpb.slurm](jobs/stage_b_sdpb.slurm))
4. ✅ **Stage B CPUs**: Increased from 4 → 8 cores
5. ✅ **Old results cleaned**: Removed all `eps_bound_*.csv` and `epsprime_bound_*.csv` files

### Expected Timeline

- **Stage A** (59716207): ~1-4 hours per task × 51 tasks (8h wall time limit)
  - Most tasks should complete in parallel
  - Watch for first results in ~30-60 min
- **Merge A** (59716230): ~1-2 minutes (runs after Stage A completes)
- **Stage B** (auto-submitted): ~1-4 hours per task × 51 tasks
- **Final plot** (auto-submitted): ~1 minute

**Total estimated time:** 4-16 hours depending on queue and solver performance

### How to Monitor

```bash
# Check job queue status
squeue -u obarrera

# Watch Stage A progress (task 0 as example)
tail -f logs/stage_a_sdpb_59716207_0.log

# Check for completed results
ls -lh data/eps_bound_*.csv | wc -l
# Should grow from 0 to 51 as tasks complete

# Check first result (when available)
head data/eps_bound_0.csv
# Should NOT be 2.499939 (old bug) or 0.500000 (old scipy bug)

# Check for MPI errors (should be none now)
grep -i "oversubscribe\|not enough slots" logs/stage_a_sdpb_59716207_0.log

# Check for OOM kills (should be none now)
grep -i "oom" logs/stage_a_sdpb_59716207_*.log

# Monitor resource usage (after job runs for a while)
sacct -j 59716207 --format=JobID,State,MaxRSS,Elapsed | head -20
```

### What Success Looks Like

#### Stage A Logs (e.g., task 0)

```
Loading block cache...
  Loading consolidated cache: .../ext_cache_consolidated.npz
  Loaded 520474 extended block arrays from consolidated cache

[1/1] Δσ = 0.5000
  Constraint matrix: 520476 operators (187401 scalars, 333075 spinning)
  Writing PMP JSON (470476 blocks, 66 vars) ...
  Running pmp2sdp (precision=1024) ...
  Running sdpb (8 cores, precision=1024) ...

  # ← Should see terminateReason output, not MPI errors

  Δε_max = 1.XXXXX  (N iterations)  # ← NOT 2.499939!

Scan complete. Results written to data/eps_bound_0.csv
```

#### Stage A Results

File: `data/eps_bound_0.csv` (and similar for tasks 1-50)
```csv
delta_sigma,delta_eps_max
0.500000,1.234567  # ← Varied values, NOT all 2.5
```

At **Δσ ≈ 0.5182** (task ~32), expect **Δε_max ≈ 1.41** (from paper)

#### Stage B Results

File: `data/epsprime_bound.csv` (after merge)
```csv
delta_sigma,delta_eps,delta_eps_prime_max
0.500000,1.234567,3.456789
```

At **Δσ ≈ 0.5182**, expect **Δε'_max ≈ 3.84** (from paper)

#### Final Output

File: `figures/fig6_reproduction.png`

Upper bound curve showing Δε' vs Δσ, with sharp spike near the Ising point.

### What to Do If Things Fail

#### All Δε_max Still = 2.5

**Problem:** SDPB still failing
**Check:**
```bash
grep "terminateReason\|mpirun.*error\|not enough slots" logs/stage_a_sdpb_59716207_0.log
```
**Solution:** Verify `--oversubscribe` is in the mpirun command at [sdpb.py:225](src/ising_bootstrap/lp/sdpb.py#L225)

#### OOM Kills Still Happening

**Problem:** 48G still insufficient
**Check:**
```bash
grep -i oom logs/stage_a_sdpb_59716207_*.log
sacct -j 59716207 --format=JobID,MaxRSS,State
```
**Solution:** Increase to 64G or reduce MPI cores to 4 (reduces parallelism but saves memory)

#### Merge Job Aborts with "All results ≤ 0.5"

**Problem:** SDPB returning "excluded" for all gaps (opposite of previous bug)
**Check:** Log files for SDPB termination reasons
**Solution:** Check SDPB precision or timeout settings

#### Some Tasks Timeout (8h limit)

**Problem:** SDPB solver taking too long on certain Δσ values
**Check:** Which tasks hit timeout
```bash
sacct -j 59716207 --format=JobID,State,Elapsed | grep TIMEOUT
```
**Solution:** Individual tasks can be re-run with higher time limit or lower precision

### Auto-Submission Chain

The merge job (`merge_a_and_submit_b.slurm`) will automatically:

1. **Merge Stage A CSVs** → `data/eps_bound.csv`
2. **Validate results:**
   - Check at least 10 data points exist
   - Check NOT all values are ≤ 0.5 (old scipy bug)
3. **Submit Stage B** → Job ID will be printed in merge log
4. **Submit plot job** with dependency on Stage B

**No manual intervention needed** — the pipeline will run to completion.

### Quick Status Check

Run this command to see the current state:

```bash
echo "=== Job Status ==="
squeue -u obarrera
echo ""
echo "=== Stage A Results ==="
ls data/eps_bound_*.csv 2>/dev/null | wc -l
echo "files (expect 51 when complete)"
echo ""
echo "=== Stage B Results ==="
ls data/epsprime_bound_*.csv 2>/dev/null | wc -l
echo "files (expect 51 when complete)"
echo ""
echo "=== Final Output ==="
ls -lh figures/fig6_reproduction.* 2>/dev/null || echo "Not yet generated"
```

### Documentation References

- **Full diagnosis:** [docs/SDPB_MPI_FIX.md](docs/SDPB_MPI_FIX.md)
- **Changelog:** [CHANGELOG_2026-02-10.md](CHANGELOG_2026-02-10.md)
- **Project guide:** [CLAUDE.md](CLAUDE.md)
- **Progress tracker:** [docs/PROGRESS.md](docs/PROGRESS.md)
- **LP conditioning bug (historical):** [docs/LP_CONDITIONING_BUG.md](docs/LP_CONDITIONING_BUG.md)

### Next Steps (After Pipeline Completes)

1. **Validate results against paper:**
   - Δε_max ≈ 1.41 at Δσ ≈ 0.5182
   - Δε'_max ≈ 3.84 at Δσ ≈ 0.5182

2. **Check Figure 6:**
   - Sharp spike in Δε' below Ising point
   - Upper bound curve matches paper qualitatively

3. **Archive results:**
   ```bash
   tar -czf results_$(date +%Y%m%d).tar.gz \
       data/eps_bound.csv \
       data/epsprime_bound.csv \
       figures/fig6_reproduction.*
   ```

4. **Clean up logs (optional):**
   ```bash
   # Keep only successful run logs
   mkdir -p logs/archive_59716207
   mv logs/stage_a_sdpb_59716207_*.log logs/archive_59716207/
   ```

---

**Pipeline submitted:** 2026-02-10
**Job IDs:** 59716207 (Stage A), 59716230 (merge + auto-submit)
**Status:** Running with all fixes applied ✅
**Expected completion:** 4-16 hours
