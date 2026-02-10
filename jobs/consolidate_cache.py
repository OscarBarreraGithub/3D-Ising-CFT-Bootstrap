#!/usr/bin/env python3
"""
Consolidate 520K individual .npy block cache files into a single .npz archive.

Problem: Loading 520K individual files from NFS takes 60+ minutes per SLURM task
because each np.load() is a separate NFS syscall with ~3-10ms latency.

Solution: Pack all files into one .npz archive. Loading the archive takes ~10-30s
(single NFS read) instead of 60+ minutes (520K NFS reads).

Usage:
    python jobs/consolidate_cache.py

    Or via SLURM:
    sbatch jobs/consolidate_cache.slurm

Input:  data/cached_blocks/ext_d*.npy  (~520K files, ~1 GB total)
Output: data/cached_blocks/ext_cache_consolidated.npz  (~1 GB)

The key format in the .npz archive is "{delta}_{spin}" where delta and spin
are parsed from the filename ext_d{delta}_l{spin}.npy. For example:
    ext_d0.50002000_l0.npy  →  key "0.50002000_0"
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cached_blocks"
OUTPUT_FILE = CACHE_DIR / "ext_cache_consolidated.npz"


def main():
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Output file: {OUTPUT_FILE}")

    if OUTPUT_FILE.exists():
        size_mb = OUTPUT_FILE.stat().st_size / 1e6
        print(f"WARNING: Output file already exists ({size_mb:.0f} MB)")
        print("Delete it first if you want to rebuild: rm", OUTPUT_FILE)
        sys.exit(1)

    # List all ext_*.npy files
    print("Listing cache directory...")
    t0 = time.time()
    all_files = sorted(f for f in os.listdir(CACHE_DIR)
                       if f.startswith("ext_") and f.endswith(".npy"))
    t1 = time.time()
    print(f"Found {len(all_files)} files ({t1 - t0:.1f}s)")

    if len(all_files) == 0:
        print("ERROR: No ext_*.npy files found in", CACHE_DIR)
        sys.exit(1)

    # Load all arrays into a dict
    print(f"Loading {len(all_files)} arrays...")
    arrays = {}
    errors = 0
    t0 = time.time()

    for i, fname in enumerate(all_files):
        if i % 10000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(all_files) - i) / rate
            print(f"  [{i:>7d}/{len(all_files)}]  {rate:.0f} files/s  ETA {eta/60:.1f} min")

        # Parse filename: ext_d0.50002000_l0.npy → key "0.50002000_0"
        # Strip "ext_d" prefix and ".npy" suffix
        stem = fname[5:-4]  # "0.50002000_l0"
        parts = stem.split("_l")
        if len(parts) != 2:
            print(f"  WARNING: Unexpected filename format: {fname}")
            errors += 1
            continue

        delta_str, spin_str = parts[0], parts[1]
        key = f"{delta_str}_{spin_str}"

        try:
            arr = np.load(CACHE_DIR / fname)
            arrays[key] = arr
        except Exception as e:
            print(f"  WARNING: Failed to load {fname}: {e}")
            errors += 1

    t1 = time.time()
    print(f"Loaded {len(arrays)} arrays in {t1 - t0:.1f}s ({errors} errors)")

    # Save consolidated archive
    print(f"Writing consolidated archive ({len(arrays)} arrays)...")
    print("This may take a few minutes for ~1 GB of data...")
    t0 = time.time()
    np.savez(OUTPUT_FILE, **arrays)
    t1 = time.time()

    size_mb = OUTPUT_FILE.stat().st_size / 1e6
    print(f"Done in {t1 - t0:.1f}s")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Size: {size_mb:.0f} MB")
    print(f"Arrays: {len(arrays)}")

    # Quick verification
    print("\nVerifying...")
    data = np.load(OUTPUT_FILE)
    assert len(data.files) == len(arrays), f"Mismatch: {len(data.files)} vs {len(arrays)}"
    # Check one random array
    sample_key = data.files[0]
    sample = data[sample_key]
    assert sample.ndim == 2, f"Expected 2D array, got {sample.ndim}D"
    print(f"Verification passed: {len(data.files)} arrays, sample shape {sample.shape}")
    data.close()


if __name__ == "__main__":
    main()
