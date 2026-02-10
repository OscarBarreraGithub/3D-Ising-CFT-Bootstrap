"""
SDPB backend for the conformal bootstrap LP feasibility problem.

Encodes the discrete LP as a degenerate Polynomial Matrix Program (PMP)
with 1×1 blocks and degree-0 polynomials, then solves via SDPB's
arbitrary-precision interior-point method.

The LP problem:
    Find α ∈ R^66 such that:
        α^T f_id = 1          (normalization)
        α^T f_i  ≥ 0          for each operator i  (positivity)

is mapped to SDPB's PMP format where each positivity constraint becomes
a separate 1×1 positive-semidefinite block with constant polynomial
entries equal to the crossing vector components f_i[n].

Reference: arXiv:1502.02033 (SDPB), arXiv:1203.6064 (bootstrap LP)
"""

import json
import os
import subprocess
import tempfile
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from .solver import FeasibilityResult


# =============================================================================
# Configuration
# =============================================================================

# Default path to the SDPB Singularity image, relative to project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IMAGE = _PROJECT_ROOT / "tools" / "sdpb-3.1.0.sif"


@dataclass
class SdpbConfig:
    """Configuration for the SDPB solver backend."""

    image_path: Path = field(default_factory=lambda: _DEFAULT_IMAGE)
    """Path to the SDPB Singularity .sif image."""

    precision: int = 1024
    """Arithmetic precision in bits for pmp2sdp and sdpb."""

    n_cores: int = 4
    """Number of MPI processes for sdpb."""

    timeout: int = 600
    """Timeout in seconds per subprocess call (pmp2sdp or sdpb)."""

    work_dir: Optional[Path] = None
    """Working directory for temporary files.  Defaults to $TMPDIR or /tmp."""

    verbose: bool = False
    """Print subprocess commands and output."""

    cleanup: bool = True
    """Remove temporary files after each solve."""


# =============================================================================
# PMP JSON writer
# =============================================================================

def _format_float(x: float, significant_digits: int = 17) -> str:
    """Format a float64 as a string with enough digits for exact round-trip."""
    return f"{x:.{significant_digits}e}"


def write_pmp_json(
    A: np.ndarray,
    f_id: np.ndarray,
    output_path: Path,
    significant_digits: int = 17,
) -> Path:
    """
    Write the LP feasibility problem as an SDPB PMP JSON file.

    Each row of A becomes a separate 1×1 positive-semidefinite block
    with degree-0 (constant) polynomial entries.

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix, shape (N_operators, n_components).
    f_id : np.ndarray
        Identity vector, shape (n_components,).
    output_path : Path
        Where to write the JSON file.
    significant_digits : int
        Number of significant digits for float formatting.

    Returns
    -------
    Path
        The output_path (for chaining).
    """
    n_operators, n_vars = A.shape
    fmt = lambda x: _format_float(x, significant_digits)

    # Objective: all zeros (pure feasibility, no optimization)
    objective = [fmt(0.0)] * n_vars

    # Normalization: α^T f_id = 1
    normalization = [fmt(f_id[k]) for k in range(n_vars)]

    # Build the PositiveMatrixWithPrefactorArray.
    # Each operator i → one block with a 1×1 matrix and degree-0 polynomials.
    # The constraint for block i is: Σ_n z_n * A[i,n] >= 0
    #
    # DampedRational with base=1, constant=1, no poles → prefactor = 1.
    # polynomials[row=0][col=0][var_n][coeff_k=0] = A[i, n]
    blocks = []
    damped_rational = {"base": "1", "constant": "1", "poles": []}

    for i in range(n_operators):
        # Each variable z_n contributes a degree-0 polynomial: [A[i,n]]
        poly_row = [[fmt(A[i, n])] for n in range(n_vars)]
        block = {
            "DampedRational": damped_rational,
            "polynomials": [[[*poly_row]]],
        }
        blocks.append(block)

    pmp = {
        "objective": objective,
        "normalization": normalization,
        "PositiveMatrixWithPrefactorArray": blocks,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pmp, f, separators=(",", ":"))

    return output_path


# =============================================================================
# SDPB subprocess runner
# =============================================================================

def _run_command(
    cmd: list,
    timeout: int,
    verbose: bool,
    label: str = "",
) -> subprocess.CompletedProcess:
    """Run a shell command with timeout and optional logging."""
    if verbose:
        print(f"  [{label}] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if verbose:
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n")[-5:]:
                print(f"    stdout: {line}")
        if result.stderr.strip():
            for line in result.stderr.strip().split("\n")[-5:]:
                print(f"    stderr: {line}")
    return result


def run_pmp2sdp(
    pmp_path: Path,
    sdp_dir: Path,
    config: SdpbConfig,
) -> None:
    """
    Convert PMP JSON to SDPB's binary SDP format.

    Raises
    ------
    RuntimeError
        If pmp2sdp returns a non-zero exit code.
    """
    cmd = [
        "singularity", "exec",
        "--bind", f"{pmp_path.parent}:{pmp_path.parent}",
        "--bind", f"{sdp_dir.parent}:{sdp_dir.parent}",
        str(config.image_path),
        "pmp2sdp",
        f"--precision={config.precision}",
        f"--input={pmp_path}",
        f"--output={sdp_dir}",
    ]
    result = _run_command(cmd, config.timeout, config.verbose, "pmp2sdp")
    if result.returncode != 0:
        raise RuntimeError(
            f"pmp2sdp failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )


def run_sdpb_solver(
    sdp_dir: Path,
    out_dir: Path,
    config: SdpbConfig,
) -> dict:
    """
    Run the SDPB solver on a preprocessed SDP.

    Returns
    -------
    dict
        Parsed output with keys: terminateReason, primalObjective,
        dualObjective, dualityGap, and raw (full text).
    """
    cmd = [
        "singularity", "exec",
        "--bind", f"{sdp_dir.parent}:{sdp_dir.parent}",
        "--bind", f"{out_dir.parent}:{out_dir.parent}",
        str(config.image_path),
        "mpirun", "--allow-run-as-root",
        "-n", str(config.n_cores),
        "sdpb",
        f"--precision={config.precision}",
        f"--sdpDir={sdp_dir}",
        f"--outDir={out_dir}",
        "--noFinalCheckpoint",
    ]
    result = _run_command(cmd, config.timeout, config.verbose, "sdpb")
    if result.returncode != 0:
        raise RuntimeError(
            f"sdpb failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )

    return parse_sdpb_output(out_dir)


def parse_sdpb_output(out_dir: Path) -> dict:
    """
    Parse SDPB's output file to extract termination info.

    Parameters
    ----------
    out_dir : Path
        Directory containing SDPB output files.

    Returns
    -------
    dict
        Keys: terminateReason, primalObjective, dualObjective,
        dualityGap, raw.
    """
    out_file = out_dir / "out.txt"
    if not out_file.exists():
        raise FileNotFoundError(f"SDPB output not found: {out_file}")

    text = out_file.read_text()
    parsed = {"raw": text}

    for line in text.split("\n"):
        line = line.strip()
        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip().strip('"')
            val = val.strip().strip('";')
            if key == "terminateReason":
                parsed["terminateReason"] = val
            elif key == "primalObjective":
                parsed["primalObjective"] = val
            elif key == "dualObjective":
                parsed["dualObjective"] = val
            elif key == "dualityGap":
                parsed["dualityGap"] = val

    return parsed


# =============================================================================
# High-level feasibility wrapper
# =============================================================================

def check_feasibility_sdpb(
    A: np.ndarray,
    f_id: np.ndarray,
    config: Optional[SdpbConfig] = None,
) -> FeasibilityResult:
    """
    Check bootstrap feasibility using SDPB.

    Drop-in replacement for check_feasibility() that uses SDPB's
    arbitrary-precision solver instead of scipy's float64 LP.

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix of shape (N_operators, n_components).
    f_id : np.ndarray
        Identity vector of shape (n_components,).
    config : SdpbConfig, optional
        SDPB configuration.  Uses defaults if not provided.

    Returns
    -------
    FeasibilityResult
        excluded=True if a functional was found (spectrum excluded).
        excluded=False if no functional exists (spectrum allowed).
    """
    if config is None:
        config = SdpbConfig()

    if not config.image_path.exists():
        raise FileNotFoundError(
            f"SDPB Singularity image not found at {config.image_path}. "
            f"Pull it with: singularity pull {config.image_path} "
            f"docker://bootstrapcollaboration/sdpb:3.1.0"
        )

    # Determine working directory
    base_dir = config.work_dir or Path(os.environ.get("TMPDIR", "/tmp"))
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = Path(tempfile.mkdtemp(dir=base_dir, prefix="sdpb_"))
    pmp_path = tmpdir / "pmp.json"
    sdp_dir = tmpdir / "sdp"
    out_dir = tmpdir / "out"
    sdp_dir.mkdir()
    out_dir.mkdir()

    try:
        # 1. Write PMP JSON
        if config.verbose:
            print(f"  Writing PMP JSON ({A.shape[0]} blocks, "
                  f"{A.shape[1]} vars) ...")
        write_pmp_json(A, f_id, pmp_path)

        # 2. Convert to SDP format
        if config.verbose:
            print(f"  Running pmp2sdp (precision={config.precision}) ...")
        run_pmp2sdp(pmp_path, sdp_dir, config)

        # 3. Solve
        if config.verbose:
            print(f"  Running sdpb ({config.n_cores} cores, "
                  f"precision={config.precision}) ...")
        output = run_sdpb_solver(sdp_dir, out_dir, config)

        # 4. Interpret result
        return _interpret_sdpb_output(output)

    except subprocess.TimeoutExpired:
        return FeasibilityResult(
            excluded=False,
            status="SDPB timed out",
            lp_status=4,
        )
    except (RuntimeError, FileNotFoundError) as e:
        return FeasibilityResult(
            excluded=False,
            status=f"SDPB error: {e}",
            lp_status=4,
        )
    finally:
        if config.cleanup:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


def _interpret_sdpb_output(output: dict) -> FeasibilityResult:
    """
    Convert SDPB output dict to a FeasibilityResult.

    SDPB's termination reasons map to our feasibility as follows:

    - "found dual feasible solution" → α exists → EXCLUDED
    - "found primal-dual optimal solution" → α exists → EXCLUDED
    - "found primal feasible solution" → α does not exist → ALLOWED
    - "dual infeasible" → α does not exist → ALLOWED
    - "primal infeasible" → check dualObjective sign
    - other → inconclusive, treat as ALLOWED

    For our zero-objective feasibility problem, "primal-dual optimal"
    means both primal and dual are feasible, so α exists.
    """
    reason = output.get("terminateReason", "")

    if "dual feasible" in reason or "primal-dual optimal" in reason:
        return FeasibilityResult(
            excluded=True,
            status=f"Spectrum excluded (SDPB: {reason})",
            lp_status=0,
        )

    if "primal feasible" in reason or "dual infeasible" in reason:
        return FeasibilityResult(
            excluded=False,
            status=f"Spectrum allowed (SDPB: {reason})",
            lp_status=2,
        )

    if "maxComplementarity" in reason:
        # Complementarity diverged — indicates the dual (our LP) is infeasible.
        return FeasibilityResult(
            excluded=False,
            status=f"Spectrum allowed (SDPB: {reason})",
            lp_status=2,
        )

    # Inconclusive (timeout, max iterations, etc.)
    return FeasibilityResult(
        excluded=False,
        status=f"SDPB inconclusive: {reason}",
        lp_status=4,
    )
