"""
Tests for the SDPB backend.

Tests the PMP JSON writer and, when SDPB is available, end-to-end
feasibility checks.

Mark SDPB-dependent tests with @pytest.mark.sdpb so they can be
skipped on systems without the Singularity image.
"""

import json
import pytest
import numpy as np
import tempfile
from pathlib import Path

from ising_bootstrap.lp.sdpb import (
    write_pmp_json,
    _format_float,
    SdpbConfig,
    check_feasibility_sdpb,
    _interpret_sdpb_output,
)
from ising_bootstrap.lp.solver import check_feasibility, FeasibilityResult


# Default SDPB image path
_DEFAULT_IMAGE = (
    Path(__file__).resolve().parents[2]
    / "tools" / "sdpb-3.1.0.sif"
)

sdpb_available = pytest.mark.skipif(
    not _DEFAULT_IMAGE.exists(),
    reason="SDPB Singularity image not found",
)


# ============================================================================
# Tests for PMP JSON writer (no SDPB required)
# ============================================================================

class TestWritePmpJson:
    """Test the PMP JSON writer produces correct structure."""

    def test_basic_structure(self):
        """JSON has objective, normalization, PositiveMatrixWithPrefactorArray."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        f_id = np.array([0.5, 0.5])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pmp.json"
            write_pmp_json(A, f_id, path)

            with open(path) as f:
                pmp = json.load(f)

        assert "objective" in pmp
        assert "normalization" in pmp
        assert "PositiveMatrixWithPrefactorArray" in pmp

    def test_dimensions_match(self):
        """Objective and normalization length equals n_vars."""
        n_vars = 5
        A = np.random.randn(10, n_vars)
        f_id = np.random.randn(n_vars)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pmp.json"
            write_pmp_json(A, f_id, path)
            with open(path) as f:
                pmp = json.load(f)

        assert len(pmp["objective"]) == n_vars
        assert len(pmp["normalization"]) == n_vars
        assert len(pmp["PositiveMatrixWithPrefactorArray"]) == 10

    def test_objective_is_zeros(self):
        """Objective should be all zeros for feasibility."""
        A = np.array([[1.0, 2.0]])
        f_id = np.array([1.0, 0.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pmp.json"
            write_pmp_json(A, f_id, path)
            with open(path) as f:
                pmp = json.load(f)

        for val in pmp["objective"]:
            assert float(val) == 0.0

    def test_normalization_matches_f_id(self):
        """Normalization should match the identity vector."""
        f_id = np.array([1.23, -4.56, 7.89])
        A = np.eye(3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pmp.json"
            write_pmp_json(A, f_id, path)
            with open(path) as f:
                pmp = json.load(f)

        for i, val in enumerate(pmp["normalization"]):
            np.testing.assert_allclose(float(val), f_id[i], rtol=1e-15)

    def test_polynomial_values_match_A(self):
        """Each block's polynomials should match the corresponding row of A."""
        A = np.array([[1.5, -2.3], [0.0, 4.2]])
        f_id = np.array([1.0, 0.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pmp.json"
            write_pmp_json(A, f_id, path)
            with open(path) as f:
                pmp = json.load(f)

        for i, block in enumerate(pmp["PositiveMatrixWithPrefactorArray"]):
            polys = block["polynomials"]
            # Structure: [1 row][1 col][n_vars entries][1 coeff each]
            assert len(polys) == 1  # 1 matrix block
            assert len(polys[0]) == 1  # 1x1 matrix
            coeffs = polys[0][0]
            assert len(coeffs) == 2  # 2 decision variables
            for n in range(2):
                np.testing.assert_allclose(
                    float(coeffs[n][0]), A[i, n], rtol=1e-15
                )

    def test_damped_rational_trivial(self):
        """DampedRational should be base=1, constant=1, no poles."""
        A = np.array([[1.0]])
        f_id = np.array([1.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pmp.json"
            write_pmp_json(A, f_id, path)
            with open(path) as f:
                pmp = json.load(f)

        dr = pmp["PositiveMatrixWithPrefactorArray"][0]["DampedRational"]
        assert dr["base"] == "1"
        assert dr["constant"] == "1"
        assert dr["poles"] == []


class TestFormatFloat:
    """Test float formatting for string precision."""

    def test_round_trip(self):
        """Float should round-trip exactly through string."""
        values = [1.0, -3.14, 1e-300, 1e300, 0.0, np.finfo(float).tiny]
        for v in values:
            s = _format_float(v)
            assert float(s) == v or (v == 0 and float(s) == 0)


# ============================================================================
# Tests for SDPB output interpretation
# ============================================================================

class TestInterpretOutput:
    """Test _interpret_sdpb_output mapping."""

    def test_dual_feasible_excluded(self):
        result = _interpret_sdpb_output(
            {"terminateReason": "found dual feasible solution"}
        )
        assert result.excluded is True
        assert result.success is True

    def test_primal_dual_optimal_excluded(self):
        result = _interpret_sdpb_output(
            {"terminateReason": "found primal-dual optimal solution"}
        )
        assert result.excluded is True
        assert result.success is True

    def test_primal_feasible_allowed(self):
        result = _interpret_sdpb_output(
            {"terminateReason": "found primal feasible solution"}
        )
        assert result.excluded is False
        assert result.success is True

    def test_dual_infeasible_allowed(self):
        result = _interpret_sdpb_output(
            {"terminateReason": "dual infeasible"}
        )
        assert result.excluded is False
        assert result.success is True

    def test_max_complementarity_inconclusive(self):
        result = _interpret_sdpb_output(
            {"terminateReason": "maxComplementarity exceeded"}
        )
        assert result.excluded is False
        assert result.success is False
        assert "inconclusive" in result.status.lower()

    def test_unknown_reason_inconclusive(self):
        result = _interpret_sdpb_output(
            {"terminateReason": "maxIterationsExceeded"}
        )
        assert result.excluded is False
        assert result.success is False
        assert "inconclusive" in result.status


# ============================================================================
# End-to-end tests (require SDPB)
# ============================================================================

class TestSdpbEndToEnd:
    """End-to-end tests using the actual SDPB solver."""

    @sdpb_available
    def test_feasible_lp_excluded(self):
        """A clearly feasible LP should return excluded=True."""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        f_id = np.array([1.0, 0.0])
        config = SdpbConfig(n_cores=1, verbose=False)

        result = check_feasibility_sdpb(A, f_id, config)
        assert result.excluded is True

    @sdpb_available
    def test_infeasible_lp_allowed(self):
        """An infeasible LP should return excluded=False."""
        # normalization: alpha_0 + alpha_1 = 1
        # constraint: -(alpha_0 + alpha_1) >= 0 => -1 >= 0 impossible
        A = np.array([[-1.0, -1.0], [1.0, 0.0]])
        f_id = np.array([1.0, 1.0])
        config = SdpbConfig(n_cores=1, verbose=False)

        result = check_feasibility_sdpb(A, f_id, config)
        assert result.excluded is False

    @sdpb_available
    def test_backend_parameter_scipy(self):
        """Backend='scipy' should use the scipy solver."""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        f_id = np.array([1.0, 0.0])

        result = check_feasibility(A, f_id, backend="scipy", scale=False)
        assert result.excluded is True

    @sdpb_available
    def test_backend_parameter_sdpb(self):
        """Backend='sdpb' should use SDPB and match scipy for simple problem."""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        f_id = np.array([1.0, 0.0])

        config = SdpbConfig(n_cores=1, verbose=False)
        result = check_feasibility(
            A, f_id, backend="sdpb", sdpb_config=config
        )
        assert result.excluded is True
