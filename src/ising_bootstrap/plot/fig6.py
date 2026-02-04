"""
Plot reproduction of Figure 6 from arXiv:1203.6064.

Plots the upper bound on Delta_epsilon' as a function of Delta_sigma,
showing the allowed region and the Ising model point.

Usage:
    python -m ising_bootstrap.plot.fig6 \
        --data data/epsprime_bound.csv \
        --output figures/fig6_reproduction.png
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import (
    ISING_DELTA_SIGMA,
    ISING_DELTA_EPSILON,
    ISING_DELTA_EPSILON_PRIME,
    FIGURES_DIR,
    DATA_DIR,
)
from ..scans.stage_b import load_stage_b_results


def plot_fig6(
    data: List[Tuple[float, float, float]],
    output: Optional[Path] = None,
    dpi: int = 300,
    show: bool = False,
) -> plt.Figure:
    """
    Create Figure 6 reproduction: Delta_epsilon' upper bound vs Delta_sigma.

    Parameters
    ----------
    data : list of (delta_sigma, delta_eps, delta_eps_prime_max)
        Stage B scan results.
    output : Path, optional
        Save figure to this path. Supports .png and .pdf.
    dpi : int
        Resolution for PNG output.
    show : bool
        Display plot interactively.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data_sorted = sorted(data, key=lambda t: t[0])
    sigma = np.array([d[0] for d in data_sorted])
    eps_prime_max = np.array([d[2] for d in data_sorted])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Bound curve
    ax.plot(sigma, eps_prime_max, "b-", linewidth=1.5)

    # Allowed region
    y_min = 2.0
    ax.fill_between(sigma, y_min, eps_prime_max, alpha=0.3, color="blue")

    # Ising vertical line
    ax.axvline(x=ISING_DELTA_SIGMA, color="red", linewidth=2, label="Ising")

    # Labels
    ax.set_xlabel(r"$\Delta_\sigma$", fontsize=14)
    ax.set_ylabel(r"$\Delta_{\epsilon'}$", fontsize=14)
    ax.set_xlim(0.50, 0.60)
    ax.set_ylim(2.0, 4.5)
    ax.tick_params(labelsize=12)

    fig.tight_layout()

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output}")

        # Also save PDF if primary output is PNG
        if output.suffix.lower() == ".png":
            pdf_path = output.with_suffix(".pdf")
            fig.savefig(pdf_path, bbox_inches="tight")
            print(f"Saved: {pdf_path}")

    if show:
        plt.show()

    return fig


def print_sanity_check(
    data: List[Tuple[float, float, float]],
) -> None:
    """
    Print sanity check comparing results to known Ising values.

    Finds the data point nearest to the Ising Delta_sigma and reports
    observed vs expected values, plus qualitative feature checks.
    """
    if not data:
        print("No data to check.")
        return

    data_sorted = sorted(data, key=lambda t: t[0])
    sigma = np.array([d[0] for d in data_sorted])
    eps = np.array([d[1] for d in data_sorted])
    eps_prime = np.array([d[2] for d in data_sorted])

    # Find nearest point to Ising Delta_sigma
    idx = int(np.argmin(np.abs(sigma - ISING_DELTA_SIGMA)))
    nearest_sigma = sigma[idx]
    nearest_eps = eps[idx]
    nearest_eps_prime = eps_prime[idx]

    print("=== Sanity Check ===")
    print(f"Near Ising point (\u0394\u03c3 \u2248 {ISING_DELTA_SIGMA}):")
    print(f"  Nearest grid point: \u0394\u03c3 = {nearest_sigma:.4f}")
    print(f"  \u0394\u03b5_max  = {nearest_eps:.2f} "
          f"(expected ~{ISING_DELTA_EPSILON})")
    print(f"  \u0394\u03b5'_max = {nearest_eps_prime:.2f} "
          f"(expected ~{ISING_DELTA_EPSILON_PRIME})")
    print()

    # Qualitative feature checks
    print("Qualitative features:")

    # Check 1: Spike present below Ising Delta_sigma
    below_ising = eps_prime[sigma < ISING_DELTA_SIGMA]
    if len(below_ising) >= 3:
        has_spike = np.min(below_ising) < np.mean(below_ising) - 0.3
        mark = "\u2713" if has_spike else "\u2717"
    else:
        mark = "?"
    print(f"  [{mark}] Spike present below Ising \u0394\u03c3")

    # Check 2: Bound increases for Delta_sigma > 0.52
    above_052 = eps_prime[sigma > 0.52]
    if len(above_052) >= 2:
        increases = above_052[-1] > above_052[0]
        mark = "\u2713" if increases else "\u2717"
    else:
        mark = "?"
    print(f"  [{mark}] Bound increases for \u0394\u03c3 > 0.52")


def main():
    """Command-line entry point for Figure 6 plotting."""
    parser = argparse.ArgumentParser(
        description="Plot Figure 6: upper bound on \u0394\u03b5' vs \u0394\u03c3"
    )
    parser.add_argument(
        "--data", type=str,
        default=str(DATA_DIR / "epsprime_bound.csv"),
        help="Path to Stage B CSV file",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(FIGURES_DIR / "fig6_reproduction.png"),
        help="Output file path (.png or .pdf)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Resolution for PNG output (default: 300)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display plot interactively",
    )
    parser.add_argument(
        "--no-sanity-check", action="store_true",
        help="Skip sanity check output",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    data = load_stage_b_results(data_path)
    print(f"Loaded {len(data)} data points from {data_path}")

    if not args.no_sanity_check:
        print()
        print_sanity_check(data)
        print()

    plot_fig6(
        data,
        output=Path(args.output),
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
