#!/bin/bash
# One-time environment setup for FASRC Cannon cluster.
# Run this on an interactive node (salloc), NOT on a login node.
#
# Usage:
#   salloc -p test --account=iaifi_lab -c 4 -t 00:30:00 --mem=8G
#   bash jobs/setup_env.sh

set -e

echo "=== Ising Bootstrap: FASRC Environment Setup ==="

# --- Conda initialization ---
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    source "$HOME/.conda/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda installation" >&2
    echo "Install miniforge3 first: https://github.com/conda-forge/miniforge" >&2
    exit 1
fi

echo "Using conda from: $(conda info --base)"

# --- Create environment ---
ENV_NAME="ising_bootstrap"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists. Updating..."
    conda activate ${ENV_NAME}
    pip install -e .
else
    echo "Creating conda environment '${ENV_NAME}'..."
    mamba create -n ${ENV_NAME} -c conda-forge \
        python=3.11 numpy scipy mpmath matplotlib -y
    conda activate ${ENV_NAME}
    echo "Installing ising_bootstrap package..."
    pip install -e .
fi

# --- Verify ---
echo ""
echo "=== Verification ==="
python -c "
from ising_bootstrap.config import N_MAX, FULL_DISCRETIZATION
print(f'n_max = {N_MAX}')
print(f'Tables: {[t.name for t in FULL_DISCRETIZATION]}')
print(f'Python: {__import__(\"sys\").version}')
print('OK: ising_bootstrap installed successfully')
"

# --- Create directories ---
mkdir -p data/cached_blocks logs

echo ""
echo "=== Setup Complete ==="
echo "To activate: conda activate ${ENV_NAME}"
echo "Quick test:  pytest tests/ -x -q --timeout=60"
