#!/bin/bash
# Verify all .slurm files use sapphire partition

PROJECT_DIR="/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap"
cd "$PROJECT_DIR/jobs"

echo "Checking partition configuration in all .slurm files:"
echo ""

ERRORS=0
for file in *.slurm; do
    line4=$(sed -n '4p' "$file")
    if [[ "$line4" == "#SBATCH --partition=sapphire" ]]; then
        echo "✓ $file: sapphire"
    elif [[ "$line4" == "#SBATCH --partition=shared" ]]; then
        echo "✗ $file: STILL SHARED (not migrated)"
        ERRORS=$((ERRORS + 1))
    else
        echo "? $file: unexpected line 4: $line4"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "✓ All 18 .slurm files use sapphire partition"
    exit 0
else
    echo "✗ Found $ERRORS files with incorrect partition"
    exit 1
fi
