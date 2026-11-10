#!/bin/bash
#SBATCH --account=nn12017k
#SBATCH --partition=accel
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=0
#SBATCH --mem=64G
#SBATCH --job-name=create_squashfs
#SBATCH --output=outputs/create_squashfs_%j.out

set -euo pipefail

echo "========================================"
echo "Creating SquashFS Python Environment"
echo "Started: $(date)"
echo "========================================"

export SIF=/cluster/projects/nn12017k/container/pytorch_nvidia_25.06_arm64.sif

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: Container image not found: $SIF" >&2
    exit 1
fi

if [[ -e anemoi-env ]]; then
    echo "ERROR: anemoi-env already exists; refusing to reuse or overwrite it." >&2
    exit 1
fi

echo ""
echo "=== Step 1: Create virtual environment ==="
apptainer exec --bind "$PWD" "$SIF" python -m venv anemoi-env --system-site-packages

echo ""
echo "=== Step 2: Install packages (10-20 min) ==="
apptainer exec --bind "$PWD" "$SIF" \
    bash -c 'source anemoi-env/bin/activate && unset PIP_CONSTRAINT && python -m pip install -r requirements_additions.txt'

echo ""
echo "=== Step 3: Test environment ==="
apptainer exec --bind "$PWD" "$SIF" \
    bash -c 'source anemoi-env/bin/activate && python test_imports.py'

echo ""
echo "=== Step 4: Create squashfs file ==="
mksquashfs anemoi-env anemoi-env.sqsh -noappend -no-progress -no-xattrs
test -s anemoi-env.sqsh

echo ""
echo "=== Step 5: Test squashfs environment ==="
apptainer exec -B "$PWD" \
    -B "$PWD/anemoi-env.sqsh:/user-software:image-src=/" "$SIF" \
    /user-software/bin/python test_imports.py

echo ""
echo "=== Step 6: Remove virtual environment directory ==="
rm -rf anemoi-env

echo ""
echo "========================================"
echo "✓ SUCCESS!"
echo "Created: anemoi-env.sqsh"
echo "Completed: $(date)"
echo "========================================"
