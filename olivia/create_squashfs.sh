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

echo "========================================"
echo "Creating SquashFS Python Environment"
echo "Started: $(date)"
echo "========================================"

# Set container path
export SIF=/cluster/projects/nn12017k/container/anemoi-base.sif

echo ""
echo "=== Step 1: Create virtual environment ==="
apptainer exec --bind $PWD $SIF bash -c "python -m venv anemoi-env --system-site-packages"

echo ""
echo "=== Step 2: Install packages (10-20 min) ==="
apptainer exec --bind $PWD $SIF bash -c "source anemoi-env/bin/activate && pip install -q -r requirements_additions.txt"

echo ""
echo "=== Step 3: Test environment ==="
apptainer exec --bind $PWD $SIF bash -c "source anemoi-env/bin/activate && python test_imports.py"

echo ""
echo "=== Step 4: Create squashfs file ==="
mksquashfs anemoi-env anemoi-env.sqsh -noappend -no-progress

echo ""
echo "=== Step 5: Remove venv directory ==="
rm -rf anemoi-env

echo ""
echo "=== Step 6: Test squashfs environment ==="
export APPTAINERENV_PREPEND_PATH=/user-software/bin
apptainer exec -B $PWD -B anemoi-env.sqsh:/user-software:image-src=/ $SIF bash -c "source /user-software/bin/activate && python test_imports.py"

echo ""
echo "========================================"
echo "✓ SUCCESS!"
echo "Created: anemoi-env.sqsh"
echo "Completed: $(date)"
echo "========================================"
