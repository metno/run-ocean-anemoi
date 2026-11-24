#!/usr/bin/env python3
"""Test that anemoi packages are installed correctly."""

print("Testing anemoi installation...")
print()

# Test anemoi packages
import anemoi.utils
import anemoi.training
import anemoi.models
import anemoi.graphs
print(f"✓ anemoi.utils      v{anemoi.utils.__version__}")
print(f"✓ anemoi.training   v{anemoi.training.__version__}")
print(f"✓ anemoi.models     v{anemoi.models.__version__}")
print(f"✓ anemoi.graphs     v{anemoi.graphs.__version__}")

# Test PyTorch and GPUs
import numpy as np
import torch
print()
print(f"✓ PyTorch v{torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
torch.from_numpy(np.zeros(1))
if torch.cuda.is_available():
    print(f"✓ GPUs detected: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}")

print()
print("✓ All tests passed!")
