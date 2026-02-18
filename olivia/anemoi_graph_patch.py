#!/usr/bin/env python3
"""
Patch anemoi-training to load graph files safely on multi-GPU setups.

The issue: anemoi-training v0.6.7's train.py:155 uses:
    torch.load(graph_filename, map_location=get_distributed_device())

This causes CUDA race conditions on GH200 architecture when multiple
torchrun workers try to load the same file to different GPUs simultaneously.

The fix: Monkey-patch the graph loading to first load to CPU, then move to GPU.

Usage:
    Import this module BEFORE importing anemoi.training.train.train:

        import anemoi_graph_patch
        from anemoi.training.train import train
        train.main()
"""
import functools
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_original_torch_load = None

def _patched_torch_load(f, *args, map_location=None, **kwargs):
    """Patched torch.load that loads to CPU first, then moves to target device."""
    import torch
    import time

    # Get the intended target device
    target_device = map_location

    # Convert torch.device to string for checking
    if hasattr(target_device, 'type'):
        device_str = str(target_device)
    else:
        device_str = str(target_device) if target_device else 'cpu'

    rank = int(os.environ.get('LOCAL_RANK', 0))

    # Check if this is a CUDA device
    if 'cuda' in device_str:
        # Load to CPU first to avoid all ranks loading to GPU simultaneously
        logger.info(f"[Rank {rank}] Intercepted torch.load for {device_str}, loading to CPU first")
        data = _original_torch_load(f, *args, map_location='cpu', **kwargs)

        # Stagger GPU move by rank to avoid race conditions on GH200
        # This is separate from the initial CUDA stagger in train_wrapper.py
        stagger_delay = rank * 1.0  # 1 second between each rank
        if stagger_delay > 0:
            logger.info(f"[Rank {rank}] Staggering GPU move by {stagger_delay}s")
            time.sleep(stagger_delay)

        # Move to target device - use current device, not the one from map_location
        # This avoids calling set_device again which can cause issues on GH200
        current_device = f'cuda:{torch.cuda.current_device()}'
        logger.info(f"[Rank {rank}] Moving data to {current_device} (requested was {device_str})")

        # For HeteroData or other complex objects, we need to move all tensors
        if hasattr(data, 'to'):
            return data.to(current_device)
        elif isinstance(data, dict):
            # Recursively move tensors in dict
            def move_to_device(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.to(current_device)
                elif isinstance(obj, dict):
                    return {k: move_to_device(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [move_to_device(v) for v in obj]
                return obj
            return move_to_device(data)
        else:
            return data
    else:
        # For non-CUDA targets, use original behavior
        return _original_torch_load(f, *args, map_location=map_location, **kwargs)

def apply_patch():
    """Apply the monkey patch to torch.load."""
    global _original_torch_load
    import torch

    if _original_torch_load is None:
        _original_torch_load = torch.load
        torch.load = _patched_torch_load
        logger.info("Applied anemoi graph loading patch (load to CPU first)")
    else:
        logger.warning("Patch already applied")

# Auto-apply when imported
apply_patch()
