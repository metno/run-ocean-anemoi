#!/usr/bin/env python3
"""
Wrapper script that applies the graph loading patch before running anemoi-training.

Usage:
    python train_wrapper.py --config-dir=/path/to/config --config-name=main-core.yaml

Supports two modes:
1. torchrun mode: WORLD_SIZE set, script runs once per GPU with LOCAL_RANK
2. Lightning mode: No WORLD_SIZE, Lightning spawns subprocesses with ddp_spawn strategy
"""
import sys
import os
import time

# Check if we're running under torchrun or Lightning will spawn processes
local_rank = int(os.environ.get('LOCAL_RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))

print(f"[Rank {local_rank}] === TRAIN WRAPPER STARTING ===", flush=True)
print(f"[Rank {local_rank}] WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}", flush=True)

if world_size > 1:
    # Running under torchrun - stagger CUDA initialization
    stagger_delay = local_rank * 2.0  # 2 seconds between each rank
    if stagger_delay > 0:
        print(f"[Rank {local_rank}] Staggering CUDA init by {stagger_delay:.1f}s", flush=True)
        time.sleep(stagger_delay)

    # Set CUDA device before any other CUDA operations
    print(f"[Rank {local_rank}] About to import torch...", flush=True)
    import torch
    print(f"[Rank {local_rank}] torch imported, CUDA initialized={torch.cuda.is_initialized()}", flush=True)

    torch.cuda.set_device(local_rank)
    print(f"[Rank {local_rank}] Set CUDA device to {local_rank}, sees {torch.cuda.device_count()} GPUs", flush=True)
    print(f"[Rank {local_rank}] CUDA initialized after set_device={torch.cuda.is_initialized()}", flush=True)
else:
    # Running single process - Lightning will handle spawning
    print(f"[Main] Single process mode - Lightning will handle DDP", flush=True)

# Add the directory containing the patch to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Apply the patch BEFORE importing anemoi
print(f"[Rank {local_rank}] About to import anemoi_graph_patch...", flush=True)
import anemoi_graph_patch
print(f"[Rank {local_rank}] anemoi_graph_patch imported", flush=True)

# Check CUDA state before importing anemoi
if world_size > 1:
    import torch
    print(f"[Rank {local_rank}] CUDA initialized before anemoi import={torch.cuda.is_initialized()}", flush=True)

# Now run the original anemoi training
print(f"[Rank {local_rank}] About to import anemoi.training.train.train...", flush=True)
from anemoi.training.train.train import main
print(f"[Rank {local_rank}] anemoi.training imported", flush=True)

if world_size > 1:
    import torch
    print(f"[Rank {local_rank}] CUDA initialized after anemoi import={torch.cuda.is_initialized()}", flush=True)

if __name__ == "__main__":
    print(f"[Rank {local_rank}] About to call main()...", flush=True)
    main()
