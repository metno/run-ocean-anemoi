#!/bin/bash
#SBATCH --account=nn12017k
#SBATCH --job-name=<experiment_name>
#SBATCH --partition=accel
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err

# 8 Node 32 GPU
echo "========================================"
echo "8 Node 32 GPU Training"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "========================================"

SIF=/cluster/projects/nn12017k/container/anemoi-base.sif
#SIF=/cluster/projects/nn12017k/container/anemoi-complete-arm64.sif
CONFIG_NAME=main.yaml
SQSH=$PWD/anemoi-env.sqsh


# Host library paths
HOST_LIBFABRIC_LIB=/opt/cray/libfabric/1.22.0/lib64
HOST_LIBFABRIC_INC=/opt/cray/libfabric/1.22.0/include
HOST_NCCL=/cluster/work/support/nccl
HOST_NVIDIA_HPC=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/compilers/lib
HOST_CXI=/usr/lib64

# Get head node IP
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes[0]}
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

echo "Head node: $head_node"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "========================================"

# Export environment for container
export APPTAINERENV_MASTER_ADDR=$MASTER_ADDR
export APPTAINERENV_MASTER_PORT=$MASTER_PORT
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export APPTAINERENV_CUDA_DEVICE_ORDER=PCI_BUS_ID
export APPTAINERENV_HYDRA_FULL_ERROR=1
export APPTAINERENV_PREPEND_PATH=/user-software/bin


# NCCL settings
#export APPTAINERENV_NCCL_DEBUG=INFO
export APPTAINERENV_NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export APPTAINERENV_NCCL_NET_GDR_LEVEL=PHB
export APPTAINERENV_NCCL_P2P_DISABLE=0
export APPTAINERENV_NCCL_ASYNC_ERROR_HANDLING=1
export APPTAINERENV_NCCL_BUFFSIZE=67108864
export APPTAINERENV_NCCL_NCHANNELS_PER_NET_PEER=4

# Anemoi seed (if under 1000 multiplies by 1000)
# If none uses slurm job ID
export ANEMOI_BASE_SEED=42
export APPTAINERENV_ANEMOI_BASE_SEED=42

# Start GPU utilization monitoring in the background
#GPU_LOG_FILE="gpu_utilization.log"
#echo "Starting GPU utilization monitoring..."
#nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &

echo "Starting 8-node 32-GPU training..."

srun apptainer exec --nv \
    --bind $HOST_LIBFABRIC_LIB:/opt/libfabric/lib \
    --bind $HOST_LIBFABRIC_INC:/opt/libfabric/include \
    --bind $HOST_NCCL:/opt/nccl \
    --bind $HOST_CXI:/usr/lib64 \
    --bind $HOST_NVIDIA_HPC:/opt/nvidia/hpc_sdk/lib \
    --bind /cluster/work/projects/nn12017k/ \
    --bind /cluster/projects/nn12017k/ \
    --bind $PWD \
    --bind ${SQSH}:/user-software:image-src=/ $SIF \
    bash -c "
        # Set up library paths
        export LD_LIBRARY_PATH=/opt/libfabric/lib:/opt/nccl/lib:/opt/nvidia/hpc_sdk/lib:/usr/lib64:\$LD_LIBRARY_PATH
        export CPATH=/opt/libfabric/include:\$CPATH

        # Symbolic link for CUDA compatibility
        ln -sf /usr/local/cuda/compat/lib/libcuda.so.1 /usr/local/cuda/compat/lib/libcuda.so 2>/dev/null || true

        # Set up distributed training environment
        export WORLD_SIZE=\$SLURM_NPROCS
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID

        echo \"[Task \$RANK/\$WORLD_SIZE on \$(hostname)] LOCAL_RANK=\$LOCAL_RANK, CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\"

        # Run training via wrapper with limited steps
        python $PWD/train_wrapper.py --config-dir=$PWD --config-name=$CONFIG_NAME
    "
