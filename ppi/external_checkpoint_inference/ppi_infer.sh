#!/bin/bash
#SBATCH --job-name="Infer"
#SBATCH --output=output/Infer_%j.log
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --partition=gpuB-research
#SBATCH --time=00:30:00
#SBATCH --account=havbris
#SBATCH --mem=80G
#SBATCH --ntasks-per-node=1

#source $(pwd -P)/.venv/bin/activate
source /lustre/storeB/project/fou/hi/foccus/python-envs/inference-env-jun26/bin/activate

CONFIG_DIR=$(pwd -P)/
CONFIG_NAME=$CONFIG_DIR/infer.yaml

export HYDRA_FULL_ERROR=1

ulimit -v unlimited
anemoi-inference run $CONFIG_NAME

python ../postpro-inference.py --yaml $CONFIG_NAME #--oslo 
