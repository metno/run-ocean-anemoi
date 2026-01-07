#!/bin/bash
#SBATCH --job-name="EnvSetup"
#SBATCH --output=output/make_env.log
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --partition=gpuB-research
#SBATCH --time=00:30:00
#SBATCH --account=havbris
#SBATCH --mem=200g
#SBATCH --ntasks-per-node=1

CHECKPOINT=<checkpoint> #insert checkpoint here

source  /modules/rhel9/x86_64/mamba-mf3/etc/profile.d/ppimam.sh
mamba activate /lustre/storeB/project/fou/hi/foccus/python-envs/h200-python-3.12.9 # I hope everyone has access to this. Need to it get correct python version. Let me know if there are problems, and I will think more. 

python3 -m venv $(pwd -P)/.venv

source $(pwd -P)/.venv/bin/activate

pip install anemoi-inference==0.8.0 # need this to run anemoi-inference inspect --requirements

anemoi-inference inspect $CHECKPOINT --requirements > requirements.txt

#pip install -r requirements.txt

cat requirements.txt | xargs -n 1 pip install # run this to ignore on errors, e.g. amdsmi likes failing as it's some amd stuff, but isn't necessary for inference

pip install --force-reinstall anemoi-inference==0.7.3 # For some reason the actual inference doesn't work with 0.8.0, so downgrading. Might find fix later. 

