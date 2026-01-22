# Overview
`run-ocean-anemoi` is a collection of utility scripts and packages to use anemoi-training (for ocean).


# Running Inference on PPI

The submit / que system for the PPI GPUs are different from the rest of PPI, it uses "Slurm". The only difference is that syntax inside the submit scripts are different, and that commands to interact with the jobs are different (but similar). 

If you have never used the GPU, test your access by logging on to the GPU (interactive session):
`srun -p gpuB-prod --account havbris  --gres=gpu:nvidia_h200_nvl:1 --mem=1G --time=00:05:00 --pty bash`
then you should see that your prompt changes from the login node to a compute node. Ask for more than 5mins if you want to stay there and test things. 

## Inference environment (`anemoi-inference`)
To be able to produce results you need to run inference on a specific checkpoint. This checkpoint has been outputted during a training, that used a specific version of anemoi. The anemoi-version used in training MUST be compatible with the anemoi versions used for inference. We make sure that this is the case by using the checkpoint to specify the anemoi version in `setup.sh`. This script creates a virtual environment `.venv` in your current directory. 

TLDR; you may re-use an environment for several checkpoints if they have been trained with same/similar anemoi versions. 

## Work directory
You may run inference on many checkpoints from many different experiments from the same directory, and specify the output directories for the nc-files as you wish. 

## How to run inference: 

1) Select where you want to run inference from.

2) Clone the run scripts (if not already done):
`git clone git@github.com:metno/run-ocean-anemoi.git`
and
`cd run-ocean-anemoi/ppi/external_checkpoint_inference`

4) If an env is not already available, create an env to use. Provide/change the checkpoint in `setup.sh` and run the script with 
```
sbatch setup.sh
```
where `sbatch` is the command to submit a script to the Slurm que. 
This will take some time. The script isn't very efficient due to installing and reinstalling a bunch of packages. Might make it better later. 

4) Add the checkpoint to `infer.yaml`. (It is also possible to change graph and datasets, but not neccecary). Specify the date and lead time/forecast duration.
No need to specify the path to the output file since the script `postpro-inference.py` will handle that automatically, renaming the file from `temp.nc` to:
`<date>_<lead_time>_<run_id>_<epoch>_<step>.nc`

5) Run 
```
sbatch ppi_infer.sh
```

### [UPDATE THIS] Creating a new enviroment to use on PPI GPU:
Then get mamba/conda:
`source  /modules/rhel9/x86_64/mamba-mf3/etc/profile.d/ppimam.sh`




# Anemoi-training on LUMI
Use of virtual python environments is strongly dicouraged on LUMI, with a container based approach being the prefered solution. Therefore we use a singularity container which contains the entire software environment except for the anemoi repositories themselves (training, graphs, models, datasets, utils). These are installed in a lightweight virtual environment that we load on top of the container, which enables us to edit these packages without rebuilding the container. 
- The virtual environment is set up by executing `bash make_env.sh` in /lumi.
This will download the anemoi-packages and install them in a .venv folder inside /lumi.

You can now train a model through the following steps:
- Setup the desider model config file and make sure it is placed in /lumi. **This file should not be named `config.yaml` or any other config name allready in anemoi-training.**
- Specify the config file name in `lumi_jobscript.sh` along with preferred sbatch settings for the job.
- Submit the job with `sbatch lumi_jobscript.sh`


