# Overview
`run-amemoi` is a collection of utility scripts and packages to use anemoi-training (for ocean).


# Running Inference on PPI

1) First test access to the GPUs:
`srun -p gpuB-prod --account havbris  --gres=gpu:nvidia_h200_nvl:1 --mem=1G --time=00:05:00 --pty bash`

2) Clone the run scripts (if not already done):
`git clone git@github.com:metno/run-ocean-anemoi.git`

3) Provide the checkpoint to setup.sh and run the script with 
```
sbatch setup.sh
```
This will take some time. The script isn't very efficient due to installing and reinstalling a bunch of packages. Might make it better later. 

4) Add the checkpoint to `infer.yaml`. 
May change graph and datasets.
Specify the date and forecast duration.

6) Run 
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


