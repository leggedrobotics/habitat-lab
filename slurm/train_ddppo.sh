#!/bin/bash

#SBATCH --ntasks 4
#SBATCH --cpus-per-task 16
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=50G
#SBATCH --gpus=2
#SBATCH --gres=gpumem:23G
#SBATCH --job-name=habitat-pointnav-dinoRGB
#SBATCH --output=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.out
#SBATCH --error=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.err


source ~/.bashrc
conda activate habitat

echo "Copying the codebase"

echo "Preparing and copying dataset"

echo "Current directory is: "

cd 

echo "$pwd"

echo "Starting Training"

python -u -m torch.distributed.launch --use_env --nproc_per_node 1 ${TMPDIR}/habitat-lab/habitat-baselines/habitat_baselines/run.py --config-name=pointnav/cluster_ddppo_pointnav_dino
