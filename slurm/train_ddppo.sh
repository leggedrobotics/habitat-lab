#!/bin/bash

#SBATCH --ntasks 4
#SBATCH --cpus-per-task 16
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=50G
#SBATCH --gpus=4
#SBATCH --gres=gpumem:23G
#SBATCH --job-name=habitat-pointnav-dinoRGB
#SBATCH --output=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.out
#SBATCH --error=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.err


source ~/.bashrc
# export PATH=/cluster/home/patelm/miniconda3/bin:$PATH
conda activate habitat
# unset PYTHONHOME
# unset PYTHONPATH
module load eth_proxy

echo "Copying the codebase"
cp -r /cluster/home/patelm/ws/rsl/habitat-lab $TMPDIR

cd $TMPDIR/habitat-lab

echo "Preparing and copying dataset"
cp -r /cluster/work/rsl/patelm/habitat_data/data $TMPDIR/habitat-lab

echo "Current directory is: "

echo "$PWD"

echo "Starting Training"

python -u -m torch.distributed.launch --use_env --nproc_per_node 4 habitat-baselines/habitat_baselines/run.py --config-name=pointnav/cluster_ddppo_pointnav_dino
