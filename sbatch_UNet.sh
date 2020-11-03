#!/bin/bash -l
#SBATCH --job-name=unet3d
#SBATCH --time=240:00:00
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=40G
#SBATCH --gres=gpu:v100:4
#SBATCH --output=/home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/code/UNet.out
#SBATCH --error=/home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/code/UNet.err

pwd; hostname; date
echo jobid=${SLURM_JOB_ID}; echo nodelist=${SLURM_JOB_NODELIST}

source /home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/code/env/bin/activate
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
module use /home/exacloud/software/modules
module load cudnn/7.6-10.1
module load cuda/10.1.243
which nvcc
nvcc --version
python /home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/code/UNet.py

echo COMPLETE

