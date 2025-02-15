#!/usr/bin/bash --login

# The name to show in queue lists for this job:

#SBATCH -J finetune_isamath


# Number of desired cpus:

#SBATCH --cpus-per-task=1



# Amount of RAM needed for this job:

#SBATCH --mem=2gb



# The time the job will be running:

#SBATCH --time=1:00:00



# To use GPUs you have to request them:

#SBATCH --gres=gpu:1

#SBATCH --constraint=dgx



# Set output and error files

#SBATCH --error=isamath.%J.err

#SBATCH --output=isamath.%J.out


eval "$(conda shell.bash hook)"

conda activate jgpu

hostname

time python3 stages/2_finetune.py