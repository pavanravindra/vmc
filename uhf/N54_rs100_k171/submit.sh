#!/bin/bash
#
#SBATCH --account=ccce         # Replace ACCOUNT with your group account name
#SBATCH --job-name=UHF         # The job name
#SBATCH -N 1                   # The number of nodes to request
#SBATCH -c 1                   # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --time=0-11:59         # The time the job will take to run in D-HH:MM
 
module load anaconda/3-2023.09
module load cuda12.0/toolkit/12.0.1
module load cudnn8.6-cuda11.8/8.6.0.163

source ~/.bashrc

conda activate nqs

~/.conda/envs/nqs/bin/python -u run_uhf.py
