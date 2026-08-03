#!/bin/bash
#SBATCH --job-name=tess_forecast
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=tess_%j.out
# #SBATCH -A <your-account>   # uncomment and set if sbalance shows more than one account

module load pytorch/2.0.1/gcc/11.3.0/openmpi/4.1.5/cuda/12.3.0/zen2

TESS_RUN=1 python real_data_msml612_demo.py
