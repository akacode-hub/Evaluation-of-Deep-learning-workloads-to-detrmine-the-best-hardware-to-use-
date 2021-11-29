#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --job-name=your_job_name
#SBATCH --mem=1G
#SBATCH --gres=gpu:1
#SBATCH --output=matmul_out
#SBATCH --partition=gpu
./matmul