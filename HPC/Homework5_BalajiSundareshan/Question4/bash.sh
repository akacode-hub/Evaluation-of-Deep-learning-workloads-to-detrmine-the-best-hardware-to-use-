#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=your_job_name
#SBATCH --mem=1G
#SBATCH --gres=gpu:1
#SBATCH --output=Q4_out1
#SBATCH --partition=gpu
#./Q4 512000
./pi