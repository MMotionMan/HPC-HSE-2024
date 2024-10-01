#!/bin/bash
#SBATCH --job-name=gemm_cpp_task
#SBATCH --error=gemm_cpp_task-%j.err 
#SBATCH --time=0:05:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1

module load OpenBlas/v0.3.18
cd /home/anvakirichenko/HPC-HSE-2024/hw1
make
