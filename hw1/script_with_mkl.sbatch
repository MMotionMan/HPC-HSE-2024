#!/bin/bash

$SBATCH --job-name=gemm_cpp_task
#SBATCH --error=gemm_cpp_task-%j.err 
#SBATCH --time=0:05:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1

module load INTEL/oneAPI_2022
module load mkl/latest
cd /home/anvakirichenko/HPC-HSE-2024/hw1
make build_mkl
make test_mkl
~      
