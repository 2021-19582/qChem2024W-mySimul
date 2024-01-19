#!/bin/bash -i
# 
#SBATCH --job-name=00009_b2
#SBATCH --output=00009_slurmDMRG.out
#SBATCH --nodelist=node7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate pyscf
echo $CONDA_PREFIX

export OMP_NUM_THREADS=36
block2main DMRG.conf > 00009_DMRG.out
