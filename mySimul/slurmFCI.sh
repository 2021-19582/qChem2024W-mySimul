#!/bin/bash -i
# 
#SBATCH --job-name=00009_FCIDUMP
#SBATCH --output=00009_slurmFCI.out
#SBATCH --nodelist=node7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate pyscf
echo $CONDA_PREFIX

export OMP_NUM_THREADS=36
python 00009_FeSdi_OX_20o_30e > 00009_FeSdi_OX_20o_30e.out

