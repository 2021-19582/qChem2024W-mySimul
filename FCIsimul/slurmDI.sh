#!/bin/bash -i
# 
#SBATCH --job-name=00003_dI
#SBATCH --output=00003_slurmDI.out
#SBATCH --nodelist=node4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate pyscf
echo $CONDA_PREFIX

export OMP_NUM_THREADS=36
python dumpInt.py > 00003_dumpInt.out
