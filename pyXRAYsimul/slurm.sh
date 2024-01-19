#!/bin/bash -i
# 
#SBATCH --job-name=06_XAS
#SBATCH --output=06_slurmXAS.out
#SBATCH --nodelist=node5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate pyscf
echo $CONDA_PREFIX

export OMP_NUM_THREADS=36
python 06_FeSdi_xas.py > 06_FeSdi_xas.out
