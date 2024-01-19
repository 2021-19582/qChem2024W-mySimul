#!/bin/bash -i
# 
#SBATCH --job-name=03_XAS
#SBATCH --output=03_slurmXAS.out
#SBATCH --nodelist=node5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate pyscf
echo $CONDA_PREFIX

export OMP_NUM_THREADS=36
python 03_FeSdi_xas.py > 03_FeSdi_xas.out
