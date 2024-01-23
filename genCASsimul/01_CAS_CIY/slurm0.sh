#!/bin/bash -i
# 
#SBATCH --job-name=01_scf
#SBATCH --output=slurm0.out
#SBATCH --nodelist=node4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate pyscf
echo $CONDA_PREFIX

export OMP_NUM_THREADS=36
python hs.py
