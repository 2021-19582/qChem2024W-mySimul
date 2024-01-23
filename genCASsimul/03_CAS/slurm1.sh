#!/bin/bash -i
# 
#SBATCH --job-name=02_loc
#SBATCH --output=slurm1.out
#SBATCH --nodelist=node4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate pyscf
echo $CONDA_PREFIX

mysimulstr='02'

export OMP_NUM_THREADS=36
python nlocal.py > nlocal.out
