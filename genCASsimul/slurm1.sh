#!/bin/bash -i
# 
#SBATCH --job-name=01_loc
#SBATCH --output=01_slurm1.out
#SBATCH --nodelist=node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate pyscf
echo $CONDA_PREFIX

mysimulstr='01'

export OMP_NUM_THREADS=36
python ./$mysimulstr'_CAS'/$mysimulstr'_nlocal.py' > ./$mysimulstr'_CAS'/$mysimulstr'_nlocal.out'
