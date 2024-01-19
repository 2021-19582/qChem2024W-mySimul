#!/bin/bash -i
# 
#SBATCH --job-name=07_XAS
#SBATCH --output=07_slurmXAS.out
#SBATCH --nodelist=node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate pyscf
echo $CONDA_PREFIX

export OMP_NUM_THREADS=36
python 07_FeScu_xas.py > 07_FeScu_xas.out
mv lunoloc.molden FeScu_lunoloc.molden
mv luno.molden FeScu_luno.molden
