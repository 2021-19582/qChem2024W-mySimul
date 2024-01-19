#!/bin/bash -i
# 
#SBATCH --job-name=08_XAS
#SBATCH --output=08_slurmXAS.out
#SBATCH --nodelist=node5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate pyscf
echo $CONDA_PREFIX

export OMP_NUM_THREADS=36
python 08_FeSdi_xas.py > 08_FeSdi_xas.out
mv lunoloc.molden FeSdi_lunoloc.molden
mv luno.molden FeSdi_luno.molden
