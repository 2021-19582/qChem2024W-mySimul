#!/bin/bash -i
# 
#SBATCH --job-name=00008_gM
#SBATCH --output=00008_slurmGM.out
#SBATCH --nodelist=node2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate pyscf
echo $CONDA_PREFIX

module purge
module add pyscf
module list

export OMP_NUM_THREADS=36
python genMolden.py > 00008_genMolden.out
