#!/bin/bash -i
# 
#SBATCH --job-name=n2_gitPush
#SBATCH --output=slurmGit.out
#SBATCH --nodelist=node2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=36
git add .
git commit -m "update node2"
git push
