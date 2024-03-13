#!/bin/bash -l

#SBATCH -o ./slurm/%j.out
#SBATCH -e ./slurm/%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=sd28

module load daint-gpu
conda activate cifma-uqma-dev

srun python main.py --mode=train --model=uno --epochs=500 --usewand
