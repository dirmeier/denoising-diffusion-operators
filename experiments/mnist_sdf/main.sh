#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=sd28

module load daint-gpu
conda activate cifma-uqma-dev

srun python main.py --mode=train --model=unet --epochs=1000 --usewand
