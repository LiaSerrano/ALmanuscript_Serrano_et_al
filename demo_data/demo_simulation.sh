#!/bin/bash
#SBATCH -c 2
#SBATCH -t 5-00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err


 python "/n/data1/hms/dbmi/farhat/LRS/Lyme_Antibiotics/simulation/updated_activeLearn.py"  --acquisition_strategy "batch_selection" \
                                                                                           --task tb \
                                                                                           --initial_train_plates /n/data1/hms/dbmi/farhat/LRS/Lyme_Antibiotics/simulation/10212025_replicates/starting_plate.csv
