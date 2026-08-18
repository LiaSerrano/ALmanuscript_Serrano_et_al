#!/bin/bash
#SBATCH -c 2
#SBATCH -t 5-00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err


module load conda/miniforge3/24.11.3-0
module load gcc/14.2.0
module load python/3.13.1
module load cuda/12.8
source ~/chemprop_env/bin/activate


python "Simulations.py" --acquisition_strategy "batch_selection" \
                        --task ecoli \
                        --initial_train_plates /n/data1/hms/dbmi/farhat/LRS/ALmanuscript_Serrano_et_al/demo_data/starting_plate.csv \
                        --data_path /n/data1/hms/dbmi/farhat/LRS/ALmanuscript_Serrano_et_al/demo_data/demo_train.csv \
                        --out_dir /n/data1/hms/dbmi/farhat/LRS/ALmanuscript_Serrano_et_al/demo_data/simulation_output
