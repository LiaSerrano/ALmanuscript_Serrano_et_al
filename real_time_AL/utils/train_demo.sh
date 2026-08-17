#!/bin/bash
#SBATCH -c 2
#SBATCH -t 2-00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err


cd ./run/chemprop


#E.coli %ctrl Model
python train.py --data_path "demo_data/ecoli_filt.csv" \
                --smiles_columns 'SMILES'\
                --target_columns 'Y'\
                --dataset_type 'classification' \
                --save_dir "demo_data/checkpoints" \
                --split_type 'scaffold_balanced'\
                --num_folds '5' \
                --metric 'auc' \
                --extra_metrics 'binary_cross_entropy' 'prc-auc'\
                --loss_function 'dirichlet' \
                --evidential_regularization 0.2\
                --class_balance \
                --epochs 200 \
                --features_path /demo_data/ecoli_filt.npz \
                --no_features_scaling \


