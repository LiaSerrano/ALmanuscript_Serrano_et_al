#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-12:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

cd ..
cd ..
cd run/chemprop

python predict.py --test_path "/demo_data/remaining_unscreened_molecules_filt.csv" \
                  --smiles_columns 'SMILES'\
                  --checkpoint_dir "demo_data/checkpoints/" \
                  --preds_path "demo_data/model_predictions/ecoli_pcontrol.csv"\
                  --features_path demo_ecoli_features.npz \
                  --no_features_scaling \
                  --uncertainty_method 'dirichlet'
