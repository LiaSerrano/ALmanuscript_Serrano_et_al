#!/bin/bash
#SBATCH -c 2
#SBATCH -t 2-00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

set -e

# train_demo.sh is located in:
# ALmanuscript_Serrano_et_al/real_time_AL/utils/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHEMPROP_DIR="$REPO_DIR/real_time_AL/chemprop"
OUTPUT_DIR="$REPO_DIR/demo_data/real_time_AL_output"

mkdir -p "$OUTPUT_DIR/checkpoints"


# E. coli % control model
python "$CHEMPROP_DIR/train.py" \
    --data_path "$OUTPUT_DIR/ecoli_filt.csv" \
    --smiles_columns "SMILES" \
    --target_columns "Y" \
    --dataset_type "classification" \
    --save_dir "$OUTPUT_DIR/checkpoints" \
    --split_type "scaffold_balanced" \
    --num_folds 5 \
    --metric "auc" \
    --extra_metrics "binary_cross_entropy" "prc-auc" \
    --loss_function "dirichlet" \
    --evidential_regularization 0.2 \
    --class_balance \
    --epochs 200 \
    --features_path "$OUTPUT_DIR/ecoli_filt.npz" \
    --no_features_scaling
