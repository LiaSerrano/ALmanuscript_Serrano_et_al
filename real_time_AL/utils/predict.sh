#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-12:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

set -e

# predict.sh is located in:
# ALmanuscript_Serrano_et_al/real_time_AL/utils/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHEMPROP_DIR="$REPO_DIR/real_time_AL/chemprop"
OUTPUT_DIR="$REPO_DIR/demo_data/real_time_AL_output"

mkdir -p "$OUTPUT_DIR/model_predictions"

python "$CHEMPROP_DIR/predict.py" \
    --test_path "$OUTPUT_DIR/remaining_unscreened_molecules_filt.csv" \
    --smiles_columns "SMILES" \
    --checkpoint_dir "$OUTPUT_DIR/checkpoints" \
    --preds_path "$OUTPUT_DIR/model_predictions/ecoli_pcontrol.csv" \
    --features_path "$OUTPUT_DIR/remaining_unscreened_molecules_filt.npz" \
    --no_features_scaling \
    --uncertainty_method "dirichlet"
