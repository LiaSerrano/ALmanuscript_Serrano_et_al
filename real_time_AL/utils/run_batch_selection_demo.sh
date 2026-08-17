#!/bin/bash
#SBATCH -c 2
#SBATCH -t 72:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=45G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err


module load conda/miniforge3/24.11.3-0
module load gcc/14.2.0
module load python/3.13.1
module load cuda/12.8

conda activate /n/data1/hms/dbmi/farhat/LRS/env/

python get_remaining_molecules.py \
    --previously_screened demo_data/previously_screened_demo.csv \
    --remaining_pool demo_data/remaining_pool_demo.csv

python generateMiniMolInputs.py \
    --features_path demo_ecoli_features.npz \
    --remaining_unscreened_path demo_data/remaining_unscreened_molecules.csv \
    --training_ecoli_path demo_data/demo_train.csv \
    --training_output_csv demo_data/ecoli_filt.csv \
    --training_output_npz demo_data/ecoli_filt.npz \
    --remaining_output_csv demo_data/remaining_unscreened_molecules_filt.csv \
    --remaining_output_npz demo_data/remaining_unscreened_molecules_filt.npz


bash train_demo.sh

bash predict.sh

python get_rationale_MCTS_demo.py \
  --data_path /demo_data/model_predictions/ecoli_pcontrol.csv \
  --output_dir /demo_data/ \
  --output_file rationales.csv \
  --property_name Y \
  --features_dict_path demo_ecoli_features.npz \
  --n_processes 1 \
  --missing_features parent \
  --device cpu \
  --model_paths \
    /path/to/fold_0/model_0/model.pt \
    /path/to/fold_1/model_0/model.pt \
    /path/to/fold_2/model_0/model.pt \
    /path/to/fold_3/model_0/model.pt \
    /path/to/fold_4/model_0/model.pt



conda activate /n/data1/hms/dbmi/farhat/LRS/env/ 

python batch_selection.py 
