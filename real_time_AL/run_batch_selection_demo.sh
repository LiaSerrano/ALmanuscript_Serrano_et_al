#!/bin/bash
#SBATCH -c 2
#SBATCH -t 5-00:00
#SBATCH -p gpu_quad
#SBATCH --mem=45G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err

set -e

module load conda/miniforge3/24.11.3-0
module load gcc/14.2.0
module load cuda/12.8

conda activate almanuscript_env


# Run from the root of ALmanuscript_Serrano_et_al
python real_time_AL/utils/get_remaining_molecules_demo.py \
    --previously_screened demo_data/previously_screened_demo.csv \
    --remaining_pool demo_data/remaining_pool_demo.csv \
    --output_dir demo_data/real_time_AL_output


python real_time_AL/utils/generateMiniMolInputs.py \
    --features_path demo_data/demo_ecoli_features.pkl \
    --remaining_unscreened_path demo_data/real_time_AL_output/remaining_unscreened_molecules.csv \
    --training_ecoli_path demo_data/demo_train.csv \
    --training_output_csv demo_data/real_time_AL_output/ecoli_filt.csv \
    --training_output_npz demo_data/real_time_AL_output/ecoli_filt.npz \
    --remaining_output_csv demo_data/real_time_AL_output/remaining_unscreened_molecules_filt.csv \
    --remaining_output_npz demo_data/real_time_AL_output/remaining_unscreened_molecules_filt.npz


bash real_time_AL/utils/train_demo.sh

bash real_time_AL/utils/predict.sh


python real_time_AL/utils/get_rationale_MCTS_demo.py \
    --data_path demo_data/real_time_AL_output/model_predictions/ecoli_pcontrol.csv \
    --output_dir demo_data/real_time_AL_output \
    --output_file rationales.csv \
    --property_name Y \
    --features_dict_path demo_data/demo_ecoli_features.pkl \
    --n_processes 1 \
    --missing_features parent \
    --device cpu \
    --model_paths \
        demo_data/real_time_AL_output/checkpoints/fold_0/model_0/model.pt \
        demo_data/real_time_AL_output/checkpoints/fold_1/model_0/model.pt \
        demo_data/real_time_AL_output/checkpoints/fold_2/model_0/model.pt \
        demo_data/real_time_AL_output/checkpoints/fold_3/model_0/model.pt \
        demo_data/real_time_AL_output/checkpoints/fold_4/model_0/model.pt


python real_time_AL/utils/batch_selection_demo.py \
    --ecoli_train_path demo_data/demo_train.csv \
    --ecoli_prediction_path demo_data/real_time_AL_output/model_predictions/ecoli_pcontrol.csv \
    --ecoli_rationale_path demo_data/real_time_AL_output/rationales.csv \
    --output_dir demo_data/real_time_AL_output
