
python predict.py --test_path "/n/scratch/users/a/aw274/checkpoints_dmpnn/evidential_classification_0.2_morgan_output/fold_0/test_data.csv" \
                  --checkpoint_dir "/n/scratch/users/a/aw274/checkpoints_dmpnn/evidential_classification_0.2_morgan_output" \
                  --preds_path "/n/scratch/users/a/aw274/predictions_dmpnn/classification_prediction/test_set_predictions_0.2_lambda_morgan.csv"\
                  --features_generator 'morgan'\
                  --uncertainty_method 'dirichlet'