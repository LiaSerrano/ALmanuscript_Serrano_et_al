
import os
import subprocess
from typing import List, Dict
from chemprop.data import MoleculeDataLoader, MoleculeDataset, MoleculeDatapoint
from chemprop.models import MoleculeModel
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import pyplot as plt
from datetime import datetime
import argparse
import torch
import chemprop
from scipy.special import psi  # digamma function




CHEMPROP_TRAIN_CMD = os.environ.get("CHEMPROP_TRAIN_CMD", "chemprop_train")
CHEMPROP_PREDICT_CMD = os.environ.get("CHEMPROP_PREDICT_CMD", "chemprop_predict")


def _is_valid_smiles(smiles: str) -> bool:
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return False
    # Disallow multi-component inputs for predict; chemprop v2 predict expects single molecules
    if '.' in smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _largest_fragment_smiles(smiles: str) -> str | None:
    """Return the largest (by heavy atoms) fragment from a possibly multi-component SMILES.
    Uses robust parsing (tries sanitize True then False). Tie-breaks by deprioritizing common salts and by string length.
    Returns None if no valid fragment can be parsed."""
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return None

    # Common counterions/salts to deprioritize in ties
    salt_like = {
        'Cl', '[Cl-]', 'Br', '[Br-]', 'I', '[I-]',
        'Na', '[Na+]', 'K', '[K+]', 'Li', '[Li+]',
        'F', '[F-]', 'O', 'H', '[H+]', 'HCl', 'HBr', 'HI'
    }

    best_part = None
    best_atoms = -1
    best_is_salt = True
    best_strlen = -1

    for part in smiles.split('.'):
        part = part.strip()
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            # retry without sanitization
            try:
                mol = Chem.MolFromSmiles(part, sanitize=False)
            except Exception:
                mol = None
        if mol is None:
            continue
        atoms = mol.GetNumHeavyAtoms()
        is_salt = part in salt_like
        strlen = len(part)

        # Primary: more heavy atoms wins
        take = atoms > best_atoms
        # Secondary: if equal atoms, prefer non-salt over salt
        if not take and atoms == best_atoms and best_part is not None:
            take = (best_is_salt and not is_salt)
        # Tertiary: if still equal, prefer longer SMILES string
        if not take and atoms == best_atoms and (best_is_salt == is_salt) and best_part is not None:
            take = strlen > best_strlen

        if take:
            best_part = part
            best_atoms = atoms
            best_is_salt = is_salt
            best_strlen = strlen

    return best_part

def largest_fragment(smiles: str) -> str | None:
    """Return the largest valid fragment."""
    if smiles is None or '.' not in smiles:
        return smiles
    fragments = smiles.split('.')
    largest = None
    max_atoms = -1
    for f in fragments:
        mol = Chem.MolFromSmiles(f)
        if mol is None:
            continue
        atoms = mol.GetNumHeavyAtoms()
        if atoms > max_atoms:
            max_atoms = atoms
            largest = f
    return largest

def safe_mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Return Mol object or None if featurization will fail."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol
    except Exception:
        return None
    
    

def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

def dirichlet_uncertainties(alpha):
    alpha = np.array(alpha, dtype=np.float64)
    # avoid extremely small values for numerical stability
    alpha = np.maximum(alpha, 1e-8)
    alpha0 = np.sum(alpha)
    K=len(alpha)
    # mean prediction
    rho_bar = alpha / alpha0

    # entropy of mean prediction
    entropy_mean = -np.sum(rho_bar * np.log(rho_bar + 1e-20))

    # expected entropy of a sample from Dirichlet
    expected_entropy = np.sum(rho_bar * (psi(alpha0) - psi(alpha)))

    # distribution uncertainty (mutual information)
    Udis = expected_entropy - entropy_mean
    Udis = K/alpha0

    # data uncertainty (expected entropy)
    Udata = np.sum(rho_bar * (psi(alpha0 + 1) - psi(alpha + 1)))

    return Udis, Udata


def train_chemprop_model(train_df: pd.DataFrame, save_dir: str, num_replicates: int = 1) -> str:
    os.makedirs(save_dir, exist_ok=True)

    df = train_df[['SMILES', 'Y']].dropna()
    # Drop rows with unparsable SMILES (RDKit returns None)
    parsable_mask = df['SMILES'].astype(str).apply(lambda s: Chem.MolFromSmiles(s) is not None)
    dropped = int((~parsable_mask).sum())
    if dropped > 0:
        print(f"Training: dropped {dropped} rows with invalid SMILES before writing train.csv")
    df = df.loc[parsable_mask]

    train_csv = os.path.join(save_dir, "train.csv")
    df.to_csv(train_csv, index=False)

    cmd = [
        CHEMPROP_TRAIN_CMD,
        "--data_path", train_csv,
        "--smiles_column", "SMILES",
        "--dataset_type", "classification",
        "--save_dir", save_dir,
        "--num_folds", "1",
        "--ensemble_size", str(num_replicates),
        "--metric", "auc",
        "--extra_metrics", "binary_cross_entropy",
        "--loss_function", "dirichlet",
        "--evidential_regularization", "0.2",
        "--class_balance",
        "--epochs", "75",
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stdout = e.stdout.decode(errors='ignore') if e.stdout else ''
        stderr = e.stderr.decode(errors='ignore') if e.stderr else ''
        raise RuntimeError(f"chemprop_train failed.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}") from e
    return save_dir


def _find_best_model_path(model_dir: str) -> str:
    candidates = [
        os.path.join(model_dir, "model_0", "best.pt"),
        os.path.join(model_dir, "best.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".pt"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No model checkpoint .pt found under {model_dir}")


def get_predictions_from_model_dir(model_dir: str, input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    if 'SMILES' not in df.columns:
        raise ValueError("input_df must contain a 'SMILES' column")

    # Use largest fragment for prediction (train remains on original data)
    df['SMILES_fragment'] = df['SMILES'].astype(str).map(_largest_fragment_smiles)
    valid_mask = df['SMILES_fragment'].notna()
    valid_smiles_df = (
        df.loc[valid_mask, ['SMILES_fragment']]
          .drop_duplicates()
          .rename(columns={'SMILES_fragment': 'SMILES'})
          .reset_index(drop=True)
    )

    preds_map: Dict[str, float] = {}
    uncertainty_map: Dict[str, float] = {}

    if len(valid_smiles_df) > 0:
        # Persist inputs for traceability
        tmp_input = os.path.join(model_dir, "predict_input.csv")
        valid_smiles_df.to_csv(tmp_input, index=False)

        # Load a single checkpoint (or best available)
        model_paths = _collect_model_paths(model_dir, max_models=1)
        models = load_models(model_paths, device='cpu')

        # Predict using Chemprop API with uncertainty
        smiles_list = valid_smiles_df['SMILES'].astype(str).tolist()
        avg_preds, avg_unc = make_prediction(models, smiles_list)

        # Ensure 1D arrays
        avg_preds = np.array(avg_preds).reshape(-1)
        avg_unc = np.array(avg_unc).reshape(-1) if avg_unc is not None else None

        # Map predictions back; if lengths mismatch, map the min length and warn
        n_map = min(len(smiles_list), len(avg_preds))
        if n_map < len(smiles_list):
            print(f"Warning: Only mapped {n_map}/{len(smiles_list)} predictions due to length mismatch")
        preds_map = dict(zip(smiles_list[:n_map], avg_preds[:n_map]))
        if avg_unc is not None and len(avg_unc) >= n_map:
            uncertainty_map = dict(zip(smiles_list[:n_map], avg_unc[:n_map]))

    out_df = df.copy()
    out_df['Y_pred'] = out_df['SMILES_fragment'].astype(str).map(preds_map)
    if len(uncertainty_map) > 0:
        out_df['Y_uncertainty'] = out_df['SMILES_fragment'].astype(str).map(uncertainty_map)
    out_df = out_df.drop(columns=['SMILES_fragment'])
    return out_df


def _collect_model_paths(model_dir: str, max_models: int | None = None) -> List[str]:
    paths = []
    # v1 layout: fold_0/model_i/model.pt or model_i/model.pt
    i = 0
    while True:
        candidate = os.path.join(model_dir, f"fold_0", f"model_{i}", "model.pt")
        if os.path.exists(candidate):
            paths.append(candidate)
            i += 1
            if max_models is not None and len(paths) >= max_models:
                break
            continue
        candidate2 = os.path.join(model_dir, f"model_{i}", "model.pt")
        if os.path.exists(candidate2):
            paths.append(candidate2)
            i += 1
            if max_models is not None and len(paths) >= max_models:
                break
            continue
        break
    # fallback: any model.pt
    if not paths:
        for root, _, files in os.walk(model_dir):
            for f in files:
                if f == 'model.pt':
                    paths.append(os.path.join(root, f))
        if max_models is not None:
            paths = paths[:max_models]
    if not paths:
        raise FileNotFoundError(f"No model checkpoints found under {model_dir}")
    return paths


def get_ensemble_predictions_from_model_dir(model_dir: str, input_df: pd.DataFrame, max_models: int = 5) -> pd.DataFrame:
    df = input_df.copy()
    if 'SMILES' not in df.columns:
        raise ValueError("input_df must contain a 'SMILES' column")

    df['SMILES_fragment'] = df['SMILES'].astype(str).map(_largest_fragment_smiles)
    valid_mask = df['SMILES_fragment'].notna()
    valid_smiles_df = (
        df.loc[valid_mask, ['SMILES_fragment']]
          .drop_duplicates()
          .rename(columns={'SMILES_fragment': 'SMILES'})
          .reset_index(drop=True)
    )

    preds_map: Dict[str, float] = {}
    entropy_map: Dict[str, float] = {}
    uncertainty_map: Dict[str, float] = {}
    individual_preds_maps = [{} for _ in range(max_models)]

    if len(valid_smiles_df) > 0:
        # Persist inputs for traceability
        tmp_input = os.path.join(model_dir, "predict_input.csv")
        valid_smiles_df.to_csv(tmp_input, index=False)

        # Load up to max_models checkpoints
        model_paths = _collect_model_paths(model_dir, max_models=max_models)
        models = load_models(model_paths, device='cpu')

        smiles_list = valid_smiles_df['SMILES'].astype(str).tolist()
        avg_preds, avg_unc, individual_preds = make_prediction(models, smiles_list, return_individual_preds=True)

        # Ensure 1D arrays
        avg_preds = np.array(avg_preds).reshape(-1)
        avg_unc = np.array(avg_unc).reshape(-1) if avg_unc is not None else None

        # Compute binary entropy from ensemble-averaged probabilities
        eps = 1e-12
        p_clip = np.clip(avg_preds, eps, 1 - eps)
        entropy = -(p_clip * np.log(p_clip) + (1 - p_clip) * np.log(1 - p_clip))

        n_map = min(len(smiles_list), len(avg_preds))
        if n_map < len(smiles_list):
            print(f"Warning: Only mapped {n_map}/{len(smiles_list)} predictions due to length mismatch")
        
        preds_map = dict(zip(smiles_list[:n_map], avg_preds[:n_map].astype(float)))
        entropy_map = dict(zip(smiles_list[:n_map], entropy[:n_map].astype(float)))
        if avg_unc is not None and len(avg_unc) >= n_map:
            uncertainty_map = dict(zip(smiles_list[:n_map], avg_unc[:n_map]))

        for i, preds in enumerate(individual_preds):
            preds = np.array(preds).reshape(-1)
            if n_map <= len(preds):
                individual_preds_maps[i] = dict(zip(smiles_list[:n_map], preds[:n_map].astype(float)))

    out_df = df.copy()
    out_df['Y_pred'] = out_df['SMILES_fragment'].astype(str).map(preds_map)
    out_df['Y_entropy'] = out_df['SMILES_fragment'].astype(str).map(entropy_map)
    if len(uncertainty_map) > 0:
        out_df['Y_uncertainty'] = out_df['SMILES_fragment'].astype(str).map(uncertainty_map)
    
    for i, pred_map in enumerate(individual_preds_maps):
        if pred_map:
            out_df[f'Y_pred_model_{i+1}'] = out_df['SMILES_fragment'].astype(str).map(pred_map)
            
    out_df = out_df.drop(columns=['SMILES_fragment'])
    return out_df

def load_models(model_paths: List[str], device: str = 'cpu') -> List:
    """Load Chemprop models from specified paths with proper CUDA memory management.

    Parameters
    ----------
    model_paths : List[str]
        List of paths to model checkpoint files
    device : str
        Device to load models on ('cuda' or 'cpu')

    Returns
    -------
    List
        List of loaded models
    """
    models = []
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load

    # Force CPU usage to prevent CUDA memory issues in multi-agent setting
    actual_device = 'cpu'
    if device == 'cuda' and torch.cuda.is_available():
        print(f"\033[1;33;40mWarning: Forcing CPU usage instead of CUDA to prevent memory conflicts in multi-agent setting\033[0m")

    try:
        for i, path in enumerate(model_paths):
            try:
                print(f"\033[1;34;40mLoading model {i+1}/{len(model_paths)} from {path} on {actual_device}\033[0m")
                model = chemprop.utils.load_checkpoint(
                    path=path,
                    device=torch.device(actual_device)
                )
                models.append(model)

                # Clear cache after each model to prevent accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError as e:
                print(f"\033[1;31;40mCUDA OOM loading model {path}, falling back to CPU\033[0m")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model = chemprop.utils.load_checkpoint(
                    path=path,
                    device=torch.device('cpu')
                )
                models.append(model)
            except Exception as e:
                print(f"\033[1;31;40mError loading model {path}: {e}\033[0m")
                # Try to continue with other models rather than failing completely
                continue

    except Exception as e:
        print(f"\033[1;31;40mCritical error in model loading: {e}\033[0m")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e
    finally:
        torch.load = _original_torch_load

    if not models:
        raise RuntimeError("No models could be loaded successfully")

    print(f"\033[1;32;40mSuccessfully loaded {len(models)} models on {actual_device}\033[0m")
    return models

def make_prediction(models, smiles, return_individual_preds=False):
    """Makes predictions on a list of SMILES with proper memory management.

    Parameters
    ----------
    models : List
        List of loaded Chemprop models
    smiles : List[str]
        SMILES strings to predict
    return_individual_preds : bool
        Whether to return predictions from individual models in the ensemble

    Returns
    -------
    np.ndarray
        Predicted values
    np.ndarray
        Uncertainty values (if applicable)
    List[np.ndarray]
        Individual model predictions (if return_individual_preds is True)
    """
    # Create a single MoleculeDatapoint for each complete SMILES string

    test_data = []
    for smi in smiles:
        try:
            datapoint = MoleculeDatapoint(
                smiles=[smi],  # Pass as a list with single SMILES
                #features_generator=["rdkit_2d_normalized"]
            )
            test_data.append(datapoint)
        except Exception as e:
            print(f"Error processing SMILES {smi}: {e}")
            default_return = (np.array([0.0]), np.array([1.0]))
            if return_individual_preds:
                default_return += ([],)
            return default_return

    test_dset = MoleculeDataset(test_data)
    test_loader = MoleculeDataLoader(
        test_dset,
        batch_size=1,
        num_workers=0,
        shuffle=False
    )

    try:
        with torch.inference_mode():
            preds_list = []
            unc_list = []
            for i, model in enumerate(models):
                try:
                    predss, alphas = chemprop.train.predict(model, test_loader, return_unc_parameters=True)
                    predss = torch.tensor(predss)
                    preds = torch.cat(tuple(predss), dim=0)
                    preds = preds.cpu().numpy()

                    alphas = np.array(alphas)
                    S = np.sum(alphas, axis=2)
                    num_classes = alphas.shape[2]
                    unc =  num_classes / S
                    unc = torch.tensor(unc)
                    unc = torch.cat(tuple(unc), dim=0)
                    unc = unc.cpu().numpy()

                    preds_list.append(preds)
                    unc_list.append(unc)

                    # Clear CUDA cache after each model prediction to prevent accumulation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError as e:
                    print(f"\033[1;31;40mCUDA OOM during prediction with model {i}, clearing cache\033[0m")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Skip this model and continue with others
                    continue
                except Exception as e:
                    print(f"\033[1;31;40mError during prediction with model {i}: {e}\033[0m")
                    # Skip this model and continue with others
                    continue

            if not preds_list:
                print(f"\033[1;31;40mNo successful predictions from any model\033[0m")
                default_return = (np.array([0.0]), np.array([1.0]))
                if return_individual_preds:
                    default_return += ([],)
                return default_return

            # Ensemble predictions
            sum_preds = sum(preds_list)
            avg_preds = sum_preds / len(preds_list)

            sum_unc = sum(unc_list)
            avg_unc = sum_unc / len(unc_list)

    except Exception as e:
        print(f"\033[1;31;40mCritical error during prediction: {e}\033[0m")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        default_return = (np.array([0.0]), np.array([1.0]))
        if return_individual_preds:
            default_return += ([],)
        return default_return

    if return_individual_preds:
        return avg_preds, avg_unc, preds_list

    return avg_preds, avg_unc

def combine_and_average_batches(model_outputs: List[List[torch.Tensor]]) -> torch.Tensor:

    combined_tensors = []
    for model_batches in model_outputs:
        # Concatenate all batch tensors along the last dimension
        combined = torch.cat(model_batches, dim=0)
        combined_tensors.append(combined)
    print(combined_tensors)
    # Stack combined tensors from all models and average element-wise
    averaged = torch.mean(torch.stack(combined_tensors), dim=0)
    return averaged

def Decompose_Uncertainty(pred_df, model_paths):
    each_model_alphas=[]
    print(model_paths)
    for i in model_paths:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(i, map_location='cpu')
        model = MoleculeModel(checkpoint['args'])  # args were saved in checkpoint
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
            
        smiles_list = list(zip(list(pred_df['SMILES']), list(pred_df['Plate'])))
    
        filtered_smiles=[]
        datapoints = []
        
        for s, plate in smiles_list:
            try:
                clean = canonicalize_smiles(largest_fragment(s))
                mol = safe_mol_from_smiles(clean)
                if mol is not None:
                    datapoints.append(MoleculeDatapoint([clean]))
                    filtered_smiles.append((clean, plate))
            except Exception:
                continue
            
    
        dataset = MoleculeDataset(datapoints)
        loader = MoleculeDataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        

        all_alphas = []
    
        with torch.no_grad():
            for batch in loader:
                print(batch.batch_graph())
                try:
                    outputs = model(batch.batch_graph())      # Dirichlet evidence
                    alphas = outputs + 1.0      # alpha = evidence + 1
                    all_alphas.append(alphas.cpu())
                except Exception as e:
                    print(f"Skipping batch due to error: {e}")
        
        
        
        if all_alphas:
            each_model_alphas.append(all_alphas)
    
        else:
            print(f"⚠️ Skipping model {i}: no valid alphas generated")


    arralph=np.array(combine_and_average_batches(each_model_alphas).tolist())
    
    
    data_dictionary={}
    
    count=0
    
    for alpha in arralph:
        if filtered_smiles[count][1] not in data_dictionary:
            data_dictionary[filtered_smiles[count][1]]={'udis':[], 
                                                        'udata':[], 
                                                        'uncertainty':[], 
                                                        'beta var':[]
                                                        
                                                        }
        
        
        udis, udata = dirichlet_uncertainties(np.array(alpha))
        uncertainty=2/sum(alpha)
        beta_var=max(alpha)*(sum(alpha)-max(alpha))/((sum(alpha)**2)*(sum(alpha)+1))
                                                     
        data_dictionary[filtered_smiles[count][1]]['udis'].append(udis)
        data_dictionary[filtered_smiles[count][1]]['udata'].append(udata)
        data_dictionary[filtered_smiles[count][1]]['uncertainty'].append(uncertainty)
        data_dictionary[filtered_smiles[count][1]]['beta var'].append(beta_var)

        count+=1
        
    return data_dictionary

def plate_scoring_random(pred_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Select k plates randomly from the pool."""
    available_plates = pred_df['Plate'].unique()
    selected_plates = np.random.choice(available_plates, size=min(k, len(available_plates)), replace=False)
    return pred_df[pred_df['Plate'].isin(selected_plates)].copy()


def plate_scoring_total_inhibition(pred_df: pd.DataFrame, k: int) -> pd.DataFrame:
    grouped = pred_df.groupby('Plate', as_index=False)['Y_pred'].mean()
    ranked_plates = grouped.sort_values('Y_pred', ascending=False).head(k)['Plate'].tolist()
    return pred_df[pred_df['Plate'].isin(ranked_plates)].copy()




def _smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def evaluate_on_holdout(pred_df: pd.DataFrame, full_labels_df: pd.DataFrame, num_replicates: int = 1) -> List[Dict[str, float]]:
    merged = pred_df.merge(full_labels_df[['SMILES', 'Y']], on='SMILES', how='left')
    
    all_metrics = []

    for i in range(1, num_replicates + 1):
        pred_col = f'Y_pred_model_{i}'
        if pred_col not in merged.columns:
            continue

        # Require both label and prediction
        sub_merged = merged.dropna(subset=['Y', pred_col])
        y_true = sub_merged['Y'].astype(float).values
        y_score = sub_merged[pred_col].astype(float).values
        
        metrics = {"auroc": np.nan, "auprc": np.nan}
        if len(y_true) > 1 and len(np.unique(y_true)) > 1:
            try:
                metrics["auroc"] = roc_auc_score(y_true, y_score)
            except Exception:
                pass
            try:
                metrics["auprc"] = average_precision_score(y_true, y_score)
            except Exception:
                pass
        all_metrics.append(metrics)
        
    # Also evaluate the ensemble average
    if 'Y_pred' in merged.columns:
        sub_merged = merged.dropna(subset=['Y', 'Y_pred'])
        y_true = sub_merged['Y'].astype(float).values
        y_score = sub_merged['Y_pred'].astype(float).values
        
        metrics = {"auroc": np.nan, "auprc": np.nan}
        if len(y_true) > 1 and len(np.unique(y_true)) > 1:
            try:
                metrics["auroc"] = roc_auc_score(y_true, y_score)
            except Exception:
                pass
            try:
                metrics["auprc"] = average_precision_score(y_true, y_score)
            except Exception:
                pass
        all_metrics.append(metrics)
        
    return all_metrics


def plot_holdout_metrics(history: List[Dict], work_dir: str) -> None:
    """Plot AUROC and AUPRC over iterations and save to work_dir."""
    if not history:
        return
    df_hist = pd.DataFrame(history)

    # Extract individual model AUROCs
    auroc_cols = [col for col in df_hist.columns if 'holdout_auroc_model_' in col]
    auprc_cols = [col for col in df_hist.columns if 'holdout_auprc_model_' in col]

    # AUROC plot
    plt.figure(figsize=(10, 5))
    for col in auroc_cols:
        plt.plot(df_hist['iteration'], df_hist[col], marker='o', linestyle='--', label=col)
    
    # Plot average AUROC if available
    if 'holdout_auroc' in df_hist.columns:
        # This will be the ensemble prediction from the last entry
        ensemble_auroc = df_hist['holdout_auroc'].apply(lambda x: x[-1] if isinstance(x, list) and x else None)
        plt.plot(df_hist['iteration'], ensemble_auroc, marker='o', color='black', linewidth=2, label='Ensemble AUROC')

    plt.xlabel('Iteration')
    plt.ylabel('Holdout AUROC')
    plt.title('Holdout AUROC by Iteration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, 'holdout_auroc_by_iteration.png'))
    plt.close()

    # AUPRC plot
    plt.figure(figsize=(10, 5))
    for col in auprc_cols:
        plt.plot(df_hist['iteration'], df_hist[col], marker='o', linestyle='--', label=col)
        
    if 'holdout_auprc' in df_hist.columns:
        ensemble_auprc = df_hist['holdout_auprc'].apply(lambda x: x[-1] if isinstance(x, list) and x else None)
        plt.plot(df_hist['iteration'], ensemble_auprc, marker='o', color='black', linewidth=2, label='Ensemble AUPRC')
        
    plt.xlabel('Iteration')
    plt.ylabel('Holdout AUPRC')
    plt.title('Holdout AUPRC by Iteration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, 'holdout_auprc_by_iteration.png'))
    plt.close()

    # Percent Hits Screened Plot
    if 'percent_hits_screened' in df_hist.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df_hist['iteration'], df_hist['percent_hits_screened'], marker='o', color='tab:red')
        plt.xlabel('Iteration')
        plt.ylabel('% of Total Hits Screened')
        plt.title('Percentage of Hits Screened by Iteration')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, 'percent_hits_screened_by_iteration.png'))
        plt.close()


def _normalize_series(values: pd.Series) -> pd.Series:
    vals = values.astype(float)
    vmin = vals.min()
    vmax = vals.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return pd.Series(np.zeros(len(vals)), index=vals.index)
    return (vals - vmin) / (vmax - vmin)


def annotate_novelty_column(
    pred_df: pd.DataFrame,
    train_df: pd.DataFrame,
    radius: int = 2,
    n_bits: int = 2048,
) -> pd.DataFrame:
    """Return a copy of pred_df with a 'novelty' column (1 - mean Tanimoto to train)."""
    # Build training fingerprint list (unique SMILES)
    train_smiles = pd.Series(train_df['SMILES'].astype(str).unique()).tolist()
    train_fps = [
        _smiles_to_morgan_fp(s, radius=radius, n_bits=n_bits)
        for s in train_smiles
    ]
    train_fps = [fp for fp in train_fps if fp is not None]

    if len(train_fps) == 0:
        novelty_per_smiles = {s: 1.0 for s in pred_df['SMILES'].astype(str).unique()}
    else:
        novelty_per_smiles = {}
        for s in pred_df['SMILES'].astype(str).unique():
            fp = _smiles_to_morgan_fp(s, radius=radius, n_bits=n_bits)
            if fp is None:
                novelty_per_smiles[s] = np.nan
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            novelty_per_smiles[s] = 1.0 - float(np.nanmean(sims)) if len(sims) > 0 else 1.0

    df_with_novelty = pred_df.copy()
    df_with_novelty['novelty'] = df_with_novelty['SMILES'].astype(str).map(novelty_per_smiles)
    return df_with_novelty


def plate_scoring_batch_selection_UdisUdata(
    pred_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int,
    prediction_threshold: float = 0.7,
    novelty_threshold: float = 0.5,
    radius: int = 2,
    n_bits: int = 2048,
    uncertainty_data= dict
) -> pd.DataFrame:
    """Selects k plates based on a ranking scheme inspired by batch_selection.py."""
    prediction_col = 'Y_pred'
    target_col = 'Y'

    pred_df_copy = pred_df.copy()
    pred_df_copy['potential_inhibitory'] = np.where(pred_df_copy[prediction_col] >= prediction_threshold, 1, 0)

    hit_df = pred_df_copy[pred_df_copy['potential_inhibitory'] == 1].copy()
    nonhit_df = pred_df_copy[pred_df_copy['potential_inhibitory'] == 0].copy()

    # Hit Novelty vs. positive training examples
    train_pos_df = train_df[train_df[target_col] >= prediction_threshold]
    ref_fps_pos = [_smiles_to_morgan_fp(s, radius=radius, n_bits=n_bits) for s in train_pos_df['SMILES'].unique()]
    ref_fps_pos = [fp for fp in ref_fps_pos if fp is not None]

    if not hit_df.empty:
        hit_smiles = hit_df['SMILES']
        hit_fps = [_smiles_to_morgan_fp(s, radius=radius, n_bits=n_bits) for s in hit_smiles]
        
        max_sim_list = []
        for hit_fp in hit_fps:
            if hit_fp is None or not ref_fps_pos:
                max_sim_list.append(0)
                continue
            sims = DataStructs.BulkTanimotoSimilarity(hit_fp, ref_fps_pos)
            max_sim_list.append(max(sims) if sims else 0)
        hit_df['hit_novelty'] = 1 - np.array(max_sim_list)

    # Non-Hit Novelty vs. negative training examples
    train_neg_df = train_df[train_df[target_col] < prediction_threshold]
    ref_fps_neg = [_smiles_to_morgan_fp(s, radius=radius, n_bits=n_bits) for s in train_neg_df['SMILES'].unique()]
    ref_fps_neg = [fp for fp in ref_fps_neg if fp is not None]
    
    if not nonhit_df.empty:
        nonhit_smiles = nonhit_df['SMILES']
        nonhit_fps = [_smiles_to_morgan_fp(s, radius=radius, n_bits=n_bits) for s in nonhit_smiles]

        max_sim_list_neg = []
        for nonhit_fp in nonhit_fps:
            if nonhit_fp is None or not ref_fps_neg:
                max_sim_list_neg.append(0)
                continue
            sims = DataStructs.BulkTanimotoSimilarity(nonhit_fp, ref_fps_neg)
            max_sim_list_neg.append(max(sims) if sims else 0)
        nonhit_df['nonhit_novelty'] = 1 - np.array(max_sim_list_neg)

    # Merge novelty scores back
    if 'hit_novelty' in hit_df.columns:
        pred_df_copy = pred_df_copy.merge(hit_df[['SMILES', 'Plate', 'hit_novelty']], on=['SMILES', 'Plate'], how='left')
    else:
        pred_df_copy['hit_novelty'] = 0.0

    if 'nonhit_novelty' in nonhit_df.columns:
        pred_df_copy = pred_df_copy.merge(nonhit_df[['SMILES', 'Plate', 'nonhit_novelty']], on=['SMILES', 'Plate'], how='left')
    else:
        pred_df_copy['nonhit_novelty'] = 0.0

    pred_df_copy['hit_novelty'] = pred_df_copy['hit_novelty'].fillna(0)
    pred_df_copy['nonhit_novelty'] = pred_df_copy['nonhit_novelty'].fillna(0)

    # Rank Plates
    pred_df_copy['novel_hits'] = np.where(pred_df_copy['hit_novelty'] >= novelty_threshold, 1, 0)
    
    rank_df = pred_df_copy.groupby('Plate', as_index=False).agg(
        potential_inhibitory=('potential_inhibitory', 'sum'),
        novel_hits=('novel_hits', 'sum'),
        nonhit_novelty=('nonhit_novelty', 'mean')
    )

    # In-plate diversity (clustering)
    plate_cluster_num = []
    for plate in rank_df['Plate']:
        plate_df = pred_df_copy[(pred_df_copy['Plate'] == plate) & (pred_df_copy['potential_inhibitory'] == 1)]
        if len(plate_df) < 2:
            plate_cluster_num.append(len(plate_df))
            continue
        
        plate_smiles = list(plate_df['SMILES'])
        plate_fps = [_smiles_to_morgan_fp(s, radius=2, n_bits=1024) for s in plate_smiles]
        plate_fps = [fp for fp in plate_fps if fp is not None]

        if len(plate_fps) < 2:
            plate_cluster_num.append(len(plate_fps))
            continue
        
        dists = []
        nfps = len(plate_fps)
        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(plate_fps[i], plate_fps[:i])
            dists.extend([1-x for x in sims])
        
        cs = Butina.ClusterData(dists, nfps, 0.5, isDistData=True)
        plate_cluster_num.append(len(cs))
        
    rank_df['num_clusters'] = plate_cluster_num


    udis_list, udata_list = [], []
    for plate in rank_df["Plate"]:
        if plate in uncertainty_data:
            udis_list.append(np.mean(uncertainty_data[plate]["udis"]))
            udata_list.append(np.mean(uncertainty_data[plate]["udata"]))
        else:
            udis_list.append(0.0)
            udata_list.append(0.0)
    rank_df["udis"] = udis_list
    rank_df["udata"] = udata_list



    # Overall rank calculation
    selection_weights = {
        'nonhit_novelty': 1,
        'potential_inhibitory': 1,
        'novel_hits': 2,
        'num_clusters': 1,
        "udis": 1,    # prioritize epistemic uncertainty
        "udata": 1,   # prioritize aleatoric uncertainty
    }
    
    
    rank_df['overall_rank'] = 0
    for column, weight in selection_weights.items():
        rank_df[f'rank_{column}'] = rank_df[column].rank(ascending=False)
        rank_df['overall_rank'] += weight * rank_df[f'rank_{column}']
    
    rank_df['overall_rank'] = rank_df['overall_rank'].rank(ascending=True)
    rank_df = rank_df.sort_values(by='overall_rank', ascending=True)

    ranked_plates = rank_df.head(k)['Plate'].tolist()
    return pred_df[pred_df['Plate'].isin(ranked_plates)].copy()


def plate_scoring_batch_selection(
    pred_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int,
    prediction_threshold: float = 0.7,
    novelty_threshold: float = 0.5,
    radius: int = 2,
    n_bits: int = 2048,
) -> pd.DataFrame:
    """Selects k plates based on a ranking scheme inspired by batch_selection.py."""
    prediction_col = 'Y_pred'
    target_col = 'Y'

    pred_df_copy = pred_df.copy()
    pred_df_copy['potential_inhibitory'] = np.where(pred_df_copy[prediction_col] >= prediction_threshold, 1, 0)

    hit_df = pred_df_copy[pred_df_copy['potential_inhibitory'] == 1].copy()
    nonhit_df = pred_df_copy[pred_df_copy['potential_inhibitory'] == 0].copy()

    # Hit Novelty vs. positive training examples
    train_pos_df = train_df[train_df[target_col] >= prediction_threshold]
    ref_fps_pos = [_smiles_to_morgan_fp(s, radius=radius, n_bits=n_bits) for s in train_pos_df['SMILES'].unique()]
    ref_fps_pos = [fp for fp in ref_fps_pos if fp is not None]

    if not hit_df.empty:
        hit_smiles = hit_df['SMILES']
        hit_fps = [_smiles_to_morgan_fp(s, radius=radius, n_bits=n_bits) for s in hit_smiles]
        
        max_sim_list = []
        for hit_fp in hit_fps:
            if hit_fp is None or not ref_fps_pos:
                max_sim_list.append(0)
                continue
            sims = DataStructs.BulkTanimotoSimilarity(hit_fp, ref_fps_pos)
            max_sim_list.append(max(sims) if sims else 0)
        hit_df['hit_novelty'] = 1 - np.array(max_sim_list)

    # Non-Hit Novelty vs. negative training examples
    train_neg_df = train_df[train_df[target_col] < prediction_threshold]
    ref_fps_neg = [_smiles_to_morgan_fp(s, radius=radius, n_bits=n_bits) for s in train_neg_df['SMILES'].unique()]
    ref_fps_neg = [fp for fp in ref_fps_neg if fp is not None]
    
    if not nonhit_df.empty:
        nonhit_smiles = nonhit_df['SMILES']
        nonhit_fps = [_smiles_to_morgan_fp(s, radius=radius, n_bits=n_bits) for s in nonhit_smiles]

        max_sim_list_neg = []
        for nonhit_fp in nonhit_fps:
            if nonhit_fp is None or not ref_fps_neg:
                max_sim_list_neg.append(0)
                continue
            sims = DataStructs.BulkTanimotoSimilarity(nonhit_fp, ref_fps_neg)
            max_sim_list_neg.append(max(sims) if sims else 0)
        nonhit_df['nonhit_novelty'] = 1 - np.array(max_sim_list_neg)

    # Merge novelty scores back
    if 'hit_novelty' in hit_df.columns:
        pred_df_copy = pred_df_copy.merge(hit_df[['SMILES', 'Plate', 'hit_novelty']], on=['SMILES', 'Plate'], how='left')
    else:
        pred_df_copy['hit_novelty'] = 0.0

    if 'nonhit_novelty' in nonhit_df.columns:
        pred_df_copy = pred_df_copy.merge(nonhit_df[['SMILES', 'Plate', 'nonhit_novelty']], on=['SMILES', 'Plate'], how='left')
    else:
        pred_df_copy['nonhit_novelty'] = 0.0

    pred_df_copy['hit_novelty'] = pred_df_copy['hit_novelty'].fillna(0)
    pred_df_copy['nonhit_novelty'] = pred_df_copy['nonhit_novelty'].fillna(0)

    # Rank Plates
    pred_df_copy['novel_hits'] = np.where(pred_df_copy['hit_novelty'] >= novelty_threshold, 1, 0)
    
    rank_df = pred_df_copy.groupby('Plate', as_index=False).agg(
        potential_inhibitory=('potential_inhibitory', 'sum'),
        novel_hits=('novel_hits', 'sum'),
        nonhit_novelty=('nonhit_novelty', 'mean')
    )

    # In-plate diversity (clustering)
    plate_cluster_num = []
    for plate in rank_df['Plate']:
        plate_df = pred_df_copy[(pred_df_copy['Plate'] == plate) & (pred_df_copy['potential_inhibitory'] == 1)]
        if len(plate_df) < 2:
            plate_cluster_num.append(len(plate_df))
            continue
        
        plate_smiles = list(plate_df['SMILES'])
        plate_fps = [_smiles_to_morgan_fp(s, radius=2, n_bits=1024) for s in plate_smiles]
        plate_fps = [fp for fp in plate_fps if fp is not None]

        if len(plate_fps) < 2:
            plate_cluster_num.append(len(plate_fps))
            continue
        
        dists = []
        nfps = len(plate_fps)
        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(plate_fps[i], plate_fps[:i])
            dists.extend([1-x for x in sims])
        
        cs = Butina.ClusterData(dists, nfps, 0.5, isDistData=True)
        plate_cluster_num.append(len(cs))
        
    rank_df['num_clusters'] = plate_cluster_num


    # Overall rank calculation
    selection_weights = {
        'nonhit_novelty': 1,
        'potential_inhibitory': 1,
        'novel_hits': 2,
        'num_clusters': 1

    }
    
    
    rank_df['overall_rank'] = 0
    for column, weight in selection_weights.items():
        rank_df[f'rank_{column}'] = rank_df[column].rank(ascending=False)
        rank_df['overall_rank'] += weight * rank_df[f'rank_{column}']
    
    rank_df['overall_rank'] = rank_df['overall_rank'].rank(ascending=True)
    rank_df = rank_df.sort_values(by='overall_rank', ascending=True)

    ranked_plates = rank_df.head(k)['Plate'].tolist()
    return pred_df[pred_df['Plate'].isin(ranked_plates)].copy()






def run_active_learning_simulation(
    full_data: pd.DataFrame,
    initial_train_plates: List[int],
    holdout_plates: List[int],
    acquisition_batch_size: int = 5,
    max_iterations: int = 5,
    work_dir: str = None,
    acquisition_strategy: str = 'mean_pred',  
    novelty_radius: int = 2,
    novelty_n_bits: int = 2048,
    num_replicates: int = 3,
):
    if work_dir is None:
        work_dir = os.path.join(os.getcwd(), "active_learning_outputs")
    os.makedirs(work_dir, exist_ok=True)

    df = full_data.copy()
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce').fillna(0.0)
 
    # Define a hit per user's spec: label equals 1 under Y
    df['is_hit'] = (df['Y'].astype(float) == 1.0).astype(int)
    total_hits = df['is_hit'].sum()

    is_holdout = df['Plate'].isin(holdout_plates)
    is_train = df['Plate'].isin(initial_train_plates)
    is_pool = ~(is_holdout | is_train)

    train_df = df[is_train].copy()
    holdout_df = df[is_holdout].copy()
    pool_df = df[is_pool].copy()

    history = []

    for it in range(1, max_iterations + 1):
        it_dir = os.path.join(work_dir, f"iter_{it:02d}")
        os.makedirs(it_dir, exist_ok=True)

        model_dir = os.path.join(it_dir, "model")
        train_chemprop_model(train_df, model_dir, num_replicates=num_replicates)

        pool_pred = get_ensemble_predictions_from_model_dir(model_dir, pool_df[['SMILES', 'Plate']].drop_duplicates(), max_models=num_replicates)
        holdout_pred = get_ensemble_predictions_from_model_dir(model_dir, holdout_df[['SMILES', 'Plate']].drop_duplicates(), max_models=num_replicates)

        pred_merged_pool = pool_pred.merge(pool_df[['SMILES','Plate']], on=['SMILES','Plate'], how='left')
        
       
        # Choose acquisition strategy
        if acquisition_strategy == 'mean_uncertainty':
            selected_pool = plate_scoring_mean_uncertainty(pred_merged_pool, acquisition_batch_size)
            
        elif acquisition_strategy == 'batch_selection_UdisUdata':
            UncertaintyDictionary=Decompose_Uncertainty(pred_merged_pool, _collect_model_paths(model_dir, max_models=num_replicates))
            selected_pool = plate_scoring_batch_selection_UdisUdata(
                pred_merged_pool,
                train_df=train_df,
                k=acquisition_batch_size,
                radius=novelty_radius,
                n_bits=novelty_n_bits,
                uncertainty_data=UncertaintyDictionary
            )

        elif acquisition_strategy == 'batch_selection':
            selected_pool = plate_scoring_batch_selection(
            pred_merged_pool,
            train_df=train_df,
            k=acquisition_batch_size,
            radius=novelty_radius,
            n_bits=novelty_n_bits,
        )
            
        elif acquisition_strategy == 'random':
            selected_pool = plate_scoring_random(pred_merged_pool, acquisition_batch_size)


        else:
            # default: by mean predicted inhibition
            selected_pool = plate_scoring_total_inhibition(pred_merged_pool, acquisition_batch_size)

        selected_plates = sorted(selected_pool['Plate'].unique().tolist())

        newly_selected_df = pool_df[pool_df['Plate'].isin(selected_plates)].copy()
        train_df = pd.concat([train_df, newly_selected_df], ignore_index=True)
        pool_df = pool_df[~pool_df['Plate'].isin(selected_plates)].copy()

        # Evaluate and persist holdout predictions/metrics
        metrics = evaluate_on_holdout(holdout_pred, holdout_df, num_replicates=num_replicates)
        holdout_eval = holdout_pred.merge(holdout_df[['SMILES','Plate','Y']], on=['SMILES','Plate'], how='left')
        holdout_eval.to_csv(os.path.join(it_dir, 'holdout_predictions.csv'), index=False)
        
        auroc_strs = [f"{m['auroc']:.4f}" if m['auroc'] == m['auroc'] else 'NA' for m in metrics]
        auprc_strs = [f"{m['auprc']:.4f}" if m['auprc'] == m['auprc'] else 'NA' for m in metrics]
        print(f"Iteration {it}: Holdout AUROC={auroc_strs}, AUPRC={auprc_strs}")

        # Track hits screened
        hits_screened = train_df[train_df['is_hit'] == 1]['SMILES'].nunique()
        percent_hits_screened = (hits_screened / total_hits) * 100 if total_hits > 0 else 0.0

        # Create and save history entry for this iteration
        history_entry = {
            'iteration': it,
            'selected_plates': selected_plates,
            'num_train_samples': int(len(train_df)),
            'holdout_auroc': [m['auroc'] for m in metrics],
            'holdout_auprc': [m['auprc'] for m in metrics],
            'percent_hits_screened': percent_hits_screened,
            'acquisition_strategy': acquisition_strategy,
        }
        history.append(history_entry)

        # Update history CSV after each iteration
        history_df = pd.DataFrame(history)
        # Explode list columns into separate columns for easier analysis
        for metric in ['holdout_auroc', 'holdout_auprc']:
            metric_df = history_df[metric].apply(pd.Series)
            metric_df = metric_df.rename(columns=lambda x: f'{metric}_model_{x+1}')
            history_df = pd.concat([history_df.drop([metric], axis=1), metric_df], axis=1)
        
        history_df.to_csv(os.path.join(work_dir, 'run_history.csv'), index=False)

        if len(pool_df) == 0:
            break

    return history 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="lyme",
        help="Task to run",
    )
    parser.add_argument(
        "--acquisition_strategy",
        type=str,
        choices=[
            "mean_pred",
            "batch_selection",
            "random",
            'batch_selection_UdisUdata',
        ],
        default="mean_pred",
        help="Acquisition strategy",
        )
    parser.add_argument(
        "--initial_train_plates",
        type=str,
        default=None
        )
    
    parser.add_argument(
            "--out_dir",
            type=str,
            required=True,
            help="Base directory for simulation outputs"
        )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to demo training CSV"
    )




    args = parser.parse_args()
    task = args.task
    acquisition_strategy = args.acquisition_strategy
    train_plates = args.initial_train_plates
    #base_out_dir=args.out_dir


    if task == "ecoli":
        data = pd.read_csv(args.data_path)
        upload_train_plates=list(pd.read_csv(train_plates)['Plate'])
        initial_train_plates = list(set(upload_train_plates))
        holdout_plates = [500]
    

    else:
        raise ValueError(f"Task {task} not supported")

    base_out_dir = args.out_dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(base_out_dir, acquisition_strategy, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    history = run_active_learning_simulation(
        full_data=data,
        initial_train_plates=initial_train_plates,
        holdout_plates=holdout_plates,
        acquisition_batch_size=5,
        max_iterations=35,
        work_dir=out_dir,
        acquisition_strategy=acquisition_strategy,
        novelty_radius=2,
        novelty_n_bits=2048,
        num_replicates=3,
    )
    print(history)
    # Plot and save curves
    plot_holdout_metrics(history, out_dir)
