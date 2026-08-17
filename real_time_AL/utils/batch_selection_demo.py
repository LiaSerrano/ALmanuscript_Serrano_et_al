import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from tqdm import tqdm
import rationale_utils
import argparse

def plate_scoring(
                  ecoli_prediction_df: pd.DataFrame,
                  ecoli_prediction_target_name: str,
                  ecoli_train_df: pd.DataFrame,
                  novelty_weight: float,
                  novelty_threshold: float,
                  ecoli_rationale_df: pd.DataFrame):
    
    if ecoli_prediction_df is None:
        print("No Ecoli Prediction provided")

    prediction_threshold = 0.7
    

    def clean_list(series):
        return [item for item in series if pd.notna(item)]  # Filter out NaNvalues


    def unique_rationales(series):
        # Flatten the lists and return unique items as a sorted list
        return sorted(set(item for sublist in series for item in sublist))


    ##### Ecoli #####

    ecoli_prediction_df = ecoli_prediction_df[ecoli_prediction_df[ecoli_prediction_target_name] != 'Invalid SMILES'].copy()
    ecoli_prediction_df[ecoli_prediction_target_name] = pd.to_numeric(ecoli_prediction_df[ecoli_prediction_target_name])


    ## E.coli - Binarize and Drop Duplicate
    ecoli_prediction_df['potential_ecoli_inhibitory'] = np.where(ecoli_prediction_df[ecoli_prediction_target_name] >= prediction_threshold, 1 ,0)
    ecoli_prediction_df = ecoli_prediction_df.drop_duplicates()

    ## E.coli - Rationale Matching
    ecoli_rationale_df = ecoli_rationale_df.rename(columns={'smiles': 'SMILES'})
    ecoli_rationale_df = ecoli_rationale_df.loc[:, (ecoli_rationale_df.columns.str.startswith('rationale_') & (~ecoli_rationale_df.columns.str.endswith('score'))) | (ecoli_rationale_df.columns == 'SMILES')]

    ecoli_hit_df = ecoli_prediction_df[ecoli_prediction_df['potential_ecoli_inhibitory'] == 1].copy()

    if (len(ecoli_hit_df[~ecoli_hit_df['SMILES'].isin(ecoli_rationale_df['SMILES'])]) > 0 | len(ecoli_rationale_df[~ecoli_rationale_df['SMILES'].isin(ecoli_hit_df['SMILES'])]) > 0):
        print("Rationale Prediction Files MisMatch")
        exit()

    ecoli_hit_df = ecoli_hit_df.merge(ecoli_rationale_df,
                                      on = 'SMILES',
                                      how = 'left')
    
    ecoli_hit_df['rationale_0'] = ecoli_hit_df['rationale_0'].fillna(ecoli_hit_df['SMILES'])

    rationale_cols = ecoli_hit_df.filter(like='rationale').columns
    ecoli_hit_df['rationales'] = ecoli_hit_df[rationale_cols].apply(list, axis=1)
    ecoli_hit_df = ecoli_hit_df.drop(rationale_cols, axis=1)

    ecoli_hit_df['rationales'] = ecoli_hit_df['rationales'].apply(clean_list)

    ## E.coli - Hit Novelty
    hit_smiles = ecoli_hit_df['SMILES']
    hit_mols = [Chem.MolFromSmiles(x) for x in hit_smiles]
    hit_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048) if x is not None else None for x in hit_mols]

    ecoli_train_pos_df = ecoli_train_df[ecoli_train_df[ecoli_prediction_target_name] >= 0.7]
    ref_smiles = ecoli_train_pos_df['SMILES']
    ref_mols = [Chem.MolFromSmiles(x) for x in ref_smiles]
    ref_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048) if x is not None else None for x in ref_mols]

    max_sim_list = []
    for hit_fp in tqdm(hit_fps):
        sim_list = []
        for ref_fp in ref_fps:
            if (hit_fp is not None) and (ref_fp is not None):
                sim_list.append(DataStructs.TanimotoSimilarity(hit_fp, ref_fp))
        max_sim_list.append(max(sim_list))

    ecoli_hit_df['hit_novelty'] = max_sim_list
    ecoli_hit_df['hit_novelty'] = 1 - ecoli_hit_df['hit_novelty']

    ## E.coli - Non-Hit Novelty
    ecoli_nonhit_df = ecoli_prediction_df[ecoli_prediction_df['potential_ecoli_inhibitory'] == 0]
    nonhit_smiles = ecoli_nonhit_df['SMILES']
    nonhit_mols = [Chem.MolFromSmiles(x) for x in nonhit_smiles]
    nonhit_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048) if x is not None else None for x in nonhit_mols]

    ecoli_train_neg_df = ecoli_train_df[ecoli_train_df[ecoli_prediction_target_name] < 0.7]
    ref_smiles = ecoli_train_neg_df['SMILES']
    ref_mols = [Chem.MolFromSmiles(x) for x in ref_smiles]
    ref_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048) if x is not None else None for x in ref_mols]

    max_sim_list = []
    for nonhit_fp in tqdm(nonhit_fps):
        sim_list = []
        for ref_fp in ref_fps:
            if (nonhit_fp is not None) and (ref_fp is not None):
                sim_list.append(DataStructs.TanimotoSimilarity(nonhit_fp, ref_fp))
        max_sim_list.append(max(sim_list))

    ecoli_nonhit_df['nonhit_novelty'] = max_sim_list
    ecoli_nonhit_df['nonhit_novelty'] = 1 - ecoli_nonhit_df['nonhit_novelty']

    ecoli_prediction_df = ecoli_prediction_df.merge(ecoli_hit_df,
                                              on = ['Plate','Library','SMILES',ecoli_prediction_target_name,'Ecoli_Inhibition_%ctl_dirichlet_uncal_uncertainty','potential_ecoli_inhibitory'],
                                              how = 'left')

    ecoli_prediction_df = ecoli_prediction_df.merge(ecoli_nonhit_df,
                                              on = ['Plate','Library','SMILES',ecoli_prediction_target_name,'Ecoli_Inhibition_%ctl_dirichlet_uncal_uncertainty','potential_ecoli_inhibitory'],
                                              how = 'left')

    ecoli_prediction_df['hit_novelty'] = ecoli_prediction_df['hit_novelty'].fillna(0) # Set non-hits as Zero
    ecoli_prediction_df['nonhit_novelty'] = ecoli_prediction_df['nonhit_novelty'].fillna(0)
    ecoli_prediction_df['rationales'] = ecoli_prediction_df['rationales'].apply(lambda x: x if isinstance(x, list) else [])

    # Drop Duplicate Molecules
    ecoli_prediction_df = ecoli_prediction_df.drop_duplicates(subset = ['Plate','Library','SMILES'])

    ## E.coli - Rank Plates Based on Molecules novelty score
    ecoli_prediction_df['novel_hits'] = np.where(ecoli_prediction_df['hit_novelty'] >= novelty_threshold, 1, 0)
    ecoli_rank_df = ecoli_prediction_df[['Plate','Library','potential_ecoli_inhibitory','rationales','novel_hits','nonhit_novelty']].groupby(['Plate','Library'], as_index=False)[['potential_ecoli_inhibitory','rationales','novel_hits','nonhit_novelty']].agg({
        'rationales': unique_rationales,
        'nonhit_novelty': 'mean',
        'potential_ecoli_inhibitory': 'sum',
        'novel_hits': 'sum'
    })
    ecoli_rank_df['rationales_mol'] = ecoli_rank_df['rationales'].apply(rationale_utils.smiles_to_mol_list)

    ecoli_rank_df['filtered_rationales_mol'] = ecoli_rank_df['rationales_mol'].apply(rationale_utils.filter_mols)
    ecoli_rank_df['rationales'] = ecoli_rank_df['filtered_rationales_mol'].apply(rationale_utils.mols_to_smiles_list)
    ecoli_rank_df['num_unique_rationales'] = ecoli_rank_df['rationales'].apply(len)

    # Inplate Similarity
    plate_cluster_num = []
    for plate in ecoli_rank_df['Plate']:
        plate_df = ecoli_hit_df[ecoli_hit_df['Plate'] == plate]
        if len(plate_df) == 0:
            plate_cluster_num.append(0)
            continue
        plate_mol = [Chem.MolFromSmiles(x) for x in list(plate_df['SMILES'])]
        plate_fps = [AllChem.GetMorganFingerprintAsBitVect(x,2,1024) for x in plate_mol if x is not None]

        dists = []
        nfps = len(plate_fps)

        for i in range(1,nfps):
                sims = DataStructs.BulkTanimotoSimilarity(plate_fps[i],plate_fps[:i])
                dists.extend([1-x for x in sims])

        cs = Butina.ClusterData(dists,nfps,0.5,isDistData=True)
        plate_cluster_num.append(len(cs))
    
    ecoli_rank_df['num_clusters'] = plate_cluster_num    

    # Rank columns
    selection_weights = {'nonhit_novelty': 1,
                         'potential_ecoli_inhibitory': 1,
                         'novel_hits': 2,
                         'num_unique_rationales': 1,
                         'num_clusters': 1}
    
    ecoli_rank_df['overall_rank'] = 0
    for column in ['nonhit_novelty','potential_ecoli_inhibitory','novel_hits','num_unique_rationales','num_clusters']:
        ecoli_rank_df[f'rank_{column}'] = ecoli_rank_df[column].rank(ascending=False)
        ecoli_rank_df['overall_rank'] += selection_weights[column] * ecoli_rank_df[f'rank_{column}']
    
    ecoli_rank_df['overall_rank'] = ecoli_rank_df['overall_rank'].rank(ascending=True)
    ecoli_rank_df = ecoli_rank_df.sort_values(by = ['overall_rank'], ascending=True)

    ## E.coli - Save Rank File
    ecoli_rank_df = ecoli_rank_df.drop(["rationales","filtered_rationales_mol", "rationales_mol"], axis = 1)
    ecoli_rank_df.to_csv(f'/demo_data/scores.csv',
                         index = False)

    return ecoli_rank_df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    # E.coli Dataframes
    ecoli_train_dir = f'/demo_data/ecoli.csv'
    ecoli_prediction_dir = f'/demo_data/ecoli_pcontrol.csv'
    ecoli_rationale_dir = f'/demo_data/ecoli_rationale.csv'

    # Load 
    ecoli_train_df = pd.read_csv(ecoli_train_dir)
    ecoli_prediction_df = pd.read_csv(ecoli_prediction_dir)
    ecoli_rationale_df = pd.read_csv(ecoli_rationale_dir)

    bb_rank_df, ecoli_rank_df = plate_scoring(
                                              ecoli_prediction_df=ecoli_prediction_df,
                                              ecoli_prediction_target_name='Ecoli_Inhibition_%ctl',
                                              ecoli_train_df = ecoli_train_df,
                                              novelty_weight = 1,
                                              novelty_threshold=0.5,
                                              ecoli_rationale_df = ecoli_rationale_df)
