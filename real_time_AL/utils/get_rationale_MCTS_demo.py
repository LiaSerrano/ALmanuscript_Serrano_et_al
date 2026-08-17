#!/usr/bin/env python3

# Standard library imports
import os
import sys
import math
import time
import glob
import random
import pickle
import logging
import argparse

import os
import sys
import math
import time
import glob
import random
import pickle
import logging
import argparse

import numpy as np

# Compatibility shim for NumPy 2 pickle -> NumPy 1 load
try:
    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric
    import numpy.core._multiarray_umath

    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
    sys.modules.setdefault("numpy._core._multiarray_umath", numpy.core._multiarray_umath)
except Exception:
    pass



from typing import Callable, Dict, List, Set, Tuple, Union, Iterable, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from multiprocessing import Pool, cpu_count
from contextlib import contextmanager

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# RDKit imports
import rdkit
from rdkit import Chem

# Chemprop imports
import chemprop
import chemprop.data as data


# ============================================================
# MCTS NODE
# ============================================================

@dataclass
class MCTSNode:
    """Represents a node in a Monte Carlo Tree Search."""

    smiles: str
    atoms: Iterable[int]
    W: float = 0
    N: int = 0
    P: float = 0
    children: List["MCTSNode"] = field(default_factory=list)

    def __post_init__(self):
        self.atoms = set(self.atoms)

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0

    def U(self, n: int, c_puct: float = 10.0) -> float:
        return c_puct * self.P * math.sqrt(n) / (1 + self.N)


# ============================================================
# FEATURE DICTIONARY HELPERS
# ============================================================

def canonicalize_smiles(smi: str) -> str:
    """
    Canonicalize SMILES so feature dictionary lookup is more robust.
    """
    if smi is None:
        return smi

    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return str(smi)

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def coerce_feature_vector(value: Any, feature_key: Optional[str] = None) -> np.ndarray:
    """
    Converts one feature dictionary value into a flat numpy vector.

    Supported formats:
        features_dict[smiles] = list / np.array
        features_dict[smiles] = {"some_key": list / np.array}
    """
    if isinstance(value, dict):
        if feature_key is None:
            raise ValueError(
                "Feature value is itself a dictionary. "
                f"Available keys: {list(value.keys())}. "
                "Pass --feature_key YOUR_KEY."
            )

        if feature_key not in value:
            raise KeyError(
                f"feature_key={feature_key!r} not found in inner feature dictionary. "
                f"Available keys: {list(value.keys())}"
            )

        value = value[feature_key]

    arr = np.asarray(value, dtype=np.float32).reshape(-1)

    if arr.size == 0:
        raise ValueError("Feature vector is empty.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("Feature vector contains NaN or inf.")

    return arr


def load_features_dict(
    features_dict_path: str,
    feature_key: Optional[str] = None,
    canonicalize_keys: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Loads a pickle feature dictionary.

    Expected:
        {SMILES: feature_vector}

    Also supports:
        {SMILES: {"some_key": feature_vector}}
    """
    with open(features_dict_path, "rb") as f:
        raw = pickle.load(f)

    if not isinstance(raw, dict):
        raise TypeError(
            f"Expected a dictionary from {features_dict_path}, got {type(raw)}"
        )

    features_dict: Dict[str, np.ndarray] = {}

    for smi, value in raw.items():
        exact_key = str(smi)
        feat = coerce_feature_vector(value, feature_key=feature_key)

        # Store exact key.
        features_dict[exact_key] = feat

        # Also store canonicalized key.
        if canonicalize_keys:
            can_key = canonicalize_smiles(exact_key)
            features_dict[can_key] = feat

    dims = sorted(set(len(v) for v in features_dict.values()))

    print("\nLoaded feature dictionary")
    print("Path:", features_dict_path)
    print("Number of usable keys:", len(features_dict))
    print("Feature dimensions found:", dims)

    if len(dims) != 1:
        raise ValueError(
            f"Feature vectors do not all have the same length: {dims}"
        )

    return features_dict


def get_feature_for_smiles(
    smi: str,
    features_dict: Dict[str, np.ndarray],
    parent_features: Optional[np.ndarray] = None,
    missing: str = "parent",
) -> np.ndarray:
    """
    Get external features for a SMILES.

    missing:
        parent -> if subgraph is missing, use parent molecule's features
        zero   -> if subgraph is missing, use zero vector
        error  -> if missing, raise error
    """
    smi = str(smi)
    can = canonicalize_smiles(smi)

    if smi in features_dict:
        return features_dict[smi]

    if can in features_dict:
        return features_dict[can]

    if missing == "parent" and parent_features is not None:
        return parent_features

    if missing == "zero":
        dim = len(next(iter(features_dict.values())))
        return np.zeros(dim, dtype=np.float32)

    raise KeyError(
        f"No feature vector found for SMILES:\n"
        f"  original:   {smi}\n"
        f"  canonical:  {can}\n"
        f"missing mode was {missing!r}"
    )


# ============================================================
# MODEL HELPERS
# ============================================================

def load_models(model_paths: List[str], device: str = "cuda") -> List:
    """
    Load Chemprop models from specified paths.
    """
    models = []

    for path in model_paths:
        print(f"Loading model: {path}")
        model = chemprop.utils.load_checkpoint(
            path=path,
            device=torch.device(device),
        )
        model.eval()
        models.append(model)

    return models


def get_first_readout_linear(model) -> Optional[torch.nn.Linear]:
    """
    Finds the first Linear layer in model.readout if possible.
    Useful for checking expected FFN input dimension.
    """
    if not hasattr(model, "readout"):
        return None

    for module in model.readout.modules():
        if isinstance(module, torch.nn.Linear):
            return module

    return None


def print_model_feature_debug(models: List, feature_dim: int):
    """
    Print model input expectations. This is helpful for the 500 vs 812 issue.
    """
    if len(models) == 0:
        return

    model = models[0]
    args = getattr(model, "args", None)

    hidden_size = getattr(args, "hidden_size", None)
    features_size = getattr(args, "features_size", None)
    features_generator = getattr(args, "features_generator", None)
    use_input_features = getattr(args, "use_input_features", None)

    first_linear = get_first_readout_linear(model)

    print("\nModel feature debug")
    print("External feature dim supplied:", feature_dim)
    print("model.args.hidden_size:", hidden_size)
    print("model.args.features_size:", features_size)
    print("model.args.features_generator:", features_generator)
    print("model.args.use_input_features:", use_input_features)

    if first_linear is not None:
        print("First readout Linear in_features:", first_linear.in_features)
        print("First readout Linear out_features:", first_linear.out_features)

        if hidden_size is not None:
            inferred_external_dim = first_linear.in_features - int(hidden_size)
            print("Inferred external feature dim expected:", inferred_external_dim)

            if inferred_external_dim != feature_dim:
                print(
                    "\nWARNING: supplied feature dim does not match inferred expected dim.\n"
                    f"  supplied: {feature_dim}\n"
                    f"  expected: {inferred_external_dim}\n"
                    "This may still be okay if hidden_size is not the correct base dimension, "
                    "but for your 500 vs 812 error, you likely want feature_dim = 312.\n"
                )


# ============================================================
# SUBGRAPH EXTRACTION
# ============================================================

def extract_subgraph_from_mol(
    mol: Chem.Mol,
    selected_atoms: Set[int],
) -> Tuple[Chem.Mol, List[int]]:
    """
    Extracts a subgraph from an RDKit molecule given selected atom indices.
    """
    selected_atoms = set(selected_atoms)
    roots = []

    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)

        aroma_bonds = [
            bond
            for bond in atom.GetBonds()
            if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC
        ]

        aroma_bonds = [
            bond
            for bond in aroma_bonds
            if bond.GetBeginAtom().GetIdx() in selected_atoms
            and bond.GetEndAtom().GetIdx() in selected_atoms
        ]

        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [
        atom.GetIdx()
        for atom in new_mol.GetAtoms()
        if atom.GetIdx() not in selected_atoms
    ]

    remove_atoms = sorted(remove_atoms, reverse=True)

    for atom_idx in remove_atoms:
        new_mol.RemoveAtom(atom_idx)

    return new_mol.GetMol(), roots


def extract_subgraph(smiles: str, selected_atoms: Set[int]) -> Tuple[Optional[str], Optional[List[int]]]:
    """
    Extracts a subgraph from a SMILES given selected atom indices.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None

    # Try with kekulization first.
    try:
        mol_kek = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol_kek)
        subgraph, roots = extract_subgraph_from_mol(mol_kek, selected_atoms)

        try:
            subgraph_smi = Chem.MolToSmiles(subgraph, kekuleSmiles=True)
            subgraph_mol = Chem.MolFromSmiles(subgraph_smi)
        except Exception:
            subgraph_mol = None

        mol_dekek = Chem.MolFromSmiles(smiles)

        if subgraph_mol is not None and mol_dekek.HasSubstructMatch(subgraph_mol):
            return Chem.MolToSmiles(subgraph_mol), roots

    except Exception:
        pass

    # Fallback without kekulization.
    try:
        subgraph, roots = extract_subgraph_from_mol(mol, selected_atoms)
        subgraph_smi = Chem.MolToSmiles(subgraph)
        subgraph_mol = Chem.MolFromSmiles(subgraph_smi)

        if subgraph_mol is not None:
            return Chem.MolToSmiles(subgraph_mol), roots

    except Exception:
        pass

    return None, None


# ============================================================
# MCTS FUNCTIONS
# ============================================================

def find_clusters(mol: Chem.Mol) -> Tuple[List[Tuple[int, ...]], List[List[int]]]:
    """
    Find atom clusters for MCTS.
    Non-ring bonds are clusters, and rings are clusters.
    """
    n_atoms = mol.GetNumAtoms()

    if n_atoms == 1:
        return [(0,)], [[0]]

    clusters = []

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()

        if not bond.IsInRing():
            clusters.append((a1, a2))

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    atom_cls = [[] for _ in range(n_atoms)]

    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls


def mcts_rollout(
    node: MCTSNode,
    state_map: Dict[str, MCTSNode],
    orig_smiles: str,
    clusters: List[Set[int]],
    atom_cls: List[Set[int]],
    nei_cls: List[Set[int]],
    scoring_function: Callable[[List[str]], float],
    min_atoms: int,
    c_puct: float = 10.0,
) -> float:
    """
    A Monte Carlo Tree Search rollout from a given MCTSNode.
    """
    cur_atoms = node.atoms

    if len(cur_atoms) <= min_atoms:
        return node.P

    if len(node.children) == 0:
        cur_cls = set([i for i, x in enumerate(clusters) if x <= cur_atoms])

        for i in cur_cls:
            leaf_atoms = [
                a
                for a in clusters[i]
                if len(atom_cls[a] & cur_cls) == 1
            ]

            if len(nei_cls[i] & cur_cls) == 1 or (
                len(clusters[i]) == 2 and len(leaf_atoms) == 1
            ):
                new_atoms = cur_atoms - set(leaf_atoms)
                new_smiles, _ = extract_subgraph(orig_smiles, new_atoms)

                if not new_smiles:
                    continue

                if new_smiles in state_map:
                    new_node = state_map[new_smiles]
                else:
                    new_node = MCTSNode(new_smiles, new_atoms)

                node.children.append(new_node)

        state_map[node.smiles] = node

        if len(node.children) == 0:
            return node.P

        scores = []

        for child in node.children:
            score = scoring_function([child.smiles])
            scores.append(float(score))

        for child, score in zip(node.children, scores):
            child.P = float(score)

    sum_count = sum(c.N for c in node.children)

    selected_node = max(
        node.children,
        key=lambda x: x.Q() + x.U(sum_count, c_puct=c_puct),
    )

    v = mcts_rollout(
        selected_node,
        state_map,
        orig_smiles,
        clusters,
        atom_cls,
        nei_cls,
        scoring_function,
        min_atoms=min_atoms,
        c_puct=c_puct,
    )

    selected_node.W += v
    selected_node.N += 1

    return float(v)


def mcts(
    smiles: str,
    scoring_function: Callable[[List[str]], float],
    n_rollout: int,
    max_atoms: int,
    prop_delta: float,
    min_atoms: int,
    c_puct: float = 10.0,
) -> List[MCTSNode]:
    """
    Runs Monte Carlo Tree Search rationale extraction.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return []

    clusters, atom_cls = find_clusters(mol)

    nei_cls = [0] * len(clusters)

    for i, cls in enumerate(clusters):
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - {i}
        clusters[i] = set(list(cls))

    for a in range(len(atom_cls)):
        atom_cls[a] = set(atom_cls[a])

    root = MCTSNode(smiles, set(range(mol.GetNumAtoms())))
    state_map = {smiles: root}

    for _ in range(n_rollout):
        mcts_rollout(
            root,
            state_map,
            smiles,
            clusters,
            atom_cls,
            nei_cls,
            scoring_function,
            min_atoms=min_atoms,
            c_puct=c_puct,
        )

    rationales = [
        node
        for _, node in state_map.items()
        if len(node.atoms) <= max_atoms and node.P >= prop_delta
    ]

    return rationales


# ============================================================
# CHEMPROP PREDICTION WITH EXTERNAL FEATURES
# ============================================================

def make_prediction(
    models: List,
    smiles: List[str],
    features_dict: Dict[str, np.ndarray],
    parent_features: Optional[np.ndarray] = None,
    missing_features: str = "parent",
) -> np.ndarray:
    """
    Makes predictions on a list of SMILES using external features.

    For full molecules:
        features come from features_dict.

    For MCTS subgraphs:
        if subgraph is not in features_dict, default behavior is to use
        the parent molecule feature vector.
    """
    test_data = []

    for smi in smiles:
        feat = get_feature_for_smiles(
            smi=smi,
            features_dict=features_dict,
            parent_features=parent_features,
            missing=missing_features,
        )

        datapoint = data.MoleculeDatapoint(
            smiles=[smi],
            features=feat,
        )

        test_data.append(datapoint)

    test_dset = data.MoleculeDataset(test_data)

    test_loader = data.MoleculeDataLoader(
        test_dset,
        batch_size=len(test_data),
        num_workers=0,
        shuffle=False,
    )

    with torch.inference_mode():
        all_model_preds = []

        for model in models:
            model.eval()

            try:
                preds = chemprop.train.predict(model, test_loader)
                preds = np.asarray(preds, dtype=float)
                all_model_preds.append(preds)

            except Exception as e:
                print("\nPrediction failed.")
                print("First few SMILES:", smiles[:3])
                print("Feature dim supplied:", len(test_data[0].features))
                print("Original error:", repr(e))
                raise

        avg_preds = np.mean(all_model_preds, axis=0)

    return avg_preds


def predict_one_scalar(
    models: List,
    smiles: str,
    features_dict: Dict[str, np.ndarray],
    parent_features: Optional[np.ndarray] = None,
    missing_features: str = "parent",
) -> float:
    """
    Convenience wrapper returning a single scalar prediction.
    """
    pred = make_prediction(
        models=models,
        smiles=[smiles],
        features_dict=features_dict,
        parent_features=parent_features,
        missing_features=missing_features,
    )

    return float(np.ravel(pred)[0])


# ============================================================
# CHUNK PROCESSING
# ============================================================

def process_chunk(args):
    """
    Process a chunk of SMILES strings with MCTS analysis.
    """
    chunk, model_paths, start_idx, kwargs = args

    models = load_models(model_paths, kwargs["device"])

    features_dict = load_features_dict(
        features_dict_path=kwargs["features_dict_path"],
        feature_key=kwargs.get("feature_key", None),
        canonicalize_keys=True,
    )

    feature_dim = len(next(iter(features_dict.values())))
    print_model_feature_debug(models, feature_dim=feature_dim)

    results = {
        "smiles": [],
        kwargs["property_name"]: [],
    }

    for i in range(kwargs["num_rationales"]):
        results[f"rationale_{i}"] = []
        results[f"rationale_{i}_score"] = []

    for idx, smiles in enumerate(tqdm(chunk)):
        global_idx = start_idx + idx

        if global_idx % kwargs["save_interval"] == 0:
            print(f"Processing molecule {global_idx}")

        try:
            parent_features = get_feature_for_smiles(
                smi=smiles,
                features_dict=features_dict,
                parent_features=None,
                missing="error",
            )

            def scoring_function(query_smiles: List[str]) -> float:
                """
                MCTS scoring function.

                For subgraphs missing from features_dict, this uses the
                parent molecule's external feature vector by default.
                """
                pred = make_prediction(
                    models=models,
                    smiles=query_smiles,
                    features_dict=features_dict,
                    parent_features=parent_features,
                    missing_features=kwargs.get("missing_features", "parent"),
                )

                return float(np.ravel(pred)[0])

            score = float(scoring_function([smiles]))

            if score > kwargs["prop_delta"]:
                rationales = mcts(
                    smiles=smiles,
                    scoring_function=scoring_function,
                    n_rollout=kwargs["rollout"],
                    max_atoms=kwargs["max_atoms"],
                    prop_delta=kwargs["prop_delta"],
                    min_atoms=kwargs["min_atoms"],
                    c_puct=kwargs["c_puct"],
                )
            else:
                rationales = []

        except Exception as e:
            print(f"\nFailed on molecule index={global_idx}, smiles={smiles}")
            print("Error:", repr(e))

            score = np.nan
            rationales = []

        results["smiles"].append(smiles)
        results[kwargs["property_name"]].append(score)

        if len(rationales) == 0:
            for i in range(kwargs["num_rationales"]):
                results[f"rationale_{i}"].append(None)
                results[f"rationale_{i}_score"].append(None)

        else:
            min_size = min(len(x.atoms) for x in rationales)
            min_rationales = [x for x in rationales if len(x.atoms) == min_size]
            rats = sorted(min_rationales, key=lambda x: x.P, reverse=True)

            for i in range(kwargs["num_rationales"]):
                if i < len(rats):
                    results[f"rationale_{i}"].append(rats[i].smiles)
                    results[f"rationale_{i}_score"].append(rats[i].P)
                else:
                    results[f"rationale_{i}"].append(None)
                    results[f"rationale_{i}_score"].append(None)

    return results


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def run_mcts_analysis(
    model_paths: List[str],
    data_path: str,
    output_dir: str,
    output_file: str,
    features_dict_path: str,
    feature_key: Optional[str] = None,
    missing_features: str = "parent",
    smiles_column: str = "SMILES",
    device: str = "cuda",
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    property_name: str = "inhibition",
    rollout: int = 1,
    c_puct: float = 10.0,
    max_atoms: int = 20,
    min_atoms: int = 8,
    prop_delta: float = 0.7,
    num_rationales: int = 5,
    save_interval: int = 5,
    n_processes: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run parallel MCTS analysis on molecules.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "intermediate"), exist_ok=True)

    dataset = pd.read_csv(data_path)

    if smiles_column not in dataset.columns:
        raise KeyError(
            f"smiles_column={smiles_column!r} not found in input CSV. "
            f"Available columns: {list(dataset.columns)}"
        )

    if property_name not in dataset.columns:
        raise KeyError(
            f"property_name={property_name!r} not found in input CSV. "
            f"Available columns: {list(dataset.columns)}"
        )

    # Keep the behavior of your original script:
    # only run MCTS on molecules with property >= prop_delta.
    dataset = dataset[dataset[property_name] != "Invalid SMILES"].copy()
    dataset[property_name] = pd.to_numeric(dataset[property_name], errors="coerce")
    dataset = dataset[dataset[property_name] >= prop_delta].reset_index(drop=True)

    if end_idx is None:
        end_idx = len(dataset)

    dataset = dataset.iloc[start_idx:end_idx].reset_index(drop=True)
    all_smiles = list(dataset[smiles_column])

    print("\nInput summary")
    print("Input CSV:", data_path)
    print("Rows after filtering:", len(dataset))
    print("SMILES column:", smiles_column)
    print("Property column:", property_name)
    print("prop_delta:", prop_delta)

    if len(all_smiles) == 0:
        print("No molecules to process after filtering.")
        final_df = pd.DataFrame({"smiles": [], property_name: []})
        final_df.to_csv(os.path.join(output_dir, output_file), index=False)
        return final_df

    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    n_processes = max(1, int(n_processes))
    n_processes = min(n_processes, len(all_smiles))

    print("n_processes:", n_processes)

    chunk_size = max(1, math.ceil(len(all_smiles) / n_processes))

    chunks = [
        all_smiles[i:i + chunk_size]
        for i in range(0, len(all_smiles), chunk_size)
    ]

    kwargs = {
        "device": device,
        "property_name": property_name,
        "rollout": rollout,
        "c_puct": c_puct,
        "max_atoms": max_atoms,
        "min_atoms": min_atoms,
        "prop_delta": prop_delta,
        "num_rationales": num_rationales,
        "save_interval": save_interval,
        "features_dict_path": features_dict_path,
        "feature_key": feature_key,
        "missing_features": missing_features,
    }

    process_args = [
        (chunk, model_paths, start_idx + idx * chunk_size, kwargs)
        for idx, chunk in enumerate(chunks)
    ]

    if n_processes == 1:
        chunk_results = [process_chunk(process_args[0])]
    else:
        with Pool(processes=n_processes) as pool:
            chunk_results = pool.map(process_chunk, process_args)

    combined_results = {
        "smiles": [],
        property_name: [],
    }

    for i in range(num_rationales):
        combined_results[f"rationale_{i}"] = []
        combined_results[f"rationale_{i}_score"] = []

    for result in chunk_results:
        for key in combined_results:
            combined_results[key].extend(result[key])

    final_df = pd.DataFrame(combined_results)

    output_path = os.path.join(output_dir, output_file)
    final_df.to_csv(output_path, index=False)

    print("\nSaved final results:")
    print(output_path)

    return final_df


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_id", type=str, help="Batch ID Number") 

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input CSV.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output CSV filename.",
    )

    parser.add_argument(
        "--property_name",
        type=str,
        required=True,
        help="Column to score/filter on.",
    )

    parser.add_argument(
        "--features_dict_path",
        type=str,
        required=True,
        help="Pickle file containing {SMILES: feature_vector}.",
    )

    parser.add_argument(
        "--feature_key",
        type=str,
        default=None,
        help=(
            "Use this if features_dict[SMILES] is itself a dictionary, "
            "e.g. features_dict[SMILES]['my_feature_key']."
        ),
    )

    parser.add_argument(
        "--missing_features",
        type=str,
        default="parent",
        choices=["parent", "zero", "error"],
        help=(
            "What to do if an MCTS subgraph is not in the feature dictionary. "
            "'parent' reuses the full molecule's feature vector. "
            "'zero' uses a zero vector. "
            "'error' raises an error."
        ),
    )

    parser.add_argument(
        "--smiles_column",
        type=str,
        default="SMILES",
        help="Name of the SMILES column in the input CSV.",
    )

    parser.add_argument(
        "--n_processes",
        type=int,
        default=None,
        help="Number of processes to use. Use 1 for debugging.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run Chemprop prediction on.",
    )

    parser.add_argument(
        "--prop_delta",
        type=float,
        default=0.7,
        help="Minimum score threshold.",
    )

    parser.add_argument(
        "--rollout",
        type=int,
        default=1,
        help="Number of MCTS rollouts.",
    )

    parser.add_argument(
        "--max_atoms",
        type=int,
        default=20,
        help="Maximum atoms allowed in rationale.",
    )

    parser.add_argument(
        "--min_atoms",
        type=int,
        default=8,
        help="Minimum atoms allowed in rationale.",
    )

    parser.add_argument(
        "--num_rationales",
        type=int,
        default=5,
        help="Number of rationales to keep per molecule.",
    )


    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to one or more Chemprop model checkpoint files.",
    )

    args = parser.parse_args()

    model_paths = args.model_paths

    results = run_mcts_analysis(
        model_paths=model_paths,
        data_path=args.data_path,
        output_dir=args.output_dir,
        output_file=args.output_file,
        features_dict_path=args.features_dict_path,
        feature_key=args.feature_key,
        missing_features=args.missing_features,
        smiles_column=args.smiles_column,
        device=args.device,
        property_name=args.property_name,
        n_processes=args.n_processes,
        prop_delta=args.prop_delta,
        rollout=args.rollout,
        max_atoms=args.max_atoms,
        min_atoms=args.min_atoms,
        num_rationales=args.num_rationales,
    )