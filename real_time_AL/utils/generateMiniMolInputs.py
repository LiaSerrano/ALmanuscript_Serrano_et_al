#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 19:50:58 2025

@author: liaserrano
"""

import pickle
import numpy as np
import rdkit
from rdkit import Chem
import pandas as pd
import numpy as np
import argparse
import re
from rdkit.Chem.MolStandardize import rdMolStandardize

parser = argparse.ArgumentParser()

parser.add_argument(
    "--features_path",
    type=str,
    required=True,
    help="Path to the molecular feature file"
)

parser.add_argument(
    "--remaining_unscreened_path",
    type=str,
    required=True,
    help="Path to remaining_unscreened_molecules.csv"
)

parser.add_argument(
    "--training_ecoli_path",
    type=str,
    required=True,
    help="Path to E. coli training CSV"
)

parser.add_argument(
    "--training_output_csv",
    type=str,
    required=True,
    help="Output path for filtered E. coli CSV"
)

parser.add_argument(
    "--training_output_npz",
    type=str,
    required=True,
    help="Output path for filtered E. coli feature NPZ"
)

parser.add_argument(
    "--remaining_output_csv",
    type=str,
    required=True,
    help="Output path for filtered remaining-pool CSV"
)

parser.add_argument(
    "--remaining_output_npz",
    type=str,
    required=True,
    help="Output path for filtered remaining-pool feature NPZ"
)

args = parser.parse_args()

import pickle
import sys
import numpy as np

def load_numpy_pickle_compat(path):
    """
    Load pickle files saved with NumPy 2.x in environments that may use NumPy 1.x.
    Handles numpy._core -> numpy.core module path changes.
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)

    except ModuleNotFoundError as e:
        if e.name == "numpy._core":
            import numpy.core
            import numpy.core.multiarray
            import numpy.core.numeric

            sys.modules["numpy._core"] = numpy.core
            sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
            sys.modules["numpy._core.numeric"] = numpy.core.numeric

            with open(path, "rb") as f:
                return pickle.load(f)

        raise

def is_valid_smiles(smile):
    mol = Chem.MolFromSmiles(smile)
    return mol is not None


loaded_data = load_numpy_pickle_compat(args.features_path)
remaining_unscreened = pd.read_csv(args.remaining_unscreened_path)
training_ecoli = pd.read_csv(args.training_ecoli_path)

FORMULA_STYLE_SALTS = {
    "HCl", "2HCl", "3HCl", "4HCl",
    "HBr", "2HBr", "3HBr", "4HBr",
    "HI", "2HI", "3HI", "4HI",
    "H2O", "2H2O", "3H2O",
    "HOOCCOOH", "2HOOCCOOH", "3HOOCCOOH",
    "HOOC-COOH",
    "TFA", "2TFA",
    "AcOH", "HOAc",
    "C6H2(NO2)3OH",
    "[Cl-]", "[Br-]", "[I-]",
    "[Na+]", "[K+]", "[Li+]",
    "[*]",
}


def is_formula_style_salt(fragment):
    frag = str(fragment).strip()

    if frag in FORMULA_STYLE_SALTS:
        return True

    if re.fullmatch(r"\d*HCl", frag):
        return True
    if re.fullmatch(r"\d*HBr", frag):
        return True
    if re.fullmatch(r"\d*HI", frag):
        return True
    if re.fullmatch(r"\d*HOOCCOOH", frag):
        return True

    return False


def remove_salts_from_smiles(smiles):
    """
    Removes formula-style salts first, then RDKit-parses and returns
    RDKit MolToSmiles(parent). This matches the version that worked locally.
    """

    if smiles is None:
        return None

    smiles = str(smiles).strip()

    if smiles == "" or smiles.lower() == "nan":
        return None

    raw_fragments = [
        frag.strip()
        for frag in smiles.split(".")
        if frag.strip()
    ]

    kept_fragments = [
        frag
        for frag in raw_fragments
        if not is_formula_style_salt(frag)
    ]

    if len(kept_fragments) == 0:
        return None

    precleaned = ".".join(kept_fragments)

    mol = Chem.MolFromSmiles(precleaned)

    if mol is None:
        return None

    chooser = rdMolStandardize.LargestFragmentChooser()
    parent = chooser.choose(mol)

    if parent is None:
        return None

    return Chem.MolToSmiles(parent)


def get_label_column(input_data):
    if "Bb_Inhibition_%ctl" in input_data.columns:
        return "Bb_Inhibition_%ctl"
    elif "Ecoli_Inhibition_%ctl" in input_data.columns:
        return "Ecoli_Inhibition_%ctl"
    elif "Inhibition" in input_data.columns:
        return "Inhibition"
    else:
        return None


def get_feature_key(original_smile, feature_dictionary):
    """
    Try desalted string first, then original string.
    No canonicalization.
    """

    if original_smile is None:
        return None

    original_smile = str(original_smile).strip()
    cleaned_smile = remove_salts_from_smiles(original_smile)

    possible_keys = [
        cleaned_smile,
        original_smile,
    ]

    for key in possible_keys:
        if key is not None and key in feature_dictionary:
            return key

    return None


def Produce_cleaned_inputs(
    input_data,
    output_filepath,
    output_npz_filepath,
    feature_dictionary
):

    array_embeds = []
    kept_rows = []
    saved_smiles = []
    missing = []

    for _, row in input_data.iterrows():

        a = str(row['SMILES']).strip()

        try:
            # First try original SMILES
            t = feature_dictionary[a]
            smile_to_save = a

        except Exception:
            # Then try salt-cleaned / RDKit MolToSmiles version
            cleaned = remove_salts_from_smiles(a)

            try:
                t = feature_dictionary[cleaned]
                smile_to_save = cleaned

            except Exception:
                missing.append((a, cleaned))
                continue

        row_out = row.copy()
        row_out['SMILES'] = smile_to_save

        if 'valid_smiles' not in row_out.index:
            row_out['valid_smiles'] = is_valid_smiles(smile_to_save)

        kept_rows.append(row_out)
        array_embeds.append(t)
        saved_smiles.append(smile_to_save)

    filtered_smis = pd.DataFrame(kept_rows)

    filtered_smis.to_csv(
        output_filepath,
        index=False
    )

    np.savez(
        output_npz_filepath,
        features=np.array(array_embeds),
        smiles=np.array(saved_smiles)
    )

    print("CSV output:", output_filepath)
    print("NPZ output:", output_npz_filepath)
    print("input rows:", len(input_data))
    print("kept rows:", len(filtered_smis))
    print("missing rows:", len(missing))

    if len(missing) > 0:
        print("first missing examples:")
        for original, cleaned in missing[:10]:
            print("original:", original)
            print("cleaned :", cleaned)
            print()


Produce_cleaned_inputs(
    training_ecoli,
    args.training_output_csv,
    args.training_output_npz,
    loaded_data
)


Produce_cleaned_inputs(
    remaining_unscreened,
    args.remaining_output_csv,
    args.remaining_output_npz,
    loaded_data
)







