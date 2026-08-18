import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--previously_screened",
    type=str,
    required=True,
    help="Path to CSV containing previously screened molecules"
)

parser.add_argument(
    "--remaining_pool",
    type=str,
    required=True,
    help="Path to CSV containing the remaining molecule pool"
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory where output files will be saved"
)

args = parser.parse_args()


# Screened molecules
previously_screened_molecules = pd.read_csv(args.previously_screened)

screened_plates = list(
    previously_screened_molecules["Plate"].unique()
)


# Remaining pool
remaining_pool = pd.read_csv(args.remaining_pool)

# Remove plates that have already been screened
remaining_molecules = remaining_pool[
    ~remaining_pool["Plate"].isin(screened_plates)
].copy()

# Remove missing SMILES
remaining_molecules = remaining_molecules[
    ~pd.isna(remaining_molecules["SMILES"])
].copy()


# Make output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

output_path = os.path.join(
    args.output_dir,
    "remaining_unscreened_molecules.csv"
)

# Save
remaining_molecules[
    ["Plate", "SMILES"]
].to_csv(
    output_path,
    index=False
)

print("Saved:", output_path)