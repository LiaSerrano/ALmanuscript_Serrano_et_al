import pandas as pd
import numpy as np
import argparse

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


# Save
remaining_molecules[
    ["Plate", "Library", "SMILES"]
].to_csv(
    "demo_data/remaining_unscreened_molecules.csv",
    index=False
)