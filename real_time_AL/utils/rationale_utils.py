from rdkit import Chem
import pandas as pd

from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit import DataStructs
from rdkit.Chem import AllChem
import rdkit
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw

def mols_to_smiles_list(mol_list):
    
    return [Chem.MolToSmiles(mol) for mol in mol_list if mol is not None]

def smiles_to_mol_list(smiles_list):
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list if smiles]
def remove_redundant_molecules(molecule_list):
    # Create a set to keep unique canonical SMILES
    unique_molecules = set()
    unique_molecule_list = []
    
    for molecule in molecule_list:
        #molecule = Chem.AddHs(molecule)
        # Get the canonical SMILES representation of the molecule
        canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
        
        # Add to set if not already present
        if canonical_smiles not in unique_molecules:
            unique_molecules.add(canonical_smiles)
            unique_molecule_list.append(molecule)
    
    return unique_molecule_list


def has_complete_ring(molecules_list):
    # Convert SMILES to RDKit molecule
    molecules_with_rings = []
    for c in molecules_list:
        ring_info= c.GetRingInfo() 
        if ring_info.NumRings() > 0:
            molecules_with_rings.append(c)

    
    # Check if molecule has at least one ring
    
    # Return True if the molecule has a complete ring, otherwise False
    return molecules_with_rings


def get_unique_canonical_molecules(mol_list):
    """
    Converts a list of RDKit Mol objects to a list of unique molecules
    based on their canonical SMILES representation.

    :param mol_list: List of RDKit Mol objects.
    :return: List of unique RDKit Mol objects.
    """
    # Use a dictionary to retain unique Mol objects keyed by their canonical SMILES
    canonical_mol_dict = {}

    # Iterate over the list of Mol objects
    for mol in mol_list:
        try:
            # Convert Mol object to canonical SMILES
            canonical_smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
            # Add Mol object to the dictionary if canonical SMILES is not already added
            if canonical_smiles not in canonical_mol_dict:
                canonical_mol_dict[canonical_smiles] = mol
        except Exception as e:
            # Handle errors if a molecule cannot be converted
            print(f"Error processing molecule {Chem.MolToSmiles(mol)}: {e}")

    # Return the unique Mol objects
    return list(canonical_mol_dict.values())
def remove_small_rationales(molecules_list, cutoff):
    passing_molecules=[]
    for c in molecules_list:
        size = c.GetNumAtoms() 
        if size >= cutoff:
            passing_molecules.append(c)

    
    # Check if molecule has at least one ring
    
    # Return True if the molecule has a complete ring, otherwise False
    return passing_molecules
def filter_single_component_molecules(mol_list):
    """
    Filters out molecules that have more than one fragment.

    :param mol_list: List of RDKit Mol objects.
    :return: List of RDKit Mol objects that are single component molecules.
    """
    single_component_mols = []

    for mol in mol_list:
        # Get number of fragments
        num_fragments = len(Chem.GetMolFrags(mol, asMols=False))

        # Keep molecule if it has only one fragment
        if num_fragments == 1:
            single_component_mols.append(mol)

    return single_component_mols


def generate_fingerprints(mol_list):
    """
    Generates Morgan fingerprints for a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings.
    
    Returns:
        List of tuples with (SMILES, fingerprint) for valid molecules.
    """
    fingerprints = []
    for mol in mol_list:
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fingerprints.append((mol, fp))
        else:
            print(f"Invalid SMILES string: {smiles}")
    
    return fingerprints
def filter_unique_molecules(fingerprints):
    """
    Filters out molecules that are identical by Tanimoto similarity.
    
    Args:
        fingerprints: List of tuples with (SMILES, fingerprint).
    
    Returns:
        List of unique SMILES strings.
    """
    unique_mols = []
    seen_fps = set()  # Use a set to store unique fingerprints
    
    for smiles, fp in fingerprints:
        # Convert fingerprint to a bit string for hashing
        fp_bits = fp.ToBitString()
        if fp_bits not in seen_fps:
            unique_mols.append(smiles)
            seen_fps.add(fp_bits)
    

    return unique_mols

def get_unique_tanimoto(mol_list):
    fingerprints =  generate_fingerprints(mol_list)
    return filter_unique_molecules(fingerprints)
def filter_mols(mol_list):

    molecules_rationales = remove_redundant_molecules(mol_list)
    print(len(molecules_rationales))

    #molecules_rationales = has_complete_ring(molecules_rationales)
    molecules_rationales = remove_small_rationales(molecules_rationales, 10)
    molecules_rationales = get_unique_canonical_molecules(molecules_rationales)
    molecules_rationales = get_unique_tanimoto(molecules_rationales)
    molecules_rationales = filter_single_component_molecules(molecules_rationales)
    print(len(molecules_rationales))
    return molecules_rationales
