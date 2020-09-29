###########################################################################################################
#                                                                                                         #
# FileName    [metrics.py]                                                                                #
#                                                                                                         #
# PackageName [postprocessing]                                                                            #
#                                                                                                         #
# Synopsis    [This file contains the functions for calculating metrics on synthesized molecules.         #
#              The functions are chemical validity, uniqness, novelty, the Tanimoto Similarity mean       #
#              of chemically valid molecules]                                                             #
#                                                                                                         #
# Author      [Leonardo Picchiami]                                                                        #
#                                                                                                         #
###########################################################################################################


from typing import List

from rdkit import Chem
from rdkit import DataStructs


def chemical_validity(generated_synt_valid_set : List[str]) -> (List[str], float):
    '''
    Returns the list of chemically valid molecules generated, 
    and the fraction of chemically valid molecules generated with respect to all syntactically valid molecules generated.

    Paramters
    ---------
    generated_synt_valid_set : list
        List of syntactically valid molecules generated.

    Returns 
    -------
    tuple : tuple of int
        Tuple containing the list of chemically valid molecules generated 
        and the fraction of chemically valid molecules generated with respect to the syntactically valid molecules generated.
    '''
    
    chem_valid_smiles = []
    
    for smile in generated_synt_valid_set:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            chem_valid_smiles.append(smile)
            
    return (chem_valid_smiles, len(chem_valid_smiles)/ len(generated_synt_valid_set))



def uniqueness(generated_synt_valid_set : List[str]) -> float:
    '''
    Returns the fraction of unique molecules generated.

    Paramters
    ---------
    generated_synt_valid_set : list
        List of syntactically valid molecules generated.

    Returns
    -------
    len(set(generated_synt_valid_set)) / len(generated_synt_valid_set) : float
        The fraction of uniques molecules generated.
    '''
    
    return len(set(generated_synt_valid_set)) / len(generated_synt_valid_set)



def novelty(dataset : List[str], generated_synt_valid_set : List[str]) -> float:
    '''
    Returns the number of syntactically valid new molecules generated with respect to the input dataset.

    Parameters
    ----------
    dataset : list
        Dataset containing the list of SMILE molecules.

    generated_synt_valid_set : list
        List of syntactically valid molecules generated.


    Returns
    -------
    total_new_molecules / len(generated_synt_valid_set) : float
        Fraction of new molecules with respect to the synthesized input dataset.
    '''
    
    total_new_molecules = 0
    for generated_smile in generated_synt_valid_set:
        if generated_smile not in dataset:
            total_new_molecules += 1

    return total_new_molecules / len(generated_synt_valid_set)



def tanimoto_similarity_average(generated_synt_valid_set : List[str]) -> float:
    '''
    Calculate the average of the tanimoto simlarity with respect to the RDK Fingerprint 
    among all the generated syntactically valid molecules.
    
    Paramters
    ---------
    generated_synt_valid_set : list
        List of syntactically valid molecules generated.
    
    Returns
    -------
    sum(fing_mols_generated) / len(fing_mols_generated)
        Average of tanimoto similarity among all syntactically valid molecules generated.
    '''
    
    mols_generated = []
    fing_mols_generated = []
    for smile in generated_synt_valid_set:
        molecule = Chem.MolFromSmiles(smile)
        if molecule:
            mols_generated.append(molecule)

    fps = [Chem.RDKFingerprint(mol) for mol in mols_generated]
    for i in range(len(fps)):
        for j in range(i, len(fps)):
            sim = DataStructs.FingerprintSimilarity(fps[i], fps[j])
            fing_mols_generated.append(sim)

    return sum(fing_mols_generated) / len(fing_mols_generated)
    

