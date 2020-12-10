import numpy as np
import pandas as pd
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as d
import rdkit.Chem.Fragments as f
import rdkit.Chem.Lipinski as l
from rdkit.Chem import AllChem
import os


def generate_features():
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    training_df = pd.read_csv(os.path.join(
        scriptDir, "training_smiles.csv"))

    testing_df = pd.read_csv(os.path.join(scriptDir, "test_smiles.csv"))

    # numpy array to store features - 8 features to include but the fingerprint feature will be encoded into 124 features
    feature_numpy = np.zeros((training_df.shape[0], (7+124)))

    # create fingerprint column names
    fp_column_name_list = ["fp_" + str(i) for i in range(124)]

    # list to store each instances value for the 124 fingerprint bits
    fingerprint_bits_list = []

    # iterates over rows and fills feature numpy with each feature value
    for idx, row in training_df.iterrows():
        m = Chem.MolFromSmiles(row['SMILES'])
        feature_numpy[idx][0] = training_df['INDEX'][idx]
        # Smiles (str) will be added later outside the for loop
        feature_numpy[idx][2] = m.GetNumAtoms()
        feature_numpy[idx][3] = d.CalcExactMolWt(m)
        feature_numpy[idx][4] = f.fr_Al_COO(m)
        feature_numpy[idx][5] = l.HeavyAtomCount(m)
        feature_numpy[idx][6] = training_df['ACTIVE'][idx]
        row_fingerprint_array = np.array(
            AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=124))
        for i in range(len(row_fingerprint_array)):
            feature_numpy[idx][7+i] = row_fingerprint_array[i]

    # gets column names
    column_list = ["INDEX", "SMILES", "nrAtoms",
                   "ExactMolWT", "Fragments", "Lipinski", "Active"]
    # Adds fingerprint columnn names
    column_list.extend(fp_column_name_list)
    # Creates new df with features included
    train_df = pd.DataFrame(data=feature_numpy, columns=column_list)
    train_df['SMILES'] = training_df['SMILES']

    train_df.to_csv(os.path.join(scriptDir, "train_df.csv"))

    # Same for testing
    # gets column names
    column_list = ["INDEX", "SMILES", "nrAtoms",
                   "ExactMolWT", "Fragments", "Lipinski"]
    # Adds fingerprint columnn names
    column_list.extend(fp_column_name_list)
    # numpy array to store features - 8 features to include but the fingerprint feature will be encoded into 124 features
    feature_numpy = np.zeros((testing_df.shape[0], (6+124)))
    fingerprint_bits_list = []
    for idx, row in testing_df.iterrows():
        m = Chem.MolFromSmiles(row['SMILES'])
        feature_numpy[idx][0] = testing_df['INDEX'][idx]
        # Smiles (str) will be added later outside the for loop
        feature_numpy[idx][2] = m.GetNumAtoms()
        feature_numpy[idx][3] = d.CalcExactMolWt(m)
        feature_numpy[idx][4] = f.fr_Al_COO(m)
        feature_numpy[idx][5] = l.HeavyAtomCount(m)
        row_fingerprint_array = np.array(
            AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=124))
        for i in range(len(row_fingerprint_array)):
            feature_numpy[idx][6+i] = row_fingerprint_array[i]

    test_df = pd.DataFrame(data=feature_numpy, columns=column_list)
    test_df['SMILES'] = testing_df['SMILES']

    test_df.to_csv(os.path.join(scriptDir, "test_df.csv"))


generate_features()
