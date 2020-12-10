from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import preprocessing

# Read in file
base_dir = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(base_dir + "/training_smiles.csv")

# rename ACTIVE to CLASS, so we can use our preprocessing functions
df = df.rename(columns={"ACTIVE": "CLASS"})
df = df.rename(columns={"INDEX": "ID"})

df["CLASS"].plot()

# subset for now TODO: remove
# df = df[:100]

# convert to moles
df["SMILES"] = df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x)).to_frame()

# turn SMILES column into fingerprint columns
n_bits = 124
fingerprints = np.zeros((df.shape[0], n_bits))
for i, row in df.iterrows():
    fingerprints[i] = AllChem.GetMorganFingerprintAsBitVect(
        row["SMILES"], 2, nBits=n_bits)

columns = ["fp_{}".format(i) for i in range(n_bits)]

df_fp = pd.DataFrame(fingerprints, columns=columns)

df = df.join(df_fp)

df = df.drop(columns=["SMILES"])

# split
df_train, df_val = preprocessing.split(df, 0.2)

df_train_labels = df_train["CLASS"]
df_val_labels = df_val["CLASS"]

# preprocess
df_train, column_filter = preprocessing.create_column_filter(df_train)

df_val = preprocessing.apply_column_filter(df_val, column_filter)

# remove ID and CLASS
features = list(set(df_train.columns.tolist())-{"ID", "CLASS"})
df_train = df_train[features]
df_val = df_val[features]

# test decision tree
dt = GaussianNB()
dt.fit(df_train, y=df_train_labels)
print(preprocessing.accuracy(pd.DataFrame(dt.predict(df_val)), df_val_labels))
