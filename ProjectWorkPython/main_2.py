import numpy as np
import pandas as pd
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as d
import rdkit.Chem.Fragments as f
import rdkit.Chem.Lipinski as l
from rdkit.Chem import AllChem
from generate_features import generate_features
import os
import time
from preprocessing import create_column_filter, apply_column_filter
from preprocessing import create_imputation, apply_imputation
from preprocessing import create_normalization, apply_normalization
from preprocessing import accuracy, auc, brier_score, split
from classification import RandomForest
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def main():
    # generate_features()
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    train_df = pd.read_csv(os.path.join(scriptDir, "./training_df.csv"))
    # test_df = pd.read_csv(os.path.join(scriptDir, "./testing_df.csv"))
    train_df = train_df.rename(columns={"Active": "CLASS"})
    train_df = train_df.rename(columns={"INDEX": "ID"})
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42)
    # train_df = under_sample(train_df)
    train_df = under_sample(train_df)
    # val_df = under_sample(val_df)
    print(train_df.groupby('CLASS').count())

    train_df, column_filter = create_column_filter(train_df)
    val_df = apply_column_filter(val_df, column_filter)
    # test_df = apply_column_filter(test_df, column_filter)

    train_df, normalization = create_normalization(train_df)
    val_df = apply_normalization(val_df, normalization)

    print(train_df.shape)
    print(val_df.shape)
    val_labels = val_df["CLASS"]
    features = list(set(train_df.columns.tolist())-{"ID", "CLASS"})

    # print(train_df)
    # print(test_df)
    # clf1 = LogisticRegression(random_state=1, multi_class='auto')
    clf2 = (MLPClassifier(solver='lbfgs', alpha=1e-4,
                          hidden_layer_sizes=(10, 2)), "MLP")
    # clf3 = GaussianNB()
    # clf4 = AdaBoostClassifier(DecisionTreeClassifier())
    # clf5 = MLPClassifier(random_state=1, max_iter=100)
    # clf6 = GradientBoostingClassifier()

    # eclf = VotingClassifier(
    #     estimators=[('rf', clf2), ('ada', clf4), ('gb', clf6)],
    #     voting='hard')

    for clf, label in [clf2]:
        clf.fit(train_df[features], train_df['CLASS'])
        score = clf.score(val_df[features], val_labels)
        train_score = clf.score(train_df[features], train_df["CLASS"])
        predictions = clf.predict(val_df[features])
        cm = confusion_matrix(val_labels, predictions)
        print("Accuracy: Val: %0.4f Train: %0.4f [%s]" % (
            score, train_score, label))
        print(cm)


def under_sample(train_df):
    false_sample = train_df[train_df["CLASS"] == 1]
    true_sample = train_df[train_df["CLASS"]
                           == 0].sample(n=false_sample.shape[0])
    return pd.concat([false_sample, true_sample]).sample(frac=1).reset_index(drop=True)


def over_sample(train_df):
    false_sample = train_df[train_df["CLASS"] == 0]
    true_sample = train_df[train_df["CLASS"] == 1].sample(
        n=false_sample.shape[0], replace=True)
    return pd.concat([false_sample, true_sample]).sample(frac=1).reset_index(drop=True)


main()
