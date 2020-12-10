import numpy as np
import pandas as pd
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as d
import rdkit.Chem.Fragments as f
import rdkit.Chem.Lipinski as l
from rdkit.Chem import AllChem
# from generate_features import generate_features
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Normalizer, MinMaxScaler


def main():
    # generate_features()
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    train_df = pd.read_csv(os.path.join(scriptDir, "train_df.csv"))

    train_df = train_df.rename(columns={"Active": "CLASS"})
    train_df = train_df.rename(columns={"INDEX": "ID"})
    train_df = under_sample(train_df)

    features = list(set(train_df.columns.tolist()) -
                    {"ID", "CLASS", "SMILES", "Lipinski"})
    p1 = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=291, max_depth=8))
    # for clf in [p1]:
    #     scores = cross_val_score(
    #         clf, train_df[features], train_df['CLASS'], scoring='f1', cv=5)
    #     print("f1_score: %0.4f (+/- %0.2f) [%s]" %
    #           (scores.mean(), scores.std(), str(clf)))

    print("start")
    # params = {'randomforestclassifier__n_estimators': [
    #     50, 100, 200, 300], 'randomforestclassifier__max_depth': [None, 5, 10, 15]}

    # # f1
    # gs = GridSearchCV(
    #     estimator=p1,
    #     scoring="f1",
    #     cv=5,
    #     n_jobs=-1,
    #     param_grid=params)

    # gs.fit(train_df[features], train_df['CLASS'])

    # print(gs.cv_results_)
    # print(gs.best_estimator_)
    # print(gs.best_params_)
    # print(gs.best_score_)

    # plot_grid_search(gs.cv_results_, params)

    # roc_auc
    # gs = GridSearchCV(
    #     estimator=p1,
    #     scoring="roc_auc",
    #     cv=5,
    #     n_jobs=-1,
    #     param_grid=params)

    # gs.fit(train_df[features], train_df['CLASS'])

    # print(gs.cv_results_)
    # print(gs.best_estimator_)
    # print(gs.best_params_)
    # print(gs.best_score_)

    # plot_grid_search(gs.cv_results_, params)
    # plt.savefig("plt.png")

    test_df = pd.read_csv(os.path.join(scriptDir, "./test_df.csv"))
    test_df = test_df.rename(columns={"INDEX": "ID"})

    p1 = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=300, max_depth=None))
    p1.fit(train_df[features], train_df['CLASS'])
    probabilities = p1.predict_proba(test_df[features])
    with open("7.txt", "w") as f:
        f.write("0.77298779")
        f.write("\r\n")
        for p in probabilities:
            f.write(str(p[1]))
            f.write("\r\n")
    print("predictions...")
    print(p1.predict(test_df[features]))


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


def plot_grid_search(cv_results, params):
    grid_param_1 = list(params.values())[0]
    grid_param_2 = list(params.values())[1]
    name_param_1 = list(params.keys())[0]
    name_param_2 = list(params.keys())[1]

    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(
        len(grid_param_2), len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(
        len(grid_param_2), len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1, 1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx, :],
                '-o', label=name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')


main()
