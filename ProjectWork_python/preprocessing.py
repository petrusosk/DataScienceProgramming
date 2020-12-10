import numpy as np
import pandas as pd
import time
from pandas.api.types import CategoricalDtype


def create_column_filter(df):
    # List of columns we need to consider.
    # Use sets to easily subtract ID and CLASS from the list of columns.
    subset = list(set(df.columns.tolist())-{"ID", "CLASS"})

    # List of columns to keep
    column_filter = ["ID", "CLASS"]

    # Only select columns that contain at least 2 unique values.
    for col in subset:
        if df[col].nunique() > 1:
            column_filter.append(col)

    return apply_column_filter(df, column_filter), column_filter


def apply_column_filter(input, column_filter):
    # Intersection of columns in input and columns to filter.
    keep_columns = list(
        set(input.columns.tolist()).intersection(set(column_filter)))
    return input[keep_columns]


def create_normalization(df, normalizationtype="minmax"):
    # List of columns we need to consider.
    # Use sets to easily subtract ID and CLASS from the list of columns.
    subset = list(set(df.columns.tolist())-{"ID", "CLASS"})

    # zip 3 lists together: A list of the type, the minimums of the columns, the maximums of the colums
    if normalizationtype == "minmax":
        norms = list(zip(["minmax" for _ in range(len(subset))],
                         df[subset].min(), df[subset].max()))
    elif normalizationtype == "zscore":
        norms = list(zip(["zscore" for _ in range(len(subset))],
                         df[subset].mean(), df[subset].std()))
    else:
        raise ValueError("wrong normalization type")

    # Combine the columns with their respective values, and convert to a dict.
    normalization = dict(zip(subset, norms))

    return apply_normalization(df, normalization), normalization


def apply_normalization(input_df, normalization):
    df = input_df.copy()

    # loop through the dictionaly, and apply the repsective normalization rowwise.
    for col, norm in normalization.items():
        if norm[0] == "minmax":
            df[[col]] = (df[[col]]-norm[1])/(norm[2]-norm[1])
        if norm[0] == "zscore":
            df[[col]] = (df[[col]]-norm[1])/(norm[2])

    return df


def create_imputation(df):
    # List of columns we need to consider.
    # Use sets to easily subtract ID and CLASS from the list of columns.
    subset = list(set(df.columns.tolist())-{"ID", "CLASS"})

    numeric_means = df[subset].mean(numeric_only=True).to_dict()
    # for catagoric (non-number) columns, calculate the modes and convert to dict.
    catagoric_modes = df[subset].select_dtypes(
        exclude=[np.number]).mode().iloc[0].to_dict()

    imputation = {**numeric_means, **catagoric_modes}
    return apply_imputation(df, imputation), imputation


def apply_imputation(input_df, imputation):
    df = input_df.copy()
    df = df.fillna(imputation)
    return df


def create_bins(df, nobins, bintype="equal-width"):
    # List of columns we need to consider, only numeric.
    # Use sets to easily subtract ID and CLASS from the list of columns.
    subset = list(set(df.select_dtypes(
        include=np.number).columns.tolist())-{"ID", "CLASS"})

    df = df.copy()
    binning = dict()
    # For every column, call the respective pandas function for binning.
    for col in subset:
        if bintype == "equal-width":
            df[col], binning[col] = pd.cut(
                df[col], nobins, retbins=True, duplicates='drop', labels=False)
        if bintype == "equal-size":
            df[col], binning[col] = pd.qcut(
                df[col], nobins, retbins=True, duplicates='drop', labels=False)

        # Convert boundaries of outer bins to infinite.
        binning[col][0] = -float('inf')
        binning[col][-1] = float('inf')

    return df, binning


def apply_bins(df, binning):
    df = df.copy()
    for col, bins in binning.items():
        df[col] = pd.cut(df[col], bins, labels=False)
    return df


def create_one_hot(df):
    # List of columns we need to consider, only numeric.
    # Use sets to easily subtract ID and CLASS from the list of columns.
    subset = list(set(df.select_dtypes(
        include=['object', 'category']).columns.tolist())-{"ID", "CLASS"})

    # Create dict of the possible values for each column
    one_hot = dict()
    for col in subset:
        one_hot[col] = df[col].dropna().unique()

    return apply_one_hot(df, one_hot), one_hot


def apply_one_hot(df, one_hot):
    df = df.copy()
    for col, values in one_hot.items():
        # convert values to categorical
        df[col] = df[col].astype(CategoricalDtype(values))
        # Use get_dummies to get one-hot encoding.
        dummies = pd.get_dummies(df[col], prefix=col, dtype=float)

        # Add new columns to dataframe, and drop the old column.
        df = pd.concat([df, dummies], axis=1)
        df.drop([col], axis=1, inplace=True)

    return df


def split(df, testfraction):
    df_split = df.copy()
    if testfraction > 1 or testfraction < 0:
        raise ValueError("testfraction must be between 0 and 1")

    # Calculate test length
    test_frac = int(df_split.shape[0] * testfraction)
    # Calculate random test indexes of length test_frac
    test_idx = np.random.permutation(df_split.shape[0])[:test_frac]

    # Create dataset based on test_idx
    trainingdf = df_split.drop(test_idx)
    testdf = df_split.loc[test_idx]

    return trainingdf, testdf


def accuracy(df, correctlabels):
    # Get the indexex of the columns with the highest values. Take the first if they are the same.
    predictions = df.idxmax(axis="columns")
    # Ternary expection to get the total amount of correct predictions.
    correct = sum(label == guess for label,
                  guess in zip(correctlabels, predictions))

    return correct/len(predictions)


def folds(input_df, nofolds=10):
    if nofolds <= 1:
        raise ValueError("nofolds should be greater than 1")
    # Randum shuffle by using a sample of fraction 1.
    df = input_df.sample(frac=1).reset_index(drop=True)
    # Use np.array_split to split into nofolds.
    return np.array_split(df, nofolds)


def brier_score(df, correctlabels):
    labels_vec = np.zeros([len(correctlabels), df.shape[1]])
    # Converts labels into one-hot encoded vectors
    for i, l in enumerate(correctlabels):
        labels_vec[i] = np.where(df.columns == l, 1, 0)

    values = df.copy().values

    # Calculate mean squared error.
    brier_score = np.sum(np.power(values - labels_vec, 2)) / values.shape[0]

    return brier_score


def auc(df: pd.DataFrame, correctlabels: []):
    auc_tot = 0
    for c in df.columns.tolist():
        class_labels = np.where(np.array(correctlabels) == c, True, False)
        weight = np.sum(class_labels) / class_labels.shape[0]

        tp = np.zeros(class_labels.shape[0])
        fp = np.zeros(class_labels.shape[0])
        for i, v in enumerate(df[c].values):
            if class_labels[i] == True:
                tp[i] = 1
            else:
                fp[i] = 1

        scores = np.zeros([len(tp), 3])
        for i in range(len(tp)):
            scores[i] = [df[c][i], tp[i], fp[i]]

        # sort in reverse
        scores = scores[scores[:, 0].argsort()[::-1]]

        auc = 0
        cov_tp = 0
        tot_tp = np.sum(tp)
        tot_fp = np.sum(fp)
        for i in range(scores.shape[0]):
            tp_i = scores[i][1]
            fp_i = scores[i][2]

            if fp_i == 0:  # no false positives
                cov_tp += tp_i
            elif tp_i == 0:  # no true positives
                auc += (cov_tp/tot_tp)*(fp_i/tot_fp)
            else:
                auc += (cov_tp/tot_tp)*(fp_i/tot_fp) + \
                    (tp_i/tot_tp)*(fp_i/tot_fp)/2
                cov_tp += tp_i

        auc_tot += auc * weight

    return auc_tot
