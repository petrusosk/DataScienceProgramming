import numpy as np
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier


def create_tree(df):
    # Choose bootstrapped Indices
    idx = np.random.choice(len(df), replace=True, size=len(df))
    # Select bootstrap sample from indices
    sample = df.iloc[idx]
    features = list(set(df.columns.tolist())-{"ID", "CLASS"})
    # Create and fit Tree Classifier with log2 maximum features
    tree = DecisionTreeClassifier(max_features="log2")
    return tree.fit(sample[features], sample["CLASS"])


class RandomForest():
    def __init__(self):
        return

    def fit(self, df, no_trees=100):
        # Convert the column to category dtype
        df['CLASS'] = df['CLASS'].astype('category')
        # List of possible labels
        self.labels = df['CLASS'].cat.categories.tolist()

        # Create and Apply Preprocessing
        # df, self.column_filter = create_column_filter(df)
        # df, self.imputation = create_imputation(df)
        # df, self.one_hot = create_one_hot(df)
        # Create no_tree classifiers
        self.model = [create_tree(df) for _ in range(no_trees)]
        return

    def predict(self, df):
        # Apply Preprocessing
        # df = apply_column_filter(df, self.column_filter)
        # df = apply_imputation(df, self.imputation)
        # df = apply_one_hot(df, self.one_hot)
        features = list(set(df.columns.tolist())-{"ID", "CLASS"})

        # Matrix to store predictions
        predictions = np.zeros((len(df), len(self.labels)))

        for tree in self.model:
            # Generate predictions for tree
            y = tree.predict_proba(df[features])

            # Temporary matrix to store tree predictions
            tree_predictions = np.zeros((len(df), len(self.labels)))

            # Transpose the predictions in order to loop through the collumns/Classes
            for i, probs in enumerate(y.T):
                # Find true index of class
                c = tree.classes_[i]
                c_i = self.labels.index(c)
                # Add class predictions for each datapoint to tree_predicitons
                tree_predictions[:, c_i] += probs

            predictions = predictions + tree_predictions

        predictions = pd.DataFrame(predictions, columns=self.labels)
        # Divide predictions by no_trees to normalize
        predictions = predictions.div(len(self.model), axis=0)
        return predictions
