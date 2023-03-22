import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms a column to category
    """

    def __init__(self, columns) -> None:
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.columns:
            X[feature] = X[feature].astype("category")
        return X


class CategoricalMapper(BaseEstimator, TransformerMixin):
    """
    Maps categories in a feature to between (0, len(distinct categories in the feature))

    """

    def __init__(self, columns) -> None:
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns

        self.map = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalTransformer":
        X = X.copy()
        for feature in self.columns:
            unique_features = X[feature].unique()
            self.map[feature] = {}
            for idx, feat in enumerate(unique_features):
                self.map[feature][feat] = idx

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for feature in self.columns:
            X[feature] = X[feature].map(self.map[feature])
            X[feature] = X[feature].fillna(-1)
            X[feature] = X[feature].astype("category")
        return X
