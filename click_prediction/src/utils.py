from sklearn.model_selection import StratifiedKFold
from config import config
from sklearn.feature_selection import chi2
import pandas as pd


def create_folds(df, y):
    """
    helper function to create folds using StratifiedKFold validation
    """
    df = df.copy()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    df["fold"] = -1

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid_idx, "fold"] = fold

    return df


def feature_selection_chisq(df, target, cols=None):
    """
    chi-sq function for feature selection
    """
    df = df.copy()

    scores = chi2(df, target)
    res = (
        pd.Series(scores[1], index=df.columns, name="p_values")
        .sort_values(ascending=False)
        .reset_index()
    )
    return res
