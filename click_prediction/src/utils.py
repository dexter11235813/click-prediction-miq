from sklearn.model_selection import StratifiedKFold
from config import config
from scipy.stats import chi2_contingency
import pandas as pd
from typing import List

def create_folds(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    helper function to create folds using StratifiedKFold validation
    """
    df = df.copy()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    df["fold"] = -1

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid_idx, "fold"] = fold

    return df




def get_chisq_result(df: pd.DataFrame, target: str, skip:List[str]) -> pd.DataFrame:
    '''
    Calculates the p-value of the chi2 test of association between target column and the df. 
    
    Returns a dataframe with the column names, and the associated p value 
    '''
    column1 = []
    column2 = []
    chisq_p_value = []

    colnames = df.columns.tolist()
    for idx1 in range(len(colnames)- 1):
        col1 = colnames[idx1]
        col2 = config.TARGET

        if col1 not in skip:
            p_value = chi2_contingency(
                pd.crosstab(index=df[col1], columns=df[col2]))[1]
            column1.append(col1)
            column2.append(col2)
            chisq_p_value.append(p_value)
    result = pd.DataFrame({"col1": column1, "col2": column2, "p_value": p_value})
    return result
