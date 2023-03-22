import optuna

import pandas as pd
from category_encoders import TargetEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTEN
from sklearn.metrics import f1_score

from src.preprocessing import preprocessors
from config import config


import xgboost as xgb
import lightgbm as lgb


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.DataFrame,
    y_valid: pd.DataFrame,
    smote: bool,
):
    """The objective function that returns the f1-score for a model fitted on parameters drawn from specified distribution."""

    target_encoder = TargetEncoder()
    categorical_encoder = preprocessors.CategoricalTransformer(config.FEATURES)
    categorical_mapper = preprocessors.CategoricalMapper(config.FEATURES)
    if smote:
        smote_nc = SMOTEN(random_state=config.SEED)

    max_depth = trial.suggest_int(name="max_depth", low=3, high=7)
    learning_rate = trial.suggest_float(name="learning_rate", low=0.001, high=0.1)
    n_estimators = trial.suggest_int(name="n_estimators", low=50, high=100)
    subsample = trial.suggest_loguniform("subsample", 0.01, 1.0)
    gamma = trial.suggest_loguniform(name="gamma", low=1e-8, high=1)
    colsample_bytree = trial.suggest_loguniform("colsample_bytree", 0.01, 1.0)

    weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    params = {
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "gamma": gamma,
        "enable_categorical": True,
        "tree_method": "hist",
        "scale_pos_weight": weight,
    }
    classifier = xgb.XGBClassifier(**params)

    if smote:
        pipeline = Pipeline(
            [
                ("categorical_encoder", categorical_encoder),
                ("categorical_mapper", categorical_mapper),
                ("SMOTENC", smote_nc),
                ("target_encoder", target_encoder),
                ("clf", classifier),
            ]
        )
    else:
        pipeline = Pipeline(
            [
                ("categorical_encoder", categorical_encoder),
                ("categorical_mapper", categorical_mapper),
                ("target_encoder", target_encoder),
                ("clf", classifier),
            ]
        )
    classifier = pipeline.fit(X=X_train, y=y_train)
    y_pred = pipeline.predict(X=X_valid)
    return f1_score(y_true=y_valid, y_pred=y_pred)


def tune_hyperparams(X_train, X_valid, y_train, y_valid, smote=False) -> optuna.study:
    """
    Returns the Optuna Trial object fitted on
        X_train :- training features
        y_train :- training labels
        X_valid :- validation features
        y_valid :- validation labels
    """
    sampler = optuna.samplers.TPESampler(seed=config.SEED)
    study = optuna.create_study(
        direction=config.HPARAM_DIRECTION, study_name="Optimize", sampler=sampler
    )
    study.optimize(
        lambda trial: objective(trial, X_train, X_valid, y_train, y_valid, smote),
        n_trials=config.HPARAM_TRIALS,
    )

    return study
