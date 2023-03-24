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
from sklearn.ensemble import RandomForestClassifier

# def objective(
#     trial: optuna.Trial,
#     X_train: pd.DataFrame,
#     X_valid: pd.DataFrame,
#     y_train: pd.DataFrame,
#     y_valid: pd.DataFrame,
#     smote: bool,
# ):
#     """The objective function that returns the f1-score for a model fitted on parameters drawn from specified distribution."""

#     target_encoder = TargetEncoder()
#     categorical_encoder = preprocessors.CategoricalTransformer(config.FEATURES)
#     categorical_mapper = preprocessors.CategoricalMapper(config.FEATURES)
#     if smote:
#         smote_nc = SMOTEN(random_state=config.SEED)

#     max_depth = trial.suggest_int(name="max_depth", low=3, high=7)
#     learning_rate = trial.suggest_float(name="learning_rate", low=0.001, high=0.1)
#     n_estimators = trial.suggest_int(name="n_estimators", low=50, high=100)
#     subsample = trial.suggest_loguniform("subsample", 0.01, 1.0)
#     gamma = trial.suggest_loguniform(name="gamma", low=1e-8, high=1)
#     colsample_bytree = trial.suggest_loguniform("colsample_bytree", 0.01, 1.0)

#     weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
#     params = {
#         "max_depth": max_depth,
#         "learning_rate": learning_rate,
#         "n_estimators": n_estimators,
#         "subsample": subsample,
#         "colsample_bytree": colsample_bytree,
#         "gamma": gamma,
#         "enable_categorical": True,
#         "tree_method": "hist",
#         "scale_pos_weight": weight,
#     }
#     classifier = xgb.XGBClassifier(**params)

#     if smote:
#         pipeline = Pipeline(
#             [
#                 ("categorical_encoder", categorical_encoder),
#                 ("categorical_mapper", categorical_mapper),
#                 ("SMOTENC", smote_nc),
#                 ("target_encoder", target_encoder),
#                 ("clf", classifier),
#             ]
#         )
#     else:
#         pipeline = Pipeline(
#             [
#                 ("categorical_encoder", categorical_encoder),
#                 ("categorical_mapper", categorical_mapper),
#                 ("target_encoder", target_encoder),
#                 ("clf", classifier),
#             ]
#         )
#     classifier = pipeline.fit(X=X_train, y=y_train)
#     y_pred = pipeline.predict(X=X_valid)
#     return f1_score(y_true=y_valid, y_pred=y_pred)

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

    model_dict = {
        "xgb": xgb.XGBClassifier,
        "rf": RandomForestClassifier,
    }
    model_type = trial.suggest_categorical(name="model_type", choices=["xgb", "rf"])
    sample = trial.suggest_float(name="sample", low=0.1, high=0.3)
    max_depth = trial.suggest_int(name="max_depth", low=3, high=7)
    learning_rate = trial.suggest_float(name="learning_rate", low=0.001, high=0.1)
    n_estimators = trial.suggest_int(name="n_estimators", low=50, high=100)
    subsample = trial.suggest_loguniform("subsample", 0.01, 1.0)
    gamma = trial.suggest_loguniform(name="gamma", low=1e-8, high=1)
    colsample_bytree = trial.suggest_loguniform("colsample_bytree", 0.01, 1.0)

    # X_train, y_train = undersample(X_train, y_train, sample=0.2)
    weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    params = {
        "max_depth": max_depth,
        "n_estimators": n_estimators,
    }
    if smote:
        pass
    else:
        if model_type == "xgb":
            params["scale_pos_weight"] = weight
        else:
            params["class_weight"] = "balanced"
    if model_type == "xgb":
        params["learning_rate"] = learning_rate
        params["gamma"] = gamma
        params["enable_categorical"] = True
        params["tree_method"] = "hist"
        params["subsample"] = subsample
        params["scale_pos_weight"] = weight
        params["colsample_bytree"] = colsample_bytree
    classifier = model_dict[model_type](**params)

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
    y_pred_proba = pipeline.predict_proba(X_valid)
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
