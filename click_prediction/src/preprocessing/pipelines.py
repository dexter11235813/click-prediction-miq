from category_encoders import TargetEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTEN
from src.preprocessing import preprocessors
from config import config

import xgboost as xgb


def get_pipeline(params, smote=False):
    """
    Returns the imblearn pipeline which combines all the transforms and classifier into one object.

    """
    target_encoder = TargetEncoder()
    categorical_encoder = preprocessors.CategoricalTransformer(config.FEATURES)
    categorical_mapper = preprocessors.CategoricalMapper(config.FEATURES)
    if smote:
        smote_nc = SMOTEN(random_state=config.SEED)
    if smote:
        pipeline = Pipeline(
            [
                ("categorical_encoding", categorical_encoder),
                ("categorical_mapping", categorical_mapper),
                ("SMOTE-NC", smote_nc),
                ("target_encoder", target_encoder),
                ("clf", xgb.XGBClassifier(**params)),
            ]
        )
    else:
        pipeline = Pipeline(
            [
                ("categorical_encoding", categorical_encoder),
                ("categorical_mapping", categorical_mapper),
                ("target_encoder", target_encoder),
                ("clf", xgb.XGBClassifier(**params)),
            ]
        )
    return pipeline
