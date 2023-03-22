import pathlib
import os 

ROOT = pathlib.Path(os.getcwd()).resolve().parent

DATA_DIR = ROOT / 'data'

DATA_RAW = DATA_DIR / 'raw' / 'DS assignment-Lead (1).xlsx'

TRAIN_DATASET_RAW = DATA_DIR / "raw" / "train_data.parquet"
TEST_DATASET_RAW = DATA_DIR / "raw" / "test_data.parquet"


X_TRAIN = DATA_DIR / "final" / "x_train.csv"
X_VALID = DATA_DIR / "final" / "x_valid.csv"
Y_TRAIN = DATA_DIR / "final" / "y_train.csv"
Y_VALID = DATA_DIR  / "final" / "y_valid.csv"

FINAL_DATASET = DATA_DIR / "final" /  "test_dataset_with_preds.csv"


###################################################################

SEED = 42

TARGET = 'click'

FEATURES = ['hash_0',
            'hash_1',
            'hash_2',
            'hash_3',
            'hash_4',
            'hash_5',
            'hash_6',
            'hash_7',
            'hash_8',
            'hash_9',
            'hash_10',
            'hash_11',
            'hash_12',
            'hash_13',
            'hash_14',
            'hash_15',
            'hash_16',
            'hash_17',
            'hash_18']

#######################################################################

HPARAM_TRIALS = 10

HPARAM_DIRECTION = 'maximize'


PARAMS =  {
    'max_depth': 4,
 'learning_rate': 0.0951207163345817,
 'n_estimators': 87,
 'subsample': 0.15751320499779725,
 'gamma': 1.77071686435378e-07,
 'colsample_bytree': 0.020511104188433976,
  'use_label_encoder': False,
 'enable_categorical': False,
 'scale_pos_weight' : 1.811455460003245
}

