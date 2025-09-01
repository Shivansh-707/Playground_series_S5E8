import pandas as pd
import numpy as np
import random
import time
import os
import warnings

from catboost import CatBoost, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.simplefilter('ignore')

# I made 3 submission files changing the seeds and taking the average , got this idea from a kaggle master @vector-
#it did help push the roc auc of this individual file by 0.0002 ( little diffrence on paper but a lot on the leaderboard )


# Configuration
class CFG:
    seed = 42
    n_splits = 10
    learning_rate = 4e-1 

    num_boost_round = 30000
    early_stopping_rounds = 100
    verbose_eval = 200 if learning_rate >= 1e-1 else 500

    target = "y"

def seed_everything(seed: int = CFG.seed) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything()

# Load data
train_df = pd.read_csv("train.csv").drop(columns=['id'])
test_df  = pd.read_csv("test.csv").drop(columns=['id'])


original_df = pd.read_csv("bank-full.csv", sep=';')
original_df[CFG.target] = (original_df[CFG.target] == "yes").astype(int)

fts_base = test_df.columns

# Feature Engineering
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Binning
    df['duration_bin_10'] = pd.qcut(df['duration'], q=10, labels=False, duplicates='drop')
    df['duration_bin_20'] = pd.qcut(df['duration'], q=20, labels=False, duplicates='drop')
    df['balance_bin_10']  = pd.qcut(df['balance'],  q=10, labels=False, duplicates='drop')
    df['balance_bin_20']  = pd.qcut(df['balance'],  q=20, labels=False, duplicates='drop')
    df['campaign_bin']    = pd.cut(df['campaign'],
                                   bins=[-1, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100],
                                   labels=False)
    
    # Numerical log transformations
    df['log_balance']   = np.sign(df['balance']) * np.log1p(np.abs(df['balance']))
    df['log_duration']  = np.log1p(df['duration'])
    df['log_campaign']  = np.log1p(df['campaign'])
    df['log_pdays']     = np.log1p(df['pdays'] + 1)  
    df['log_previous']  = np.log1p(df['previous'])
    
    return df

train_fe    = create_features(train_df)
test_fe     = create_features(test_df)
original_fe = create_features(original_df)

# Final feature list
features = test_fe.columns

# Categorical features 
fts_categorical = fts_base.to_list()  # Only base features as categorical


print(train_fe.shape, test_fe.shape, original_fe.shape)
print("Categorical features:", fts_categorical)

# CatBoost parameters 
params = {
    'bootstrap_type': "MVS",
    'boosting_type': "Plain",
    'loss_function' : "Logloss",
    'random_state'  : CFG.seed,
    'iterations'    : CFG.num_boost_round,
    'learning_rate' : CFG.learning_rate,
    'depth'         : 8,
    'subsample'     : 0.9,         
    'random_strength': 2.0,
    'grow_policy'   : 'SymmetricTree',
    'task_type'     : "CPU",
}

oof = np.zeros(train_fe.shape[0])  
pred = np.zeros(test_fe.shape[0])
feature_importances = pd.DataFrame()

cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
splitter = cv.split(train_fe, train_fe[CFG.target])

for fold, (trn_idx, val_idx) in enumerate(splitter):
    start_time = time.time()

    # Augment with original data
    M_train = pd.concat([train_fe.iloc[trn_idx], original_fe])
    M_train = M_train.drop_duplicates(subset=fts_base, keep="first", ignore_index=True)
    X_train = M_train[features]
    y_train = M_train[CFG.target]

    X_valid = train_fe.loc[val_idx, features]
    y_valid = train_fe.loc[val_idx, CFG.target]
    X_test = test_fe[features].copy()

    # Set categorical dtypes 
    X_train[fts_categorical] = X_train[fts_categorical].astype("category")
    X_valid[fts_categorical] = X_valid[fts_categorical].astype("category")
    X_test[fts_categorical] = X_test[fts_categorical].astype("category")

    # Create Pool
    dtrain = Pool(X_train, label=y_train, cat_features=fts_categorical)
    dvalid = Pool(X_valid, label=y_valid, cat_features=fts_categorical)
    dtest = Pool(test_fe[features], cat_features=fts_categorical)

    # Training - CatBoost
    model = CatBoost(params)
    model.fit(
        dtrain,
        eval_set=dvalid,
        verbose=CFG.verbose_eval,
        early_stopping_rounds=CFG.early_stopping_rounds,
        use_best_model=True
    )

    # Prediction - (the best part haha)
    oof[val_idx] = model.predict(dvalid, prediction_type="Probability")[:, 1]
    pred += model.predict(dtest, prediction_type="Probability")[:, 1] / CFG.n_splits

    score = roc_auc_score(y_valid, oof[val_idx])
    end_time = time.time()
    print("----------------------------------------------------------------")
    print(
        f"fold: {fold:02d}, "+
        f"auc: {score:.6f}, "+
        f"best iteration: {model.get_best_iteration()}, "+
        f"best score: {model.get_best_score()['validation']['Logloss']: .6f}, "+
        f"elapsed time: {end_time-start_time: .2f} sec.\n"
    )

    # Track feature importance
    imp = model.get_feature_importance(
        Pool(X_train, label=y_train, cat_features=fts_categorical),
        type="PredictionValuesChange"
    )
    fi_tmp = pd.DataFrame()
    fi_tmp["feature"] = X_train.columns.to_list()
    fi_tmp["importance"] = imp
    fi_tmp["fold"] = fold
    fi_tmp["seed"] = CFG.seed
    feature_importances = pd.concat([feature_importances, fi_tmp], ignore_index=True)

# Overall CV score
score = roc_auc_score(train_fe[CFG.target], oof)
print("----------------------------------------------------------------")
print(f"auc: {score:.6f}")
print("----------------------------------------------------------------\n")

# Submission
sub_df = pd.read_csv("sample_submission.csv")
sub_df[CFG.target] = pred
sub_df.to_csv("submission_catboost_seed_change.csv", index=False)
print(sub_df.head())