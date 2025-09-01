import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')
print("Imports done successfully")

def preprocess_data(df, encoders=None):
    df = df.copy()
    df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
    df['loan'] = df['loan'].map({'yes': 1, 'no': 0})
    df['default'] = df['default'].map({'yes': 1, 'no': 0})
    cols_to_encode = ['job', 'marital', 'education', 'contact', 'poutcome', 'month']
    if encoders is None:
        encoders = {}
        for col in cols_to_encode:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        df['pdays'] = df['pdays'].replace(-1, 999)
        return df, encoders
    else:
        for col in cols_to_encode:
            le = encoders.get(col)
            if le is None:
                raise ValueError(f"Encoder for column {col} not found.")
            df[col] = df[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
            if '<unknown>' in df[col].values and '<unknown>' not in le.classes_:
                le.classes_ = np.append(le.classes_, '<unknown>')
            df[col] = le.transform(df[col])
        df['pdays'] = df['pdays'].replace(-1, 999)
        return df, None

def feature_engineering(df, job_categories=None):
    df = df.copy()
    df['duration_balance'] = df['duration'] * df['balance']
    df['duration_age'] = df['duration'] * df['age']
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 97], labels=[0,1,2,3,4]).astype(int)
    df['duration_bin'] = pd.cut(df['duration'], bins=[0,60,300,600,float('inf')], labels=[0,1,2,3], right=False).astype(int)
    df['has_prev_contact'] = (df['previous'] > 0).astype(int)
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['was_contacted'] = (df['pdays'] != 999).astype(int)
    df['balance_posi'] = (df['balance'] > 0).astype(int)
    df['has_previous'] = (df['previous'] > 0).astype(int)
    df['is_first_contact'] = (df['campaign'] == 1).astype(int)
    df['log_duration'] = np.log1p(df['duration'])
    df['sqrt_duration'] = np.sqrt(df['duration'])
    df['log_campaign'] = np.log1p(df['campaign'])
    df['pdays_log'] = np.log1p(df['pdays'] + 2)
    df['previous_log'] = np.log1p(df['previous'] + 1)
    df['log_age'] = np.log1p(df['age'])
    df['is_contacted'] = (df['contact'] != -1).astype(int)
    df['has_previous_contact'] = ((df['previous'] > 0) & (df['pdays'] != -1)).astype(int)
    df['is_working_age'] = df['age'].between(25, 60).astype(int)
    df['loan_and_default_risk'] = ((df['loan'] == 1) & (df['default'] == 1)).astype(int)
    if job_categories is not None:
        job_map = dict(enumerate(job_categories))
        df['employment_risk'] = df['job'].map(job_map).isin(['unemployed','unknown','housemaid']).astype(int)
    df['campaign_intensity'] = df['campaign'] / (df['pdays'].replace(-1, np.nan).fillna(999) + 1)
    df['avg_balance_per_contact'] = df['balance'] / (df['previous'] + 1)
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['recently_contacted'] = ((df['pdays'] < 30) & (df['pdays'] != -1)).astype(int)
    return df

print("Loading and preprocessing data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test['id'].copy()
train = train.drop(columns='id')
test = test.drop(columns='id')

print("Loading external dataset (positive samples only)...")
orig = pd.read_csv("bank-full.csv", sep=';')
orig['y'] = orig['y'].map({'no':0, 'yes':1})
external_positive = orig[orig['y']==1].copy()
print(f"Found {len(external_positive)} positive samples")

train_combined = pd.concat([train, external_positive], ignore_index=True)
train_combined = train_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Combined train size: {len(train_combined)}")
print(f"Class distribution:\n{train_combined['y'].value_counts().sort_index()}")

train_processed, fitted_encoders = preprocess_data(train_combined)
train_final = feature_engineering(train_processed)

X_train = train_final.drop(columns=['y'])
y_train = train_final['y']

test_processed, _ = preprocess_data(test, encoders=fitted_encoders)
test_final = feature_engineering(test_processed)
X_test = test_final
print(f"Test data shape: {X_test.shape}")

cat_cols = ['job', 'marital', 'education', 'contact', 'poutcome', 'month',
            'age_group', 'duration_bin']

print("\nTraining LightGBM model with cross-validation...")

# Use 5-fold stratified cross-validation
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
y_probs = np.zeros(len(X_test))
models = []
val_aucs = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"Training fold {fold + 1}/{n_splits} >>>")
    X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    #got these params from 100 trials optuna
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=20000,
        learning_rate=0.04770062672645345,
        num_leaves=110,
        max_depth=15,
        min_child_samples=7,
        subsample=0.7535901856612678,
        colsample_bytree=0.3086234831127026,
        reg_alpha=1.493378518925405,
        reg_lambda=1.8558078940110203,
        max_bin=5927,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(500),
            lgb.log_evaluation(period=500)
        ]
    )
    
    models.append(model)
    
    # Average predictions across all folds
    y_probs += model.predict_proba(X_test)[:, 1] / n_splits
    
    # Val AUC
    val_pred = model.predict_proba(X_fold_val)[:, 1]
    val_auc = roc_auc_score(y_fold_val, val_pred)
    val_aucs.append(val_auc)

print(f"Mean AUC: {np.mean(val_aucs):.6f}")

print("\nGenerating final predictions...")
submission = pd.DataFrame({
    'id': test_ids,
    'y': y_probs
})
submission.to_csv('submission_lightgbm.csv', index=False)
print("Submission file saved: submission_lightgbm.csv")
print(submission.head(10))