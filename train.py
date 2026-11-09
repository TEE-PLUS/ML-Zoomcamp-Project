import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


import xgboost as xgb

import kagglehub
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')



# parameters
xgb_params =   {
        'eta': 0.1,
        'max_depth': 3,
        'min_child_weight': 5,

        'objective': 'binary:logistic',
        'eval_metric': 'auc',

        'nthread': 8,
        'seed': 42,
        'verbosity': 1,
    }
output_file = f'diabetes_model.bin'

# data preparation


# Download the dataset from Kaggle
dataset_path = Path(kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset"))
csv_path = dataset_path / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

print(f"Dataset downloaded to: {dataset_path}")
df = pd.read_csv(csv_path)

df.columns = df.columns.str.lower()



# Split the dataset into training and test sets
df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42,
    stratify=df["diabetes_binary"]
)

numerical = ["bmi", "menthlth", "physhlth"]
categorical = [col for col in df.columns if col not in numerical + ["diabetes_binary"]]



df_train, df_val = train_test_split(
    df_full_train,
    test_size=0.25,
    random_state=42,
    stratify=df_full_train["diabetes_binary"],
)


def to_feature_dicts(frame):
    features = frame.copy()
    features[categorical] = features[categorical].astype(str)
    return features.to_dict(orient="records")

features = categorical + numerical

print(f'doing validation')

def train_validation_model(df_train, df_val, features, params, num_boost_round=200):
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(to_feature_dicts(df_train[features]))
    y_train = df_train["diabetes_binary"].values
    X_val = dv.transform(to_feature_dicts(df_val[features]))
    y_val = df_val["diabetes_binary"].values

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(dtrain, "train"), (dval, "val")]

    model = xgb.train(
        params,
        dtrain,
        evals=watchlist,
        verbose_eval=10,
        num_boost_round=num_boost_round,
    )
    return dv, model, dval, y_val




def train_final_model(dv, df_full_train, df_test, features, params, num_boost_round=200):
    X_full_train = dv.transform(to_feature_dicts(df_full_train[features]))
    y_full_train = df_full_train["diabetes_binary"].values
    X_test = dv.transform(to_feature_dicts(df_test[features]))
    y_test = df_test["diabetes_binary"].values

    dtrain_full = xgb.DMatrix(X_full_train, label=y_full_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    watchlist = [(dtrain_full, "train"), (dtest, "test")]

    model = xgb.train(
        params,
        dtrain_full,
        evals=watchlist,
        verbose_eval=10,
        num_boost_round=num_boost_round,
    )
    return model, dtest, y_test


def report_auc(model, dmatrix, labels, description):
    auc = roc_auc_score(labels, model.predict(dmatrix))
    print(f"{description}: {auc:.4f}")


dv, validation_model, dval, y_val = train_validation_model(
    df_train, df_val, features, xgb_params
)
report_auc(validation_model, dval, y_val, "Validation ROC AUC (XGBoost)")

final_model, dtest, y_test = train_final_model(
    dv, df_full_train, df_test, features, xgb_params
)
report_auc(final_model, dtest, y_test, "AUC")


# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, final_model), f_out)

print(f'the model is saved to {output_file}')