
# imports
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import MultiTaskElasticNet
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor

from sklearn.base import clone

import time

# ============================== #
#           Functions            #
# ============================== #

def normalized_rmse(y_true, y_pred): 
    stds = np.std(y_true, axis=0)
    rmse_per_outcome = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
    return np.mean(rmse_per_outcome / stds)

custom_scorer = make_scorer(normalized_rmse, greater_is_better=False)

def print_fold_errors(gs, model_label, n_folds=5, mtl=True):
    """
    Print fold errors
    gs        : GridSearchCV object
    model_label: String, model
    mtl       : if TRUE uses NRMSE, else RMSE
    """
    idx = gs.best_index_
    cv_res = gs.cv_results_
    metric = "NRMSE" if mtl else "RMSE"

    print(f"\n  Fold-level errors for best {model_label}:")
    print(f"  {'Fold':<8} {'Train ' + metric:<18} {'Val ' + metric:<18}")
    print(f"  {'-'*44}")

    for fold in range(n_folds):
        train_score = -cv_res[f'split{fold}_train_score'][idx]
        val_score   = -cv_res[f'split{fold}_test_score'][idx]
        print(f"  Fold {fold+1:<3}  {train_score:<18.4f} {val_score:<18.4f}")

    mean_train = -cv_res['mean_train_score'][idx]
    mean_val   = -cv_res['mean_test_score'][idx]
    print(f"  {'Mean':<8} {mean_train:<18.4f} {mean_val:<18.4f}")

def best_cv_summary(gs):
    """Medie train/val della combinazione migliore."""
    idx = gs.best_index_
    res = gs.cv_results_
    return {
        'train': -res['mean_train_score'][idx],
        'val':   -res['mean_test_score'][idx],
    }

