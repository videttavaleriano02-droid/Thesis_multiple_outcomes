# ============================================================================= #
#                           PREDICTION PIPELINE                                 #            
# ============================================================================= #

# imports
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

from sklearn.preprocessing import StandardScaler

from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import MultiTaskElasticNet
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor

import time


# ============================== #
#           Functions            #
# ============================== #

def normalized_rmse(y_true, y_pred): 
    stds = np.std(y_true, axis=0)
    rmse_per_outcome = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
    return np.mean(rmse_per_outcome / stds)

custom_scorer = make_scorer(normalized_rmse, greater_is_better=False)


# Set random seed for reproducibility
seed = 2727
np.random.seed(seed)

# ============================== #
#        Importing folds         #
# ============================== #
os.chdir("/Users/valerianovidetta/Desktop/Tesi/Dataset")

outcomes = ["mbi_t1", "mrs_t1", "tct_t1"]


train_files = ['folds_synthetic/train_syn_fold1.csv','folds_synthetic/train_syn_fold2.csv',
               'folds_synthetic/train_syn_fold3.csv','folds_synthetic/train_syn_fold4.csv',
               'folds_synthetic/train_syn_fold5.csv']

val_files = ['folds_imputed/val_fold1.csv','folds_imputed/val_fold2.csv',
             'folds_imputed/val_fold3.csv','folds_imputed/val_fold4.csv',
             'folds_imputed/val_fold5.csv']


X_chunks = []
y_chunks = []
custom_cv = []
current_idx = 0

for t_path, v_path in zip(train_files, val_files):
    df_train = pd.read_csv(t_path)
    df_val = pd.read_csv(v_path)
    # separate Feature (X) and Target (y)
    feature_cols = [c for c in df_train.columns if c not in outcomes]
    X_tr, y_tr = df_train[feature_cols], df_train[outcomes]
    X_va, y_va = df_val[feature_cols], df_val[outcomes]
    # Bind Train and Val of this fold one below the other
    X_fold = pd.concat([X_tr, X_va], axis=0)
    y_fold = pd.concat([y_tr, y_va], axis=0)
    X_chunks.append(X_fold)
    y_chunks.append(y_fold)
    # Absolute indeces for the global matrix
    len_tr = len(X_tr)
    len_va = len(X_va)
    indici_train = np.arange(current_idx, current_idx + len_tr)
    indici_val = np.arange(current_idx + len_tr, current_idx + len_tr + len_va)
    # Save the tuple of indices for this fold in the custom_cv list
    custom_cv.append((indici_train, indici_val))
    # go to next fold
    current_idx += (len_tr + len_va)


# Creazione dei dataset globali finali ordinati per fold
X_global = pd.concat(X_chunks, axis=0).reset_index(drop=True)
y_global = pd.concat(y_chunks, axis=0).reset_index(drop=True)


# ==================== #
#       GRIDS          #
# ==================== #

# ========== #
# MULTI-TASK #
# ========== #

# --- Random Forest MTL ---
rf_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [10, 20, 50],
    'min_samples_leaf': [2, 5, 10, 20],
    'max_features': ['sqrt', 0.33, 0.5]
}

# --- CatBoost MTL ---
catboost_grid = {
    'iterations': [300, 600, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [3, 4, 6],
    'l2_leaf_reg': [1, 3, 5, 10, 20],
    'loss_function': ['MultiRMSE']
}

# --- XGBoost MTL ---
#     'reg_alpha':     [0, 0.1, 0.5, 1.0],  # L1: weights of leaves pushed towards zero
xgboost_grid = {
    'n_estimators':  [300, 600, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth':     [3, 4, 6],
    'reg_lambda':    [1, 5, 10, 20],       # L2: penalises weights for big leaves
    'subsample':     [0.6, 0.8, 1.0],      # obs fraction for each tree
    'colsample_bytree': [0.6, 0.8, 1.0],   # feature fraction for each tree
}
# --- ElasticNet MTL ---
elasticnet_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 10, 100],
    'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
}

# =========== #
# SINGLE-TASK #
# =========== #

# --- Random Forest STL ---
rf_stl_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [10, 20, 50],
    'min_samples_leaf': [2, 5, 10, 20],
    'max_features': ['sqrt', 0.33, 0.5]
}

# --- CatBoost STL ---
catboost_stl_grid = {
    'iterations': [300, 600, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [3, 4, 6],
    'l2_leaf_reg': [1, 3, 5, 10, 20],
    'loss_function': ['RMSE']
}

# --- XGBoost STL ---
#     'reg_alpha':     [0, 0.1, 0.5, 1.0],  # L1: weights of leaves pushed towards zero

xgboost_stl_grid = {
    'n_estimators': [300, 600, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 6],
    'reg_lambda': [1, 3, 5, 10, 20],
    'subsample':     [0.6, 0.8, 1.0],     # obs fraction for each tree
    'colsample_bytree': [0.6, 0.8, 1.0],  # feature fraction for each tree
}

# --- ElasticNet STL ---
elasticnet_stl_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 10, 100],
    'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
}


# ==================== #
#       PIPELINE       #
# ==================== #

# We need to standardize and normalize the targets to prevent different scales from influencing 
# #the training of multitask models.
# # To do this, we can use scikit-learn's StandardScaler to standardize the targets before training and then apply
#  the inverse transformation to the final predictions

# Pipeline to scale features
pipeline_features = Pipeline([('scaler_x', StandardScaler())])

# ========== #
# MULTI-TASK
# ========== #

# -----------------
# RANDOM FOREST

MVRF = TransformedTargetRegressor(
    regressor=RandomForestRegressor(random_state=seed),
    transformer=StandardScaler() # <--- Forza la standardizzazione di Y dentro il fold!
)

# -----------------
# ELASTIC NET (MultiTask)

MTEN = TransformedTargetRegressor(
    regressor=MultiTaskElasticNet(max_iter=5000, random_state=seed),
    transformer=StandardScaler()
)

# -----------------
# CATBOOST (MultiTask)

MVCatBoost = TransformedTargetRegressor(
    regressor=CatBoostRegressor(
        loss_function='MultiRMSE',
        random_seed=seed,
        verbose=0  
    ),
    transformer=StandardScaler()
)

# -----------------
# XGBOOST (MultiTask)

MVXGB = TransformedTargetRegressor(
    regressor=XGBRegressor(
        tree_method='hist',           
        multi_strategy='multi_output_tree',
        random_state=seed,
        verbosity=0
    ),
    transformer=StandardScaler()
)


# ==================== #
# GRID SEARCH WRAPPERS #
# (Multi-task)         #
# ==================== #
 
# Prefix: models are wrapped in TransformedTargetRegressor.
# GridSearchCV sees the params as: regressor__<param>
# Example: RandomForestRegressor(n_estimators=...) → 'regressor__n_estimators'
 
rf_mtl_param_grid = {f'regressor__{k}': v for k, v in rf_grid.items()}
en_mtl_param_grid = {f'regressor__{k}': v for k, v in elasticnet_grid.items()}
cb_mtl_param_grid = {f'regressor__{k}': v for k, v in catboost_grid.items()
                     if k != 'loss_function'}  
xgb_mtl_param_grid = {f'regressor__{k}': v for k, v in xgboost_grid.items()}

 
GS_MVRF = GridSearchCV(
    estimator=MVRF,
    param_grid=rf_mtl_param_grid,
    cv=custom_cv,
    scoring=custom_scorer,
    refit=True,
    n_jobs=-1,
    verbose=1
)
 
GS_MTEN = GridSearchCV(
    estimator=MTEN,
    param_grid=en_mtl_param_grid,
    cv=custom_cv,
    scoring=custom_scorer,
    refit=True,
    n_jobs=-1,
    verbose=1
)
 
GS_MVCatBoost = GridSearchCV(
    estimator=MVCatBoost,
    param_grid=cb_mtl_param_grid,
    cv=custom_cv,
    scoring=custom_scorer,
    refit=True,
    n_jobs=-1,
    verbose=1
)

GS_MVXGB = GridSearchCV(
    estimator=MVXGB,
    param_grid=xgb_mtl_param_grid,
    cv=custom_cv,
    scoring=custom_scorer,
    refit=True,
    n_jobs=-1,
    verbose=1
)

# =========== #
# SINGLE-TASK #
# =========== #
 
#Each stl model is trained independently on each target.
# Example RF STL: 
#   rf_mbi fitta su y['mbi_t1'] -> split based only on mbi_t1
#   rf_mrs fitta su y['mrs_t1'] -> split based only on mrs_t1
#   rf_tct fitta su y['tct_t1'] -> split based only on tct_t1
#
# MultiOutputRegressor wraps them together -> we can pass y_global into GridSearchCV
# and use custom_scorer which aggregates the 3 NRMSE into a single score.
#
# Wrapper stack for each STL model:
#   MultiOutputRegressor(          <- wraps the 3 models, receives y shape (n, 3)
#       TransformedTargetRegressor(    <- standardizes Y fold-by-fold for each outcome
#           regressor=<modello>
#       )
#   )
# Prefix: estimator__regressor__<param>
#   estimator__       -> MultiOutputRegressor
#   regressor__       -> TransformedTargetRegressor
#   <param>           -> parametro del modello base
 
# --- Random Forest STL ---
STL_RF = MultiOutputRegressor(
    TransformedTargetRegressor(
        regressor=RandomForestRegressor(random_state=seed),
        transformer=StandardScaler()
    )
)
rf_stl_param_grid = {f'estimator__regressor__{k}': v for k, v in rf_stl_grid.items()}
 
# --- ElasticNet STL ---
STL_EN = MultiOutputRegressor(
    TransformedTargetRegressor(
        regressor=ElasticNet(max_iter=5000, random_state=seed),
        transformer=StandardScaler()
    )
)
en_stl_param_grid = {f'estimator__regressor__{k}': v for k, v in elasticnet_stl_grid.items()}
 
# --- CatBoost STL ---
STL_CB = MultiOutputRegressor(
    TransformedTargetRegressor(
        regressor=CatBoostRegressor(
            loss_function='RMSE',
            random_seed=seed,
            verbose=0
        ),
        transformer=StandardScaler()
    )
)
cb_stl_param_grid = {f'estimator__regressor__{k}': v for k, v in catboost_stl_grid.items()
                     if k != 'loss_function'}

STL_XGB = MultiOutputRegressor(
    TransformedTargetRegressor(
        regressor=XGBRegressor(
            tree_method='hist',
            random_state=seed,
            verbosity=0
        ),
        transformer=StandardScaler()
    )
)
xgb_stl_param_grid = {f'estimator__regressor__{k}': v for k, v in xgboost_stl_grid.items()}


# ==================== #
# GRID SEARCH WRAPPERS #
# (Single-task)        #
# ==================== #

GS_STL_RF = GridSearchCV(
    estimator=STL_RF,
    param_grid=rf_stl_param_grid,
    cv=custom_cv,
    scoring=custom_scorer,
    refit=True,
    n_jobs=-1,
    verbose=1
)
 
GS_STL_EN = GridSearchCV(
    estimator=STL_EN,
    param_grid=en_stl_param_grid,
    cv=custom_cv,
    scoring=custom_scorer,
    refit=True,
    n_jobs=-1,
    verbose=1
)
 
GS_STL_CB = GridSearchCV(
    estimator=STL_CB,
    param_grid=cb_stl_param_grid,
    cv=custom_cv,
    scoring=custom_scorer,
    refit=True,
    n_jobs=-1,
    verbose=1
)

GS_STL_XGB = GridSearchCV(
    estimator=STL_XGB,
    param_grid=xgb_stl_param_grid,
    cv=custom_cv,
    scoring=custom_scorer,
    refit=True,
    n_jobs=-1,
    verbose=1
)

 

# ==================== #
#       FITTING        #
# ==================== #

mtl_models = {
    'MVRF':       GS_MVRF,
    'MTEN':       GS_MTEN,
    'MVCatBoost': GS_MVCatBoost,
    'MVXGB':      GS_MVXGB,   
}

stl_models = {
    'STL_RF':  GS_STL_RF,
    'STL_EN':  GS_STL_EN,
    'STL_CB':  GS_STL_CB,
    'STL_XGB': GS_STL_XGB,  
}

results = {}


for group_name, group in [ ('MTL', mtl_models), ('STL', stl_models) ]:
    print( f'Fitting {group_name} models')

    for model_name, gs in group.items():
        print(f'Fitting {model_name}...')
        t0 = time.time()
        gs.fit(X_global, y_global)
        elapsed = time.time() - t0
        
        nrmse = -gs.best_score_

        results[model_name] = {
            'best_params': gs.best_params_,
            'nrmse': nrmse,
            'group': group_name,
            'time_min': elapsed / 60
        }
        print (f'Done in {elapsed/60:.3f} mins | NRMSE: {nrmse:.4f}')



# ==================== #
#       RESULTS        #
# ==================== #

print(f"{'-'*50}")
print(f"{'Model':<15} {'Group':<8} {'Mean NRMSE':<15} {'Time (min)':<12}")
print(f"{'-'*50}")

for name, res in results.items():
    print(f"{name:<15} {res['group']:<8} {res['nrmse']:<15.4f} {res['time_min']:<12.3f}")

best_mtl = min((x for x in results if results[x]['group'] == 'MTL'), 
                key=lambda x: results[x]['nrmse'])
best_stl = min((x for x in results if results[x]['group'] == 'STL'), 
                key=lambda x: results[x]['nrmse'])

print(f"\n→ Best MTL: {best_mtl} (NRMSE={results[best_mtl]['nrmse']:.5f})")
print(f"→ Best STL: {best_stl} (NRMSE={results[best_stl]['nrmse']:.5f})")

# GridsearchCV object of best MTL and STL models
best_mtl_gs = mtl_models[best_mtl]
best_stl_gs = stl_models[best_stl]



# =========================== #
#       FINAL COMPARISON      #
# =========================== #

development = pd.read_csv('development_synthetic/development_syn.csv')
test = pd.read_csv('hold_out_imputed/test.csv')

feature_cols = [c for c in development.columns if c not in outcomes]

X_dev  = development[feature_cols]
y_dev  = development[outcomes]
X_test = test[feature_cols]
y_test = test[outcomes]

# training best models
best_mtl_gs.best_estimator_.fit(X_dev, y_dev)
best_stl_gs.best_estimator_.fit(X_dev, y_dev)

# predictions on test set
y_pred_mtl = best_mtl_gs.best_estimator_.predict(X_test) 
y_pred_stl = best_stl_gs.best_estimator_.predict(X_test) 

rmse_results = {}
for i, outcome in enumerate(outcomes):
    rmse_mtl = np.sqrt(np.mean((y_test[outcome].values - y_pred_mtl[:, i])**2))
    rmse_stl = np.sqrt(np.mean((y_test[outcome].values - y_pred_stl[:, i])**2))
    rmse_results[outcome] = {'MTL': rmse_mtl, 'STL': rmse_stl}


print(f"  FINAL COMPARISON — RMSE by outcome (test set)")
print(f"{'-'*50}")
print(f"{'Outcome':<12} {best_mtl:<15} {best_stl:<15} {'Best':<10}")
print(f"{'-'*50}")

for outcome, res in rmse_results.items():
    winner = best_mtl if res['MTL'] < res['STL'] else best_stl
    print(f"{outcome:<12} {res['MTL']:<15.4f} {res['STL']:<15.4f} {winner:<10}")

mean_rmse_mtl = np.mean([v['MTL'] for v in rmse_results.values()])
mean_rmse_stl = np.mean([v['STL'] for v in rmse_results.values()])
print(f"{'-'*50}")
print(f"{'Mean':<12} {mean_rmse_mtl:<15.4f} {mean_rmse_stl:<15.4f}")