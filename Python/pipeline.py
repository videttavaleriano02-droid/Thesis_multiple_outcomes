# ============================================================================= #
#                           PREDICTION PIPELINE                                 #            
# ============================================================================= #

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
    'max_depth': [5, 10, 15],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [2, 5, 10, 20],
    'max_features': ['sqrt', 0.33, 0.5]
}

# --- CatBoost MTL ---
catboost_grid = {
    'iterations': [200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [5, 10, 15],
    'l2_leaf_reg': [1, 3, 5, 10, 20],
    'loss_function': ['MultiRMSE']
}

# --- XGBoost MTL ---
#     'reg_alpha':     [0, 0.1, 0.5, 1.0],  # L1: weights of leaves pushed towards zero
xgboost_grid = {
    'n_estimators':  [200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth':     [5, 10, 15],
    'reg_lambda':    [1, 5, 10, 20],       # L2: penalises weights for big leaves
    'subsample':     [0.6, 0.8, 1.0],      # obs fraction for each tree
    'colsample_bytree': [0.11, 0.33, 0.5],   # feature fraction for each tree
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
    'max_depth': [5, 10, 15],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [2, 5, 10, 20],
    'max_features': ['sqrt', 0.33, 0.5]
}

# --- CatBoost STL ---
catboost_stl_grid = {
    'iterations': [200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [5, 10, 15],
    'l2_leaf_reg': [1, 3, 5, 10, 20],
    'loss_function': ['RMSE']
}

# --- XGBoost STL ---
#     'reg_alpha':     [0, 0.1, 0.5, 1.0],  # L1: weights of leaves pushed towards zero

xgboost_stl_grid = {
    'n_estimators': [200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 10, 15],
    'reg_lambda': [1, 3, 5, 10, 20],
    'subsample':     [0.6, 0.8, 1.0],     # obs fraction for each tree
    'colsample_bytree': [0.11, 0.33, 0.5],  # feature fraction for each tree
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
# the training of multitask models.
# To do this, we can use scikit-learn's StandardScaler to standardize the targets before training and then apply
#  the inverse transformation to the final predictions

# ========== #
# MULTI-TASK
# ========== #

# -----------------
# RANDOM FOREST

MVRF_pipe = Pipeline([
    ('scaler_x', StandardScaler()),
    ('MVRF', TransformedTargetRegressor(
        regressor=RandomForestRegressor(random_state=seed),
        transformer=StandardScaler()
    )
  )
])

# -----------------
# ELASTIC NET (MultiTask)

MTEN_pipe = Pipeline([
    ('scaler_x', StandardScaler()),
    ('MTEN',TransformedTargetRegressor(
    regressor=MultiTaskElasticNet(max_iter=5000, random_state=seed),
    transformer=StandardScaler() 
    ) 
) 
])

# -----------------
# CATBOOST (MultiTask)

MVCatBoost_pipe = Pipeline( [
    ('scaler_x', StandardScaler()),
    ('MVCatBoost', TransformedTargetRegressor(
    regressor=CatBoostRegressor(
        loss_function='MultiRMSE',
        random_seed=seed,
        verbose=0  
    ),
    transformer=StandardScaler()
    ) 
  ) 
])

# -----------------
# XGBOOST (MultiTask)

MVXGB_pipe = Pipeline( [
    ('scaler_x', StandardScaler()),
    ('MVXGB', TransformedTargetRegressor(
    regressor=XGBRegressor(
        tree_method='hist',           
        multi_strategy='multi_output_tree',
        random_state=seed,
        verbosity=0
    ),
    transformer=StandardScaler()
    ) 
  ) 
])


# ==================== #
# GRID SEARCH WRAPPERS #
# (Multi-task)         #
# ==================== #
 
# Prefix: models are wrapped in TransformedTargetRegressor.
# GridSearchCV sees the params as: regressor__<param>
# Example: RandomForestRegressor(n_estimators=...) → 'regressor__n_estimators'

rf_mtl_param_grid  = {f'MVRF__regressor__{k}': v for k, v in rf_grid.items()}
en_mtl_param_grid  = {f'MTEN__regressor__{k}': v for k, v in elasticnet_grid.items()}
cb_mtl_param_grid  = {f'MVCatBoost__regressor__{k}': v for k, v in catboost_grid.items()
                      if k != 'loss_function'}
xgb_mtl_param_grid = {f'MVXGB__regressor__{k}': v for k, v in xgboost_grid.items()}

GS_MVRF = GridSearchCV(
    estimator=MVRF_pipe,
    param_grid=rf_mtl_param_grid,
    cv=custom_cv,
    scoring=custom_scorer,
    refit=True,
    n_jobs=-1,
    verbose=1
)
 
GS_MTEN = GridSearchCV(
    estimator=MTEN_pipe,
    param_grid=en_mtl_param_grid,
    cv=custom_cv,
    scoring=custom_scorer,
    refit=True,
    n_jobs=-1,
    verbose=1
)
 
GS_MVCatBoost = GridSearchCV(
    estimator=MVCatBoost_pipe,
    param_grid=cb_mtl_param_grid,
    cv=custom_cv,
    scoring=custom_scorer,
    refit=True,
    n_jobs=-1,
    verbose=1
)

GS_MVXGB = GridSearchCV(
    estimator=MVXGB_pipe,
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
#   rf_mbi fits on y['mbi_t1'] -> split based only on mbi_t1
#   rf_mrs fits on y['mrs_t1'] -> split based only on mrs_t1
#   rf_tct fits on y['tct_t1'] -> split based only on tct_t1

# Prefix: estimator__regressor__<param>
#   estimator__       -> MultiOutputRegressor
#   regressor__       -> TransformedTargetRegressor
#   <param>           -> base model parameter
 
# --- Random Forest STL ---
STL_RF_pipe = Pipeline([
    ('scaler_x', StandardScaler()),
    ('STL_RF', TransformedTargetRegressor(
        regressor=RandomForestRegressor(random_state=seed),
        transformer=StandardScaler()
    )
    )
])

# --- ElasticNet STL ---
STL_EN_pipe = Pipeline([
    ('scaler_x', StandardScaler()),
    ('STL_EN', TransformedTargetRegressor(
        regressor=ElasticNet(max_iter=5000, random_state=seed),
        transformer=StandardScaler()
    )
    )
])

# --- CatBoost STL ---
STL_CB_pipe = Pipeline([
    ('scaler_x', StandardScaler()),
    ('STL_CB', TransformedTargetRegressor(
        regressor=CatBoostRegressor(loss_function='RMSE', random_seed=seed, verbose=0),
        transformer=StandardScaler()
    )
    )
])

# --- XGBoost STL ---
STL_XGB_pipe = Pipeline([
    ('scaler_x', StandardScaler()),
    ('STL_XGB', TransformedTargetRegressor(
        regressor=XGBRegressor(tree_method='hist', random_state=seed, verbosity=0),
        transformer=StandardScaler()
    )
    )
])

rf_stl_param_grid  = {f'STL_RF__regressor__{k}': v for k, v in rf_stl_grid.items()}
en_stl_param_grid  = {f'STL_EN__regressor__{k}': v for k, v in elasticnet_stl_grid.items()}
cb_stl_param_grid  = {f'STL_CB__regressor__{k}': v for k, v in catboost_stl_grid.items()
                      if k != 'loss_function'}
xgb_stl_param_grid = {f'STL_XGB__regressor__{k}': v for k, v in xgboost_stl_grid.items()}

# ==================== #
# GRID SEARCH WRAPPERS #
# (Single-task)        #
# ==================== #

stl_grids = {}
for outcome in outcomes:
    stl_grids[(outcome, 'RF')]  = GridSearchCV(estimator=clone(STL_RF_pipe),  param_grid=rf_stl_param_grid, 
                                                cv=custom_cv, scoring='neg_root_mean_squared_error', refit=True, n_jobs=-1, verbose=1)
    
    stl_grids[(outcome, 'EN')]  = GridSearchCV(estimator=clone(STL_EN_pipe),  param_grid=en_stl_param_grid, 
                                                cv=custom_cv, scoring='neg_root_mean_squared_error', refit=True, n_jobs=-1, verbose=1)
    
    stl_grids[(outcome, 'CB')]  = GridSearchCV(estimator=clone(STL_CB_pipe),  param_grid=cb_stl_param_grid,  
                                               cv=custom_cv, scoring='neg_root_mean_squared_error', refit=True, n_jobs=-1, verbose=1)
    
    stl_grids[(outcome, 'XGB')] = GridSearchCV(estimator=clone(STL_XGB_pipe), param_grid=xgb_stl_param_grid, 
                                               cv=custom_cv, scoring='neg_root_mean_squared_error', refit=True, n_jobs=-1, verbose=1)

 

# ==================== #
#       FITTING        #
# ==================== #

# ========== #
#    MTL      #
# ========== #

mtl_models = {
    'MVRF':       GS_MVRF,
    'MTEN':       GS_MTEN,
    'MVCatBoost': GS_MVCatBoost,
    'MVXGB':      GS_MVXGB,
}

mtl_results = {}

print('Fitting MTL models...')
for model_name, gs in mtl_models.items():
    print(f'Fitting {model_name}...')
    t0 = time.time()
    gs.fit(X_global, y_global)
    elapsed = time.time() - t0

    nrmse = -gs.best_score_
    mtl_results[model_name] = {
        'best_params': gs.best_params_,
        'nrmse': nrmse,
        'time_min': elapsed / 60
    }
    print(f'Done in {elapsed/60:.3f} mins | NRMSE: {nrmse:.4f}')

# ========== #
#    STL      #
# ========== #

stl_results = {}

print('Fitting STL models...')
for (outcome, model_name), gs in stl_grids.items():
    print(f'Fitting {model_name} on {outcome}...')
    t0 = time.time()
    gs.fit(X_global, y_global[outcome])
    elapsed = time.time() - t0

    rmse = -gs.best_score_
    stl_results[(outcome, model_name)] = {
        'best_params': gs.best_params_,
        'rmse': rmse,
        'time_min': elapsed / 60
    }
    print(f'Done in {elapsed/60:.3f} mins | RMSE: {rmse:.4f}')

# ==================== #
#       RESULTS        #
# ==================== #

# --- MTL ---
print(f"{'-'*50}")
print(f"{'Model':<15} {'NRMSE':<15} {'Time (min)':<12}")
print(f"{'-'*50}")

for name, res in mtl_results.items():
    print(f"{name:<15} {res['nrmse']:<15.4f} {res['time_min']:<12.3f}")

best_mtl = min(mtl_results, key=lambda x: mtl_results[x]['nrmse'])
print(f"\n→ Best MTL: {best_mtl} (NRMSE={mtl_results[best_mtl]['nrmse']:.5f})")
best_mtl_gs = mtl_models[best_mtl]

# --- STL ---
print(f"\n{'-'*50}")
print(f"{'Outcome':<12} {'Model':<10} {'RMSE':<15} {'Time (min)':<12}")
print(f"{'-'*50}")

for (outcome, model_name), res in stl_results.items():
    print(f"{outcome:<12} {model_name:<10} {res['rmse']:<15.4f} {res['time_min']:<12.3f}")

# miglior modello per ogni outcome
best_stl_per_outcome = {}
for outcome in outcomes:
    best_model = min(['RF', 'EN', 'CB', 'XGB'],
                     key=lambda m: stl_results[(outcome, m)]['rmse'])
    best_stl_per_outcome[outcome] = (best_model, stl_grids[(outcome, best_model)])
    print(f"\n→ Best STL for {outcome}: {best_model} (RMSE={stl_results[(outcome, best_model)]['rmse']:.5f})")

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

# training best MTL model on development
mtl_pipe_map = {
    'MVRF': MVRF_pipe, 'MTEN': MTEN_pipe,
    'MVCatBoost': MVCatBoost_pipe, 'MVXGB': MVXGB_pipe
}

best_mtl_model = clone(mtl_pipe_map[best_mtl])
best_mtl_model.set_params(**best_mtl_gs.best_params_)
best_mtl_model.fit(X_dev, y_dev)
y_pred_mtl = best_mtl_model.predict(X_test)

# training best STL model per outcome on development
stl_pipe_map = {
    'RF': STL_RF_pipe, 'EN': STL_EN_pipe,
    'CB': STL_CB_pipe, 'XGB': STL_XGB_pipe
}
y_pred_stl = np.zeros((len(X_test), len(outcomes)))
for i, outcome in enumerate(outcomes):
    best_model_name, best_gs = best_stl_per_outcome[outcome]
    best_stl_model = clone(stl_pipe_map[best_model_name])
    best_stl_model.set_params(**best_gs.best_params_)
    best_stl_model.fit(X_dev, y_dev[outcome])
    y_pred_stl[:, i] = best_stl_model.predict(X_test)

# RMSE by outcome
rmse_results = {}
for i, outcome in enumerate(outcomes):
    rmse_mtl = np.sqrt(np.mean((y_test[outcome].values - y_pred_mtl[:, i])**2))
    rmse_stl = np.sqrt(np.mean((y_test[outcome].values - y_pred_stl[:, i])**2))
    rmse_results[outcome] = {'MTL': rmse_mtl, 'STL': rmse_stl}

# ----- Printing -----
print(f"  FINAL COMPARISON — RMSE by outcome (test set)")
print(f"{'-'*60}")
print(f"{'Outcome':<12} {best_mtl:<15} {'Best STL':<20} {'Best':<10}")
print(f"{'-'*60}")

for outcome, res in rmse_results.items():
    best_model_name = best_stl_per_outcome[outcome][0]
    winner = best_mtl if res['MTL'] < res['STL'] else f"{best_model_name}"
    print(f"{outcome:<12} {res['MTL']:<15.4f} {res['STL']:<20.4f} {winner:<10}")

mean_rmse_mtl = np.mean([v['MTL'] for v in rmse_results.values()])
mean_rmse_stl = np.mean([v['STL'] for v in rmse_results.values()])
print(f"{'-'*60}")
print(f"{'Mean':<12} {mean_rmse_mtl:<15.4f} {mean_rmse_stl:<20.4f}")

# ========================== #
os.makedirs("results", exist_ok=True)

# ========================== #
#       SAVING CV RESULTS    #
# ========================== #

# --- MTL: cv_results_ for each model ---

mtl_cv_dfs = []
for name, gs in mtl_models.items():
    df = pd.DataFrame(gs.cv_results_)
    df.insert(0, 'model', name) 
    mtl_cv_dfs.append(df)

pd.concat(mtl_cv_dfs, axis=0).to_csv("results/cv_results_mtl.csv", index=False)

# --- MTL: best params + score by model
best_params_rows = []
for name, res in mtl_results.items():
    row = {'model': name, 'nrmse_cv': res['nrmse'], 'time_min': res['time_min']}
    row.update(res['best_params'])  # hyperparameters are columns
    best_params_rows.append(row)

pd.DataFrame(best_params_rows).to_csv("results/best_params_mtl.csv", index=False)

# --- STL: cv_results_ for each (outcome, model) ---
stl_cv_dfs = []
for (outcome, model_name), gs in stl_grids.items():
    df = pd.DataFrame(gs.cv_results_)
    df.insert(0, 'outcome', outcome)
    df.insert(1, 'model', model_name)
    stl_cv_dfs.append(df)

pd.concat(stl_cv_dfs, axis=0).to_csv("results/cv_results_stl.csv", index=False)

# --- STL: best params + score by (outcome, model) ---
stl_best_rows = []
for (outcome, model_name), res in stl_results.items():
    row = {'outcome': outcome, 'model': model_name,
           'rmse_cv': res['rmse'], 'time_min': res['time_min']}
    row.update(res['best_params'])
    stl_best_rows.append(row)

pd.DataFrame(stl_best_rows).to_csv("results/best_params_stl.csv", index=False)


# ================================ #
#      SAVING FINAL RESULTS        #
# ================================ #

# --- Predictions on test set (MTL vs STL) --- #

pred_df = y_test.copy().reset_index(drop=True)
pred_df.columns = [f"{c}_true" for c in outcomes]  

for i, outcome in enumerate(outcomes):
    pred_df[f"{outcome}_pred_mtl"] = y_pred_mtl[:, i]
    pred_df[f"{outcome}_pred_stl"] = y_pred_stl[:, i]

pred_df.to_csv("results/predictions_test.csv", index=False)

# --- RMSE by outcome (MTL vs STL) ---
rmse_rows = []
for outcome, res in rmse_results.items():
    best_stl_name = best_stl_per_outcome[outcome][0]
    rmse_rows.append({
        'outcome':       outcome,
        f'rmse_{best_mtl}': res['MTL'],  
        f'rmse_{best_stl_name}': res['STL'],
        'winner':        best_mtl if res['MTL'] < res['STL'] else best_stl_name
    })

pd.DataFrame(rmse_rows).to_csv("results/rmse_comparison.csv", index=False)

# residuals --
resid_df = y_test.copy().reset_index(drop=True)
resid_df.columns = [f"{c}_true" for c in outcomes]

for i, outcome in enumerate(outcomes):
    resid_df[f"{outcome}_resid_mtl"] = y_test[outcome].values - y_pred_mtl[:, i]
    resid_df[f"{outcome}_resid_stl"] = y_test[outcome].values - y_pred_stl[:, i]

resid_df.to_csv("results/residuals_test.csv", index=False)

# ------ Saving models ------ #
joblib.dump(best_mtl_model, "results/best_mtl_model.pkl")

stl_models_to_save = {
    outcome: best_stl_per_outcome[outcome][1].best_estimator_
    for outcome in outcomes
}
joblib.dump(stl_models_to_save, "results/best_stl_models.pkl")
