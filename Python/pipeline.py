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

import time




# ============================== #
#           Functions            #
# ============================== #

def normalized_rmse(y_true, y_pred): # chiedere a Chiara se è ok usare sd() test. 
    stds = np.std(y_true, axis=0)
    rmse_per_outcome = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
    return np.mean(rmse_per_outcome / stds)

custom_scorer = make_scorer(normalized_rmse, greater_is_better=False)


# Set random seed for reproducibility
seed = 2026
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
    # Caricamento dei singoli CSV
    df_train = pd.read_csv(t_path)
    df_val = pd.read_csv(v_path)
    # Separazione Feature (X) e Target (y)
    feature_cols = [c for c in df_train.columns if c not in outcomes]
    X_tr, y_tr = df_train[feature_cols], df_train[outcomes]
    X_va, y_va = df_val[feature_cols], df_val[outcomes]
    # Concateniamo Train e Val di QUESTO specifico fold uno sotto l'altro
    X_fold = pd.concat([X_tr, X_va], axis=0)
    y_fold = pd.concat([y_tr, y_va], axis=0)
    X_chunks.append(X_fold)
    y_chunks.append(y_fold)
    # Calcolo degli indici assoluti nella futura matrice globale
    len_tr = len(X_tr)
    len_va = len(X_va)
    indici_train = np.arange(current_idx, current_idx + len_tr)
    indici_val = np.arange(current_idx + len_tr, current_idx + len_tr + len_va)
    # Salviamo la tupla (train, val) per questo fold
    custom_cv.append((indici_train, indici_val))
    # Incrementiamo il puntatore per il prossimo fold
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
# Esempio: n_estimators=1000, max_depth=10 → 1000 alberi profondi max 10 livelli
# su y shape (2000, 3) → ogni split minimizza MSE(mbi) + MSE(mrs) + MSE(tct) insieme
rf_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [10, 20, 50],
    'min_samples_leaf': [5, 10, 20],
    'max_features': ['sqrt', 0.33, 0.5]
}

# --- CatBoost MTL ---
# loss_function='MultiRMSE' → gradient calcolato congiuntamente su tutti e 3 gli outcome
catboost_grid = {
    'iterations': [300, 600, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [3, 4, 6],
    'l2_leaf_reg': [1, 3, 5, 10, 20],
    'loss_function': ['MultiRMSE']
}

# --- ElasticNet MTL ---
# MultiTaskElasticNet: penalità L2,1 condivisa → se alpha=0.1 azzera feature_età,
# la azzera per TUTTI e 3 gli outcome contemporaneamente
elasticnet_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
    'l1_ratio': [0, 0.25, 0.5, 0.75, 1.0]
}

# rfsrc_grid <- list(
#   ntree = c(500, 1000),
#   nodesize = c(1, 5, 10, 15),       # Controlla l'overfitting (dimensione minima nodo terminale)
#   mtry = c(                         # Feature valutate ad ogni split
#     floor(sqrt(ncol(data))), 
#     floor(ncol(data) / 3), 
#     floor(ncol(data) / 2)
#   ),
#   splitrule = c("mv.mse","mahalanobis")           # Regola di split multivariata (MSE)
# )

# nn_grid = {
#     # Architettura dei layer condivisi (Hard Sharing)
#     'shared_hidden_layers': [[64, 32], [128, 64], [256, 128, 64]],
#     'learning_rate': [1e-4, 1e-3, 5e-3],
#     'dropout_rate': [0.1, 0.2, 0.3],       # Regolarizzazione per strati nascosti
#     'batch_size': [32, 64, 128],
#     'epochs': [50, 100, 200]               # Spesso gestito con Early Stopping piuttosto che via grid
# }


# =========== #
# SINGLE-TASK #
# =========== #

# --- Random Forest STL ---
# Esempio: n_estimators=1000, max_depth=10 → 3 foreste indipendenti (una per outcome)
# rf_mbi splitta considerando solo MSE(mbi_t1), ignora mrs e tct
rf_stl_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [10, 20, 50],
    'min_samples_leaf': [5, 10, 20],
    'max_features': ['sqrt', 0.33, 0.5]
}

# --- CatBoost STL ---
# loss_function='RMSE' → gradient calcolato separatamente per ogni outcome
# Esempio: cb_mrs ottimizza solo RMSE(mrs_t1), non sa nulla di mbi e tct
catboost_stl_grid = {
    'iterations': [300, 600, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [3, 4, 6],
    'l2_leaf_reg': [1, 3, 5, 10, 20],
    'loss_function': ['RMSE']
}

# --- ElasticNet STL ---
# ElasticNet univariato: penalità indipendente per ogni outcome
# Esempio: alpha=0.1 può azzerare feature_età su mbi_t1 ma tenerla su mrs_t1
elasticnet_stl_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
    'l1_ratio': [0, 0.25, 0.5, 0.75, 1.0]
}


# ==================== #
#       PIPELINE       #
# ==================== #

# dobbiamo STD e normalizzare i target per evitare che scale diverse influenzino l'addestramento dei modelli multitask, 
# #specialmente quelli basati su gradient boosting e reti neurali. Per questo, possiamo utilizzare lo StandardScaler 
# di scikit-learn per standardizzare i target prima dell'addestramento e poi invertire la trasformazione sulle 
# predizioni finali.

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
# CatBoostRegressor con loss_function='MultiRMSE' supporta multi-output nativo
 
 
MVCatBoost = TransformedTargetRegressor(
    regressor=CatBoostRegressor(
        loss_function='MultiRMSE',
        random_seed=seed,
        verbose=0  
    ),
    transformer=StandardScaler()
)


# ==================== #
# GRID SEARCH WRAPPERS #
# (Multi-task)         #
# ==================== #
 
# Nota sui prefissi: i modelli sono wrappati in TransformedTargetRegressor.
# GridSearchCV vede i parametri come: regressor__<param>
# Esempio: RandomForestRegressor(n_estimators=...) → 'regressor__n_estimators'
 
rf_mtl_param_grid = {f'regressor__{k}': v for k, v in rf_grid.items()}
en_mtl_param_grid = {f'regressor__{k}': v for k, v in elasticnet_grid.items()}
cb_mtl_param_grid = {f'regressor__{k}': v for k, v in catboost_grid.items()
                     if k != 'loss_function'}  # loss_function è fixed, non va in grid
 
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


# =========== #
# SINGLE-TASK #
# =========== #
 
# Ogni modello STL è indipendente per outcome.
# Esempio RF STL: 
#   rf_mbi fitta su y['mbi_t1'] → split basati solo su mbi_t1
#   rf_mrs fitta su y['mrs_t1'] → split basati solo su mrs_t1
#   rf_tct fitta su y['tct_t1'] → split basati solo su tct_t1
#
# MultiOutputRegressor li wrappa insieme → possiamo passare y_global intero a GridSearchCV
# e usare custom_scorer che aggrega i 3 NRMSE in un unico score.
#
# Stack dei wrapper per ogni modello STL:
#   MultiOutputRegressor(          ← wrappa i 3 modelli, riceve y shape (n, 3)
#       TransformedTargetRegressor(    ← standardizza Y fold-by-fold per ogni outcome
#           regressor=<modello>
#       )
#   )
# Prefisso parametri grid: estimator__regressor__<param>
#   estimator__       → MultiOutputRegressor
#   regressor__       → TransformedTargetRegressor
#   <param>           → parametro del modello base
 
# --- Random Forest STL ---
STL_RF = MultiOutputRegressor(
    TransformedTargetRegressor(
        regressor=RandomForestRegressor(random_state=seed),
        transformer=StandardScaler()
    )
)
rf_stl_param_grid = {f'estimator__regressor__{k}': v for k, v in rf_stl_grid.items()}
 
# --- ElasticNet STL ---
# Univariate ElasticNet
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

# ==================== #
# GRID SEARCH WRAPPERS #
# (Single-task)         #
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
 

# ==================== #
#       FITTING        #
# ==================== #

mtl_models = {
    'MVRF':        GS_MVRF,
    'MTEN':        GS_MTEN,
    'MVCatBoost':  GS_MVCatBoost,
}

stl_models = {
    'STL_RF':  GS_STL_RF,
    'STL_EN':  GS_STL_EN,
    'STL_CB':  GS_STL_CB,
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
        print (f'Done in {elapsed/60:.1f} mins | NRMSE: {nrmse:.4f}')



# ==================== #
#       RESULTS        #
# ==================== #

print(f"{'Model':<15} {'Group':<8} {'Mean NRMSE':<15} {'Time (min)':<12}")
print(f"{'-'*50}")

for name, res in results.items():
    print(f"{name:<15} {res['group']:<8} {res['nrmse']:<15.4f} {res['time_min']:<12.1f}")

best_mtl = min((x for x in results if results[x]['group'] == 'MTL'), 
                key=lambda x: results[x]['nrmse'])
best_stl = min((x for x in results if results[x]['group'] == 'STL'), 
                key=lambda x: results[x]['nrmse'])

print(f"\n→ Best MTL: {best_mtl} (NRMSE={results[best_mtl]['nrmse']:.4f})")
print(f"→ Best STL: {best_stl} (NRMSE={results[best_stl]['nrmse']:.4f})")

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


print(f"  FINAL COMPARISON — RMSE per outcome (test set)")
print(f"{'Outcome':<12} {best_mtl:<15} {best_stl:<15} {'Migliore':<10}")
print(f"{'-'*50}")

for outcome, res in rmse_results.items():
    winner = best_mtl if res['MTL'] < res['STL'] else best_stl
    print(f"{outcome:<12} {res['MTL']:<15.4f} {res['STL']:<15.4f} {winner:<10}")

mean_rmse_mtl = np.mean([v['MTL'] for v in rmse_results.values()])
mean_rmse_stl = np.mean([v['STL'] for v in rmse_results.values()])
print(f"{'-'*50}")
print(f"{'Media':<12} {mean_rmse_mtl:<15.4f} {mean_rmse_stl:<15.4f}")