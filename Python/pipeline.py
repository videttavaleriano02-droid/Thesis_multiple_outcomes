# ============================================================================= #
#                           PREDICTION PIPELINE                                 #            
# ============================================================================= #

# imports
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import MultiTaskElasticNet



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
# Multi-task #
# ========== #

catboost_grid = {
    'iterations': [500, 1000, 1500],       # # trees
    'learning_rate': [0.01, 0.05, 0.1],    
    'depth': [4, 6, 8, 10],                
    'l2_leaf_reg': [1, 3, 5, 10],          # Regularization L2
    'loss_function': ['MultiRMSE']         # Clearly FIXED
}

rf_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20, 30],       # 'None' makes nodes expand untill pure leaves
    'min_samples_split': [2, 5, 10],       # To regularize the tree
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.33] # 0.33 1/3 of features is ok
}

elasticnet_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0], # Total penalty (L1 + L2)
    'l1_ratio': [0, 0.3, 0.5, 0.7, 0.9, 1] #0 = Ridge, 1 =  Lasso
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
# Single-task #
# =========== #


catboost_stl_grid = {
    'iterations': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 10],
    'loss_function': ['RMSE']            
}

rf_stl_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.33]
}

elasticnet_stl_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0, 0.3, 0.5, 0.7, 0.9, 1]
}

# nn_stl_grid = {
#     'hidden_layers': [[64, 32], [128, 64], [256, 128, 64]], # Strati standard sequenziali
#     'learning_rate': [1e-4, 1e-3, 5e-3],
#     'dropout_rate': [0.1, 0.2, 0.3],
#     'batch_size': [32, 64, 128],
#     'epochs': [50, 100, 200]
# }




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


