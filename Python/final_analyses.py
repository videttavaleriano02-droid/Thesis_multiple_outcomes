# ============================================================================= #
#                           RESULTS: PLOTS AND ANALYSES                         #            
# ============================================================================= #

import pandas as pd
import joblib


# -------

rmse = pd.read_csv("results/rmse_comparison.csv")
predictions = pd.read_csv("results/predictions_test.csv")
residuals = pd.read_csv("results/residuals_test.csv")


# ---- best mtl model ---- #

best_params_mtl = pd.read_csv("results/best_params_mtl.csv")
best_mtl_name = best_params_mtl.loc[best_params_mtl['nrmse_cv'].idxmin(), 'model']

# --- winner by outcome
print(rmse[['outcome', 'winner']])

