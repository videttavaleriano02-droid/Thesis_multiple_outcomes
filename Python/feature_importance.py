# ============================================================================= #
#                  SHAP COMPARISON: MULTI-TASK vs SINGLE-TASK                   #
# ============================================================================= #
#
# Objective: compare interpretability (via SHAP) of the best
# multi-task (MTL, trained jointly on all 3 outcomes) model with the best
# single-task (STL, one per outcome) models, in order to extract clinical
# insights about which features drive each outcome and whether MTL and STL
# "see" the same patterns.
#
# Prerequisites: pipeline.py must have already been executed to completion,
# so that the following files are available:
#   results/best_mtl_model.pkl      -> Pipeline (scaler + TransformedTargetRegressor)
#   results/best_stl_models.pkl     -> dict {outcome: Pipeline}
#   development_synthetic/development_syn.csv
#   hold_out_imputed/test.csv
#
# Output (in results/shap/):
#   - feature_importance_comparison.csv  (mean|SHAP| and rank per feature/outcome/model)
#   - agreement_metrics.csv              (Spearman + Jaccard top-k between MTL and STL)
#   - beeswarm_<outcome>_mtl.png / _stl.png
#   - barplot_<outcome>_comparison.png


import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ==================== #
#       CONFIG         #
# ==================== #

os.chdir("/Users/valerianovidetta/Desktop/Tesi/Dataset")

outcomes = ["mbi_t1", "mrs_t1", "tct_t1"]
SEED = 2727
np.random.seed(SEED)

n_background = 200   # dimension of the background dataset for SHAP (subset of development set)
top_K = 10            # How many features to consider for Jaccard similarity between MTL and STL top-K

color_MTL = "#2F5D50"  
color_STL = "#A3C9A8"

out_dir = "results/shap"
os.makedirs(out_dir, exist_ok=True)

# ==================== #
#     LOAD DATA        #
# ==================== #

development = pd.read_csv("development_synthetic/development_syn.csv")
test = pd.read_csv("hold_out_imputed/test.csv")

feature_cols = [c for c in development.columns if c not in outcomes]
X_dev = development[feature_cols].reset_index(drop=True)
X_test = test[feature_cols].reset_index(drop=True)

# ==================== #
#    LOAD MODELS       #
# ==================== #

best_mtl_model = joblib.load(os.path.join("results", "best_mtl_model.pkl"))
best_stl_models = joblib.load(os.path.join("results", "best_stl_models.pkl"))
# best_stl_models: dict {outcome: fitted Pipeline}   

print(f"MTL model: {type(best_mtl_model.named_steps[[k for k in best_mtl_model.named_steps if k != 'scaler_x'][0]].regressor_).__name__}")
for outcome, mdl in best_stl_models.items():
    step = [k for k in mdl.named_steps if k != 'scaler_x'][0]
    print(f"STL model [{outcome}]: {type(mdl.named_steps[step].regressor_).__name__}")

# ==================== #
#      HELPERS         #
# ==================== #

TREE_MODELS = {"RandomForestRegressor", "XGBRegressor", "CatBoostRegressor"}
LINEAR_MODELS = {"ElasticNet", "MultiTaskElasticNet"}


def unwrap_pipeline(pipeline):
    """
    Estracts (regressor_fitted, scaler, model_type_name) from a Pipeline of the form
    [('scaler_x', StandardScaler()), (name, TransformedTargetRegressor(...))]
    """
    scaler = pipeline.named_steps["scaler_x"]
    step_name = [n for n in pipeline.named_steps if n != "scaler_x"][0]
    ttr = pipeline.named_steps[step_name]
    regressor = ttr.regressor_  
    model_type = type(regressor).__name__
    return regressor, scaler, model_type


def linear_shap_values(regressor, X_background_scaled, X_explain_scaled, multitask):
    """
    Exact SHAP values for linear models: shap_ij = coef_j * (x_ij - mean_j).
    Computed manually to avoid ambiguity in shape of shap.LinearExplainer
    for multi-output models (MultiTaskElasticNet).
    Returns array with shape (n_samples, n_features) or (n_outputs, n_samples, n_features).
    """
    baseline = X_background_scaled.mean(axis=0)  # (n_features,)
    diff = X_explain_scaled - baseline  # (n_samples, n_features)

    if multitask:
        coef = regressor.coef_  # (n_targets, n_features)
        sv = np.stack([diff * coef[k] for k in range(coef.shape[0])], axis=0)
        # shape: (n_outputs, n_samples, n_features)
    else:
        coef = regressor.coef_  # (n_features,)
        sv = diff * coef  # (n_samples, n_features)
    return sv


def compute_shap(pipeline, X_background, X_explain, is_mtl):
    """
    Computes SHAP values on the Pipeline's regressor, in the scaled space
    Returns sv with shape:
      - (n_outputs, n_samples, n_features) if is_mtl=True
      - (n_samples, n_features)            if is_mtl=False
    """
    regressor, scaler, model_type = unwrap_pipeline(pipeline)
    Xb = scaler.transform(X_background)
    Xe = scaler.transform(X_explain)

    if model_type in TREE_MODELS:
        explainer = shap.TreeExplainer(regressor)
        explanation = explainer(Xe)
        sv = explanation.values
        if is_mtl:
            # (n_samples, n_features, n_outputs)
            # normalizes: (n_outputs, n_samples, n_features)
            if sv.ndim == 3:
                sv = np.transpose(sv, (2, 0, 1))
            else:
                raise ValueError(
                    f"Expected shape multi-output 3D for {model_type}, found {sv.shape}. "
                    "Check the version of shap installed."
                )
    elif model_type in LINEAR_MODELS:
        sv = linear_shap_values(regressor, Xb, Xe, multitask=is_mtl)
    else:
        raise ValueError(f"Unhandled model type: {model_type}")

    return sv, model_type


# ==================== #
#   COMPUTE SHAP: MTL  #
# ==================== #

X_background = X_dev.sample(n=min(n_background, len(X_dev)), random_state=SEED)

print("\n Computing SHAP for MTL")
sv_mtl_all, mtl_type = compute_shap(best_mtl_model, X_background, X_test, is_mtl=True)
# sv_mtl_all shape: (n_outputs, n_samples, n_features), order of outcomes = `outcomes`
assert sv_mtl_all.shape[0] == len(outcomes), (
    f"Expected {len(outcomes)} output, found {sv_mtl_all.shape[0]}. "
    "Verify that the order of targets matches `outcomes`."
)

# ==================== #
#   COMPUTE SHAP: STL  #
# ==================== #

sv_stl = {}
stl_types = {}
for outcome in outcomes:
    print(f"Compute SHAP for STL model [{outcome}]...")
    sv, model_type = compute_shap(best_stl_models[outcome], X_background, X_test, is_mtl=False)
    sv_stl[outcome] = sv  # shape (n_samples, n_features)
    stl_types[outcome] = model_type

# ==================== #
#  FEATURE IMPORTANCE  #
#  TABLE + PLOTS       #
# ==================== #

rows = []
agreement_rows = []

for i, outcome in enumerate(outcomes):
    sv_mtl_outcome = sv_mtl_all[i]          # (n_samples, n_features)
    sv_stl_outcome = sv_stl[outcome]        # (n_samples, n_features)

    mean_abs_mtl = np.abs(sv_mtl_outcome).mean(axis=0)
    mean_abs_stl = np.abs(sv_stl_outcome).mean(axis=0)

    imp_mtl = pd.Series(mean_abs_mtl, index=feature_cols).sort_values(ascending=False)
    imp_stl = pd.Series(mean_abs_stl, index=feature_cols).sort_values(ascending=False)

    for feat in feature_cols:
        rows.append({
            "outcome": outcome, "feature": feat, "model": "MTL",
            "mean_abs_shap": mean_abs_mtl[feature_cols.index(feat)],
            "rank": int(imp_mtl.index.get_loc(feat)) + 1,
        })
        rows.append({
            "outcome": outcome, "feature": feat, "model": "STL",
            "mean_abs_shap": mean_abs_stl[feature_cols.index(feat)],
            "rank": int(imp_stl.index.get_loc(feat)) + 1,
        })

    # --- Concordance between MTL and STL ---
    rho, pval = spearmanr(
        [imp_mtl[f] for f in feature_cols],
        [imp_stl[f] for f in feature_cols],
    )
    top_mtl = set(imp_mtl.head(top_K).index)
    top_stl = set(imp_stl.head(top_K).index)
    jaccard = len(top_mtl & top_stl) / len(top_mtl | top_stl)

    agreement_rows.append({
        "outcome": outcome,
        "spearman_rho": rho,
        "spearman_pval": pval,
        f"jaccard_top{top_K}": jaccard,
        "top_features_mtl": ", ".join(imp_mtl.head(top_K).index),
        "top_features_stl": ", ".join(imp_stl.head(top_K).index),
    })

    # --- beeswarm plots ---
    for label, sv_arr, model_name in [
        ("mtl", sv_mtl_outcome, mtl_type),
        ("stl", sv_stl_outcome, stl_types[outcome]),
    ]:
        plt.figure()
        shap.summary_plot(sv_arr, X_test[feature_cols], show=False, max_display=15)
        plt.title(f"{outcome} — {label.upper()} ({model_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"beeswarm_{outcome}_{label}.png"), dpi=150)
        plt.close()

    # --- Comparative bar-plot top-K ---
    top_union = list(dict.fromkeys(list(imp_mtl.head(top_K).index) + list(imp_stl.head(top_K).index)))
    comp = pd.DataFrame({
        "MTL": [imp_mtl.get(f, 0) for f in top_union],
        "STL": [imp_stl.get(f, 0) for f in top_union],
    }, index=top_union).sort_values("MTL", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(top_union) + 1))
    y_pos = np.arange(len(top_union))
    ax.barh(y_pos - 0.2, comp["MTL"], height=0.4, label=f"MTL ({mtl_type})", color=color_MTL)
    ax.barh(y_pos + 0.2, comp["STL"], height=0.4, label=f"STL ({stl_types[outcome]})", color=color_STL)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comp.index)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"{outcome} — Feature importance: MTL vs STL")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"barplot_{outcome}_comparison.png"), dpi=150)
    plt.close()

# ==================== #
#      SAVE CSV        #
# ==================== #

pd.DataFrame(rows).to_csv(os.path.join(out_dir, "feature_importance_comparison.csv"), index=False)
agreement_df = pd.DataFrame(agreement_rows)
agreement_df.to_csv(os.path.join(out_dir, "agreement_metrics.csv"), index=False)

print("\n=== MTL Vs STL concordance by outcome ===")
print(agreement_df[["outcome", "spearman_rho", f"jaccard_top{top_K}"]].to_string(index=False))
print(f"\nOutput saved in: {out_dir}/")

