
dir.create(paste0(diroutput, "/synthetic_comparison"), showWarnings = FALSE)

# Cross-validation
results_folds <- list()

for (i in 1:5) {
  
  real <- read_csv(
    paste0(dirintermediate, "/folds_imputed/train_fold", i, ".csv"),
    show_col_types = FALSE
  )
  
  augmented <- read_csv(
    paste0(dirintermediate, "/folds_synthetic/train_syn_fold", i, ".csv"),
    show_col_types = FALSE
  )
  
  # augmented = file with real data -> then synths
  synthetic <- augmented[(nrow(real) + 1):nrow(augmented), ]
  
  results_folds[[i]] <- run_comparison(real, synthetic, paste0("fold", i))

}

results_folds <- bind_rows(results_folds)

# Development set
real_dev <- read_csv(
  paste0(dirintermediate, "/hold_out_imputed/development.csv"),
  show_col_types = FALSE
)

augmented_dev <- read_csv(
  paste0(dirintermediate, "/development_synthetic/development_syn.csv"),
  show_col_types = FALSE
)

synthetic_dev <- augmented_dev[
  (nrow(real_dev) + 1):nrow(augmented_dev),
]

result_dev <- run_comparison(real_dev, synthetic_dev, "development")


mean_folds <- results_folds %>%
  summarise(
    dataset = "mean_folds",
    n_real  = mean(n_real),
    n_synth = mean(n_synth),
    pMSE    = mean(pMSE),
    S_pMSE  = mean(S_pMSE)
  )

overall_summary <- bind_rows(results_folds, mean_folds, result_dev)

write.csv(overall_summary,
          paste0(diroutput, "/synthetic_comparison/utility_summary_overall.csv"),
          row.names = FALSE)

print(overall_summary)

## Make plots

var_labels <- c(
  "mbi_t1" = "MBI at follow-up",
  "tct_t1" = "TCT at follow-up",
  "mrs_t1" = "mRS at follow-up",
  "totale_mbi_t0_arm_1" = "MBI at baseline",
  "totale_tct_t0_arm_1" = "TCT at baseline",
  "mrs_t0_arm_1" = "mRS at baseline",
  "totale_nihss_t0_arm_1" = "NIHSS at baseline",
  "eta_ammissione_t0_arm_1" = "Age at admission",
  "sesso" = "Sex",
  "hemorragic_stroke" = "Hemorrhagic stroke",
  "lesione_sopratentoriale" = "Supratentorial lesion",
  "lato_lesione_destro_t0" = "Right-sided lesion"
)

make_plots(real_dev, synthetic_dev)
