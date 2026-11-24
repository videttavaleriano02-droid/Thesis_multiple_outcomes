
dir.create(paste0(diroutput, "/synthetic_comparison"), showWarnings = FALSE)

# cat vars as factors
prepare_data <- function(df) {
  as.data.frame(
    df %>% mutate(across(any_of(cat_vars), as.factor))
  )
}

# comparison synthtic vs real data
run_comparison <- function(real, synthetic, label) {
  
  real <- prepare_data(real)
  synthetic <- prepare_data(synthetic)
  
  cat("\nComparing:", label,
      "| Real:", nrow(real),
      "| Synthetic:", nrow(synthetic), "\n")
  
  util <- utility.gen(
    object = synthetic,
    data = real,
    method = "cart",
    print.flag = FALSE
  )
  
   result <- data.frame(
    dataset = label,
    n_real = nrow(real),
    n_synth = nrow(synthetic),
    pMSE = util$pMSE,
    S_pMSE = util$S_pMSE
  )
 
  cat("S_pMSE:", round(util$S_pMSE, 3), "\n")
  result
}


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

dir.create(paste0(diroutput, "/synthetic_comparison/plots"), showWarnings = FALSE)

vars_plot <- c(
  # outcomes (t1)
  "mbi_t1", "tct_t1", "mrs_t1",
  # outcomes at baseline (t0)
  "totale_mbi_t0_arm_1", "totale_tct_t0_arm_1", "mrs_t0_arm_1",
  # Clinical gravity
  "totale_nihss_t0_arm_1",
  # Demographic
  "eta_ammissione_t0_arm_1", "sesso",
  # Lesion
  "hemorragic_stroke", "lesione_sopratentoriale", "lato_lesione_destro_t0"
)

make_plots <- function(real, synthetic) {
  
  real      <- prepare_data(real)      
  synthetic <- prepare_data(synthetic)

  cmp_p <- compare(
    object = synthetic,
    data   = real,
    vars   = vars_plot,
    stat   = "percents",
    nrow   = 5,
    ncol   = 3,
    table  = FALSE,
    plot   = TRUE,
    print.flag = FALSE,
    utility.for.plot = NULL
  )
  
  ggsave(
    filename = paste0(diroutput, "/synthetic_comparison/plot_development.png"),
    plot   = cmp_p$plots,
    width  = 12,
    height = 8,
    dpi    = 300
  )
}

make_plots(real_dev, synthetic_dev)
