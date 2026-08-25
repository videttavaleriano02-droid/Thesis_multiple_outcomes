rm(list = ls())
setwd("/Users/valerianovidetta/Desktop/Tesi/Dataset/")

library(tidyverse)
library(synthpop)

dir.create("synthetic_comparison", showWarnings = FALSE)

# Variabili categoriche
cat_vars <- c(
  "sesso", "caregiver_t0_arm_1", "convivi___1_t0_arm_1", "convivi___2_t0_arm_1",
  "convivi___3_t0_arm_1", "convivi___4_t0_arm_1", "centro",
  "hemorragic_stroke", "recidiva_t0_arm_1", "sede_lesione___0_t0_arm_1",
  "lato_lesione_destro_t0", "lato_lesione_sinistro_t0", "lato_lesione_bilaterale_t0",
  "lesione_sopratentoriale", "lesione_sottotentoriale",
  "scala_instabilit_clinica___1_t0_arm_1", "scala_instabilit_clinica___2_t0_arm_1",
  "scala_instabilit_clinica___3_t0_arm_1", "scala_instabilit_clinica___4_t0_arm_1",
  "scala_instabilit_clinica___5_t0_arm_1", "scala_instabilit_clinica___0_t0_arm_1",
  "ridotta_vigilanza_o_coma_t0_arm_1", "delirium_t0_arm_1",
  "instabilita_clinica_t0_arm_1", "infezione_acuta_t0_arm_1",
  "depresisone_t0_arm_1", "dolore_marc_t0_arm_1", "disfagia_t0_arm_1",
  "malnutrizione_t0_arm_1", "sng_peg_t0_arm_1", "ulcera_da_pressione_t0_arm_1",
  "catetere_vescicale_t0_arm_1", "catetere_venoso_centrale_t0_arm_1",
  "tracheostomia_t0_arm_1", "dialisi_t0_arm_1", "anemia_t0_arm_1"
)

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
  
  pdf(
    paste0("synthetic_comparison/compare_", label, ".pdf"),
    width = 12,
    height = 9
  )
  
  cmp <- compare(
    object = synthetic,
    data = real,
    stat = "percents",
    nrow = 4,
    ncol = 3,
    utility.stats = c("pMSE", "S_pMSE", "df"),
    table = TRUE,
    plot = TRUE,
    print.flag = TRUE
  )
  
  dev.off()
  
  if (!is.null(cmp$tables)) {
    saveRDS(
      cmp$tables,
      paste0("synthetic_comparison/compare_tables_", label, ".rds")
    )
  }
  
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
  
  write.csv(
    result,
    paste0("synthetic_comparison/utility_", label, ".csv"),
    row.names = FALSE
  )
  
  cat("S_pMSE:", round(util$S_pMSE, 3), "\n")
  
  result
}


# Cross-validation
results_folds <- list()

for (i in 1:5) {
  
  real <- read_csv(
    paste0("folds_imputed/train_fold", i, ".csv"),
    show_col_types = FALSE
  )
  
  augmented <- read_csv(
    paste0("folds_synthetic/train_syn_fold", i, ".csv"),
    show_col_types = FALSE
  )
  
  # augmented = file with real data -> then synths
  synthetic <- augmented[(nrow(real) + 1):nrow(augmented), ]
  
  results_folds[[i]] <- run_comparison(
    real,
    synthetic,
    paste0("fold", i)
  )
}

results_folds <- bind_rows(results_folds)

write.csv(
  results_folds,
  "synthetic_comparison/utility_summary_folds.csv",
  row.names = FALSE
)


# Development set
real_dev <- read_csv(
  "hold_out_imputed/development.csv",
  show_col_types = FALSE
)

augmented_dev <- read_csv(
  "development_synthetic/development_syn.csv",
  show_col_types = FALSE
)

synthetic_dev <- augmented_dev[
  (nrow(real_dev) + 1):nrow(augmented_dev),
]

result_dev <- run_comparison(
  real_dev,
  synthetic_dev,
  "development"
)


# Risultati complessivi
overall_summary <- bind_rows(
  results_folds,
  result_dev
)

write.csv(
  overall_summary,
  "synthetic_comparison/utility_summary_overall.csv",
  row.names = FALSE
)

