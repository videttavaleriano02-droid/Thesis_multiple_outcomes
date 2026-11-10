rm(list = ls())
setwd("/Users/valerianovidetta/Desktop/Tesi/Dataset/")

library(tidyverse)
library(synthpop)
set.seed(2026)

dir.create("synthetic_comparison", showWarnings = FALSE)

# Categorical vars

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
  
  cmp <- compare(
    object = synthetic,
    data = real,
    stat = "percents",
    nrow = 4,
    ncol = 3,
    utility.stats = c("pMSE", "S_pMSE", "df"),
    table = TRUE,
    plot = FALSE,
    print.flag = TRUE
  )
  
  util <- utility.gen(
    object = synthetic,
    data = real,
    method = "cart",
    print.flag = FALSE
  )
  
  # S_pMSE
  tu <- cmp$tab.utility
  by_var <- as.data.frame(tu)
  by_var$variable <- rownames(by_var)
  by_var$dataset  <- label
  rownames(by_var) <- NULL
  
  
  write.csv(
    by_var,
    paste0("synthetic_comparison/utility_byvar_", label, ".csv"),
    row.names = FALSE
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
  
  list(overall = result, by_var = by_var)
}


# Cross-validation
results_folds <- list()
byvar_folds   <- list()

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
  
  # folds
  res_i <- run_comparison(real, synthetic, paste0("fold", i))
  results_folds[[i]] <- res_i$overall
  byvar_folds[[i]]   <- res_i$by_var
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

res_dev    <- run_comparison(real_dev, synthetic_dev, "development")
result_dev <- res_dev$overall
byvar_all  <- bind_rows(bind_rows(byvar_folds), res_dev$by_var)

write.csv(byvar_all, "synthetic_comparison/utility_byvar_all.csv", row.names = FALSE)


# Overall results
overall_summary <- bind_rows(
  results_folds,
  result_dev
)

byvar_ok <- byvar_all %>% filter(is.finite(S_pMSE), df > 0)

# median and maxima
byvar_ok %>%
  group_by(dataset) %>%
  summarise(median = median(S_pMSE), max = max(S_pMSE))

# 10 worst variables
byvar_ok %>%
  filter(dataset != "development") %>%
  group_by(variable) %>%
  summarise(mean_S_pMSE = mean(S_pMSE)) %>%
  arrange(desc(mean_S_pMSE)) %>%
  head(10)

# the three outcomes
byvar_ok %>%
  filter(variable %in% c("mbi_t1", "tct_t1", "mrs_t1")) %>%
  arrange(variable, dataset)


# mean across the 5 folds
global_folds <- results_folds %>%
  summarise(
    n_real_mean  = mean(n_real),
    n_synth_mean = mean(n_synth),
    pMSE_mean    = mean(pMSE),
    pMSE_sd      = sd(pMSE),
    S_pMSE_mean  = mean(S_pMSE),
    S_pMSE_sd    = sd(S_pMSE),
    S_pMSE_min   = min(S_pMSE),
    S_pMSE_max   = max(S_pMSE)
  )


byvar_folds_mean <- byvar_ok %>%
  filter(dataset != "development") %>%
  group_by(variable) %>%
  summarise(
    S_pMSE_mean = mean(S_pMSE),
    S_pMSE_sd   = sd(S_pMSE),
    n_folds     = n(),
    .groups = "drop"
  )

dev_vals <- byvar_ok %>%
  filter(dataset == "development") %>%
  select(variable, S_pMSE_dev = S_pMSE)

byvar_summary <- byvar_folds_mean %>%
  left_join(dev_vals, by = "variable") %>%
  arrange(desc(S_pMSE_mean))

write.csv(byvar_summary, "synthetic_comparison/utility_byvar_mean_folds.csv", 
          row.names = FALSE)
write.csv(global_folds,  "synthetic_comparison/utility_global_mean_folds.csv", 
          row.names = FALSE)

write.csv(
  overall_summary,
  "synthetic_comparison/utility_summary_overall.csv",
  row.names = FALSE
)

