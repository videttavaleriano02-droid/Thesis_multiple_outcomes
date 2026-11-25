# Required packages

if (!require("dplyr")) install.packages("dplyr")
library(dplyr)
if (!require("synthpop")) install.packages("synthpop")
library(synthpop)
if (!require("tidyverse")) install.packages("tidyverse")
library(tidyverse)
if (!require("caret")) install.packages("caret")
library(caret)
if (!require("recipes")) install.packages("recipes")
library(recipes)
if (!require("tableone")) install.packages("tableone")
library(tableone)
if(!require(writexl)) install.packages("writexl")
library(writexl)

# Required Functions

split_random <- function(df, p = 0.80, seed = 1) {
  set.seed(seed)
  n <- nrow(df)
  idx <- sample(seq_len(n), size = floor(p * n))
  list(
    train = df[idx, ],
    test  = df[-idx, ]
  )
}

# --------

get_bin_vars <- function(df) {
  names(df)[sapply(df, function(x) {
    ux <- unique(na.omit(x))
    all(ux %in% c(0,1))
  })]
}

make_tableone <- function(df1, df2, label1, label2) {
  
  df1 <- df1 %>% mutate(group = label1)
  df2 <- df2 %>% mutate(group = label2)
  
  combined <- bind_rows(df1, df2)
  
  bin_vars <- get_bin_vars(combined)
  
  tab <- CreateTableOne(
    vars = setdiff(names(combined), "group"),
    strata = "group",
    data = combined,
    factorVars = bin_vars,
    test = TRUE,
    smd= T,
    includeNA = T
  )
  
  out <- print(tab, test = TRUE, smd = TRUE, showAllLevels = TRUE, printToggle = FALSE)
  out
}



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


make_plots <- function(real, synthetic) {
  
  real      <- prepare_data(real)      
  synthetic <- prepare_data(synthetic)

  # Rename variables for plotting only
  names(real)[match(names(var_labels), names(real))] <-
    var_labels

  names(synthetic)[match(names(var_labels), names(synthetic))] <-
    var_labels

  cmp_p <- compare(
    object = synthetic,
    data   = real,
    vars   = unname(var_labels),
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
