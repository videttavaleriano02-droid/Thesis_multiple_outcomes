## Desc tables

library(tidyverse)
library(dplyr)
library(tableone)

setwd('/Users/valerianovidetta/Desktop/Tesi/Dataset')

# ----------
# reading files

raw_data <- readxl::read_xlsx('strategy_valeriano_wide_solo_event.xlsx')
clean_data <- read_csv('clean_dataset.csv')

dev_data <- read_csv('hold_out/development.csv')
test_data <- read_csv('hold_out/test.csv')

train1 <- read_csv('folds_imputed/train_fold1.csv')
train2 <- read_csv('folds_imputed/train_fold2.csv')
train3 <- read_csv('folds_imputed/train_fold3.csv')
train4 <- read_csv('folds_imputed/train_fold4.csv')
train5 <- read_csv('folds_imputed/train_fold5.csv')

train_imp <- rbind(train1, train2, train3, train4, train5)

syn_train1 <- read_csv('folds_synthetic/train_syn_fold1.csv')
syn_train2 <- read_csv('folds_synthetic/train_syn_fold2.csv')
syn_train3 <- read_csv('folds_synthetic/train_syn_fold3.csv')
syn_train4 <- read_csv('folds_synthetic/train_syn_fold4.csv')
syn_train5 <- read_csv('folds_synthetic/train_syn_fold5.csv')

syn_train <- rbind(syn_train1, syn_train2, syn_train3, syn_train4, syn_train5)

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
    test = TRUE
  )
  
  print(tab, test = TRUE)
}

raw_data <- raw_data %>%
  mutate(across(where(is.character),
                ~ suppressWarnings(as.numeric(gsub(",", ".", .)))))

# ---------
# table 1: raw vs clean dataset

raw_clean_table <- make_tableone(raw_data, clean_data, "raw", "clean")
View(raw_clean_table)

View(raw_clean_table)

# --------
# table 2: dev vs test

dev_test_table <- make_tableone(dev_data, test_data, "dev", "test")
View(dev_test_table)

# --------
# table 3: imp vs syn trains

train_compare_table <- make_tableone(train_imp, syn_train, "imputed", "synthetic")
View(train_compare_table)
