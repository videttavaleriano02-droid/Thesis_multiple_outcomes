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
    test = TRUE,
    smd= T,
    includeNA = T
  )
  
  print(tab, test = TRUE, smd=T)
}

raw_data <- raw_data %>%
  mutate(across(where(is.character),
                ~ suppressWarnings(as.numeric(gsub(",", ".", .)))))

# ---------
# Clean data descriptives
bin_vars_clean <- get_bin_vars(clean_data)

clean_table <- CreateTableOne(
  vars = names(clean_data),
  data = clean_data,
  factorVars = bin_vars_clean,
  test=T,
  smd=T,
  includeNA = T)

print(clean_table, showAllLevels = TRUE, test=T, smd=T)

# per esportarla in un formato "guardabile" / esportabile
clean_table_df <- print(clean_table, showAllLevels = TRUE, printToggle = FALSE)
View(clean_table_df)


# --------
# table 2: dev vs test

dev_test_table <- make_tableone(dev_data, test_data, "dev", "test_set")
View(dev_test_table)

# --------
# table 3: imp vs syn trains

bin_vars_official <- get_bin_vars(clean_data)

train_imp <- train_imp %>%
  mutate(across(all_of(bin_vars_official), ~ factor(round(as.numeric(.)))))

syn_train <- syn_train %>%
  mutate(across(all_of(bin_vars_official), ~ factor(round(as.numeric(.)))))

train_compare_table <- make_tableone(train_imp, syn_train, "imputed", "synthetic")
View(train_compare_table)
