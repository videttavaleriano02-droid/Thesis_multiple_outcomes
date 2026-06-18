## Desc tables

library(tidyverse)
library(dplyr)

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


desc_stats <- function(data) {
  
  num_data <- data %>% select(where(is.numeric))
  
  if (ncol(num_data) == 0) {
    stop("Nessuna variabile numerica nel dataset")
  }
  
  out <- lapply(names(num_data), function(v) {
    x <- num_data[[v]]
    
    data.frame(
      variable = v,
      mean = mean(x, na.rm = TRUE),
      sd  = sd(x, na.rm = TRUE),
      iqr  = IQR(x, na.rm = TRUE)
    )
  })
  
  bind_rows(out)
}

raw_data <- raw_data %>%
  mutate(across(where(is.character),
                ~ suppressWarnings(as.numeric(gsub(",", ".", .)))))


# ---------
# table 1: raw vs clean dataset

raw_table  <- desc_stats(raw_data)
clean_table <- desc_stats(clean_data)

raw_clean_table <- raw_table %>%
  rename_with(~paste0("raw_", .), -variable) %>%
  full_join(
    clean_table %>% rename_with(~paste0("clean_", .), -variable),
    by = "variable"
  )

View(raw_clean_table)

# --------
# table 2: dev vs test

dev_table <- desc_stats(dev_data)
test_table <- desc_stats(test_data)


dev_test_table <- dev_table %>%
  rename(
    dev_mean = mean,
    dev_sd  = sd,
    dev_iqr  = iqr
  ) %>%
  left_join(
    test_table %>%
      rename(
        test_mean = mean,
        test_sd  = sd,
        test_iqr  = iqr
      ),
    by = "variable"
  )

View(dev_test_table)

# ---------
# table 3: 

final_table <- raw_clean_table %>%
  full_join(dev_test_table, by = "variable")

View(final_table)

# --------
# table 4: imp vs syn trains

train_imp_table <- desc_stats(train_imp)
syn_train_table <- desc_stats(syn_train)

train_compare_table <- train_imp_table %>%
  rename_with(~paste0("imp_", .), -variable) %>%
  full_join(
    syn_train_table %>% rename_with(~paste0("syn_", .), -variable),
    by = "variable"
  )

View(train_compare_table)
