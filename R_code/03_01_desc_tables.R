## Desc tables

# ----------
# reading files

raw_data <- readxl::read_xlsx(paste0(dirinput, "/strategy_valeriano_wide_solo_event.xlsx"))
clean_data <- read_csv(paste0(dirintermediate, "/clean_dataset.csv"))

dev_data <- read_csv(paste0(dirintermediate, "/hold_out/development.csv"))
test_data <- read_csv(paste0(dirintermediate, "/hold_out/test.csv"))

train1 <- read_csv(paste0(dirintermediate, "/folds_imputed/train_fold1.csv"))
train2 <- read_csv(paste0(dirintermediate, "/folds_imputed/train_fold2.csv"))
train3 <- read_csv(paste0(dirintermediate, "/folds_imputed/train_fold3.csv"))
train4 <- read_csv(paste0(dirintermediate, "/folds_imputed/train_fold4.csv"))
train5 <- read_csv(paste0(dirintermediate, "/folds_imputed/train_fold5.csv"))

train_imp <- rbind(train1, train2, train3, train4, train5)

syn_train1 <- read_csv(paste0(dirintermediate, "/folds_synthetic/train_syn_fold1.csv"))
syn_train2 <- read_csv(paste0(dirintermediate, "/folds_synthetic/train_syn_fold2.csv"))
syn_train3 <- read_csv(paste0(dirintermediate, "/folds_synthetic/train_syn_fold3.csv"))
syn_train4 <- read_csv(paste0(dirintermediate, "/folds_synthetic/train_syn_fold4.csv"))
syn_train5 <- read_csv(paste0(dirintermediate, "/folds_synthetic/train_syn_fold5.csv"))

syn_train <- rbind(syn_train1, syn_train2, syn_train3, syn_train4, syn_train5)

# --------
# create table folder

dir.create(paste0(diroutput, "/tables"), showWarnings = FALSE)

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

# to export in a more friendly format, we can convert the table to a data frame and view it
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

# ---------
# save tables

write_xlsx(
  list(
    clean_data  = as.data.frame(clean_table_df) %>% tibble::rownames_to_column("variable"),
    dev_vs_test = as.data.frame(dev_test_table) %>% tibble::rownames_to_column("variable"),
    imp_vs_syn  = as.data.frame(train_compare_table) %>% tibble::rownames_to_column("variable")
  ),
  paste0(diroutput, "/tables/descriptive_tables.xlsx")
)