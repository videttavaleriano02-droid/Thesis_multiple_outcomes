# ================================================================================================ #
#                                         SPLIT & IMPUTE                                           #
# ================================================================================================ #

rm(list = ls())
setwd('/Users/valerianovidetta/Desktop/Tesi/Dataset/') 

# libraries:
library(tidyverse)
library(caret)
library(synthpop)
library(recipes)

set.seed(2026)

# --------------------------------------------------------------------------------------------------

# import dataset

original <- read_csv('clean_dataset.csv')
head(original)
original <- original[,-c(1,2)]

# ----
# categorical vars as factors

cat_vars <- c(
  "sesso", "caregiver_t0_arm_1", "convivi___1_t0_arm_1", "convivi___2_t0_arm_1",
  "convivi___3_t0_arm_1", "convivi___4_t0_arm_1", "centro",
  
  "hemorragic_stroke", "recidiva_t0_arm_1", "sede_lesione___0_t0_arm_1",
  "lato_lesione_destro_t0", "lato_lesione_sinistro_t0", "lato_lesione_bilaterale_t0",
  "lesione_sopratentoriale", "lesione_sottotentoriale",
  
  "scala_instabilit_clinica___1_t0_arm_1", "scala_instabilit_clinica___2_t0_arm_1",
  "scala_instabilit_clinica___3_t0_arm_1", "scala_instabilit_clinica___4_t0_arm_1",
  "scala_instabilit_clinica___5_t0_arm_1", "scala_instabilit_clinica___0_t0_arm_1",
  
  "ridotta_vigilanza_o_coma_t0_arm_1", "delirium_t0_arm_1", "instabilita_clinica_t0_arm_1",
  "infezione_acuta_t0_arm_1", "depresisone_t0_arm_1", "dolore_marc_t0_arm_1", "disfagia_t0_arm_1",
  "malnutrizione_t0_arm_1", "sng_peg_t0_arm_1", "ulcera_da_pressione_t0_arm_1",
  "catetere_vescicale_t0_arm_1", "catetere_venoso_centrale_t0_arm_1",
  "tracheostomia_t0_arm_1", "dialisi_t0_arm_1", "anemia_t0_arm_1"
)

original <- original %>%
  mutate(across(any_of(cat_vars), as.factor))

                                    # ------------------ #
                                    # HOLD-OUT SPLITTING #
                                    # ------------------ #
dir.create('hold_out', showWarnings = F)

split_random <- function(df, p = 0.80, seed = 1) {
  set.seed(seed)
  n <- nrow(df)
  idx <- sample(seq_len(n), size = floor(p * n))
  list(
    train = df[idx, ],
    test  = df[-idx, ]
  )
}

split <- split_random(original, p = .8, seed = 2026)

develop <- split$train
test <- split$test

write.csv(develop, "hold_out/development.csv", row.names = F)
write.csv(test, "hold_out/test.csv", row.names = F)


                                    # ------------------ #
                                    #  CROSS VALIDATION  #
                                    # ------------------ #


folds <- createFolds(rownames(develop), k = 5, list = T, returnTrain = F)

dir.create('folds_csv', showWarnings = F)

folds_spl <- lapply(folds,
            function(x){
              validation <- develop[x,]
              train <- develop[-x,]
              list(train = train, validation = validation) 
            }
)

for (i in seq_along(folds_spl)) {
  write.csv(folds_spl[[i]]$train,      paste0("folds_csv/train_fold", i, ".csv"), row.names = F)
  write.csv(folds_spl[[i]]$validation, paste0("folds_csv/val_fold",   i, ".csv"), row.names = F)
}


                                    # ------------------ #
                                    #     IMPUTATION     #
                                    # ------------------ #


dir.create('folds_imputed', showWarnings = F)

k = 5 # neighbours to include, between 5-10

for (i in seq_along(folds_spl)){
  
  fold_train <- folds_spl[[i]]$train
  fold_val <- folds_spl[[i]]$validation
  
  rec <- recipe(~., data=fold_train) %>%
    step_impute_knn(everything(), neighbors = k) %>%
    prep(training = fold_train)
  
  
  train_imp <- bake(rec, new_data=NULL)
  val_imp <- bake(rec, new_data = fold_val)
  
  write.csv(train_imp, paste0("folds_imputed/train_fold", i, ".csv"), row.names = F)
  write.csv(val_imp,   paste0("folds_imputed/val_fold",   i, ".csv"), row.names = F)
  
}

## Imputing development and test set

dir.create('hold_out_imputed', showWarnings = F)

rec_dev <- recipe(~., data=develop) %>%
  step_impute_knn(everything(), neighbors = k) %>%
  prep(training = develop)

dev_imp <- bake(rec_dev, new_data=NULL)
test_imp <- bake(rec_dev, new_data = test)

write.csv(dev_imp, paste0("hold_out_imputed/development.csv"), row.names = F)
write.csv(test_imp,   paste0("hold_out_imputed/test.csv"), row.names = F)


                                    # ------------------ #
                                    #     SYNTHETIZE     #
                                    # ------------------ #

dir.create('folds_synthetic', showWarnings = F)

for ( i in seq_along(folds_spl)) {

  train_imp <- read_csv(paste0("folds_imputed/train_fold", i, ".csv"))

  n_real <- nrow(train_imp)
  n_synth <- 2000 - n_real
  
  synth <- syn(train_imp, k= n_synth, seed=2026)
  
  train_augmented <- rbind(train_imp, synth$syn)
  
  write.csv(train_augmented, paste0("folds_synthetic/train_syn_fold", i, ".csv"), row.names = F)
  
  cat("Fold", i, "- real:", n_real, "| synthetic:", n_synth, 
      "| totale:", nrow(train_augmented), "\n")
  
}


## synthetize also development
dir.create("development_synthetic", showWarnings = F)

dev_imp <- read_csv("hold_out_imputed/development.csv")

n_real <- nrow(dev_imp)
n_synth <- 10000 - n_real

synth <- syn(dev_imp, k= n_synth, seed=2026)

dev_augmented <- rbind(dev_imp, synth$syn)
write.csv(dev_augmented, "development_synthetic/development_syn.csv", row.names = F)

cat("Development:", n_real, "| synthetic:", n_synth, "| totale:", nrow(dev_augmented), "\n")



