# ================================================================================================ #
#                                         SPLIT & IMPUTE                                           #
# ================================================================================================ #

rm(list = ls())
setwd('/Users/valerianovidetta/Desktop/Tesi/Dataset/') 

# libraries:
library(tidyverse)
library(caret)


# --------------------------------------------------------------------------------------------------

# import dataset

original <- readxl::read_xlsx('strategy_firenze.xlsx')

                                    # ------------------ #
                                    # HOLD-OUT SPLITTING #
                                    # ------------------ #

split_random <- function(df, p = 0.70, seed = 1) {
  set.seed(seed)
  n <- nrow(df)
  idx <- sample(seq_len(n), size = floor(p * n))
  list(
    train = df[idx, ],
    test  = df[-idx, ]
  )
}

split <- split_random(original, p = .8, seed = 272727)

develop <- split$train
test <- split$test

write.csv(develop, "development.csv")
write.csv(test, "test.csv")


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


















