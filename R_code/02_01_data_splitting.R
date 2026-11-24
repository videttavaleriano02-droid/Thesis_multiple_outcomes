                                    # ------------------ #
                                    # HOLD-OUT SPLITTING #
                                    # ------------------ #

dir.create(paste0(dirintermediate, "/hold_out"), showWarnings = F)

split_random <- function(df, p = 0.80, seed = 1) {
  set.seed(seed)
  n <- nrow(df)
  idx <- sample(seq_len(n), size = floor(p * n))
  list(
    train = df[idx, ],
    test  = df[-idx, ]
  )
}

split <- split_random(clean_data, p = .8, seed = 2026)

develop <- split$train
test <- split$test

write.csv(develop, paste0(dirintermediate, "/hold_out/development.csv"), row.names = F)
write.csv(test, paste0(dirintermediate, "/hold_out/test.csv"), row.names = F)


                                    # ------------------ #
                                    #  CROSS VALIDATION  #
                                    # ------------------ #


folds <- createFolds(rownames(develop), k = 5, list = T, returnTrain = F)

dir.create(paste0(dirintermediate, "/folds_csv"), showWarnings = F)

folds_spl <- lapply(folds,
            function(x){
              validation <- develop[x,]
              train <- develop[-x,]
              list(train = train, validation = validation) 
            }
)

for (i in seq_along(folds_spl)) {
  write.csv(folds_spl[[i]]$train,      paste0(dirintermediate, "/folds_csv/train_fold", i, ".csv"), row.names = F)
  write.csv(folds_spl[[i]]$validation, paste0(dirintermediate, "/folds_csv/val_fold",   i, ".csv"), row.names = F)
}
