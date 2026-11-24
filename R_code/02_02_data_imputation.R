
                                    # ------------------ #
                                    #     IMPUTATION     #
                                    # ------------------ #


dir.create(paste0(dirintermediate, "/folds_imputed"), showWarnings = F)

k = 5 # neighbours to include, between 5-10

for (i in seq_along(folds_spl)){
  
  fold_train <- folds_spl[[i]]$train
  fold_val <- folds_spl[[i]]$validation
  
  rec <- recipe(~., data=fold_train) %>%
    step_impute_knn(everything(), neighbors = k) %>%
    prep(training = fold_train)
  
  
  train_imp <- bake(rec, new_data=NULL)
  val_imp <- bake(rec, new_data = fold_val)
  
  write.csv(train_imp, paste0(dirintermediate, "/folds_imputed/train_fold", i, ".csv"), row.names = F)
  write.csv(val_imp,   paste0(dirintermediate, "/folds_imputed/val_fold",   i, ".csv"), row.names = F)
  
}

## Imputing development and test set

dir.create(paste0(dirintermediate, "/hold_out_imputed"), showWarnings = F)

rec_dev <- recipe(~., data=develop) %>%
  step_impute_knn(everything(), neighbors = k) %>%
  prep(training = develop)

dev_imp <- bake(rec_dev, new_data=NULL)
test_imp <- bake(rec_dev, new_data = test)

write.csv(dev_imp, paste0(dirintermediate, "/hold_out_imputed/development.csv"), row.names = F)
write.csv(test_imp,   paste0(dirintermediate, "/hold_out_imputed/test.csv"), row.names = F)
