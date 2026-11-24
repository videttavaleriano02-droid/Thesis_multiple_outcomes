                                    # ------------------ #
                                    #     SYNTHETIZE     #
                                    # ------------------ #

dir.create(paste0(dirintermediate, "/folds_synthetic"), showWarnings = F)

for ( i in seq_along(folds_spl)) {

  train_imp <- read_csv(paste0(dirintermediate, "/folds_imputed/train_fold", i, ".csv"))

  n_real <- nrow(train_imp)
  n_synth <- 2000 - n_real
  
  synth <- syn(train_imp, k= n_synth, seed=2026)
  
  train_augmented <- rbind(train_imp, synth$syn)
  
  write.csv(train_augmented, paste0(dirintermediate, "/folds_synthetic/train_syn_fold", i, ".csv"), row.names = F)
  
  cat("Fold", i, "- real:", n_real, "| synthetic:", n_synth, 
      "| totale:", nrow(train_augmented), "\n")
  
}


## synthetize also development
dir.create(paste0(dirintermediate, "/development_synthetic"), showWarnings = F)

dev_imp <- read_csv(paste0(dirintermediate, "/hold_out_imputed/development.csv"))

n_real <- nrow(dev_imp)
n_synth <- 10000 - n_real

synth <- syn(dev_imp, k= n_synth, seed=2026)

dev_augmented <- rbind(dev_imp, synth$syn)
write.csv(dev_augmented, paste0(dirintermediate, "/development_synthetic/development_syn.csv"), row.names = F)

cat("Development:", n_real, "| synthetic:", n_synth, "| totale:", nrow(dev_augmented), "\n")



