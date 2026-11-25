# CODE TO RUN ALL STEPS WITHIN R
# THEN PASS TO "04_01_utils.py" and "04_02_ML_pipeline.py" TO RUN THE MACHINE LEARNING PIPELINE

# clean env
rm(list=ls())

maindir <- "/Users/valerianovidetta/Desktop/Tesi"

dirinput <- "/Users/valerianovidetta/Desktop/Tesi/Dataset"
dirRscripts <- paste0(maindir, "/code/R_code")
dir.create(paste0(maindir, "/Intermediate"), showWarnings = FALSE)
dirintermediate <- paste0(maindir, "/Intermediate")
dir.create(paste0(maindir, "/code/Output"), showWarnings = FALSE)
diroutput <- paste0(maindir, "/code/Output")

# set the seed
set.seed(2026)

## 00 Setting parameters
source(paste0(dirRscripts, "/00_parameters.R"))

## 01 Data cleaning and adjustment
source(paste0(dirRscripts, "/01_data_cleaning.R"))

## 02_01 Data splitting
source(paste0(dirRscripts, "/02_01_data_splitting.R"))

## 02_02 Imputation
source(paste0(dirRscripts, "/02_02_data_imputation.R"))

## 02_03 Synthetic data augmentation
source(paste0(dirRscripts, "/02_03_data_augmentation.R"))

## 03_01 Descriptive tables
source(paste0(dirRscripts, "/03_01_desc_tables.R"))

## 03_02 Synthetic data comparison
source(paste0(dirRscripts, "/03_02_synthetic_comparison.R"))

