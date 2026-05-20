library(tableone)
library(readxl)

strategy <- read_xlsx('/Users/valerianovidetta/Desktop/Tesi/Dataset/strategy_firenze.xlsx')

vars <- names(strategy)

factor_vars <- c('') # inserire variabili categoriche da codebook 

tab1 <- CreateTableOne(vars= vars, 
                       factorVars = setdiff(factor_vars), 
                       data = data_reaction, 
                       includeNA = TRUE)
print(tab1)
tab1_df <- print(tab1, printToggle = FALSE, showAllLevels = TRUE)
write.csv(tab1_df, "table1_descriptives.csv")