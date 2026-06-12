library(tableone)
library(readxl)
setwd('/Users/valerianovidetta/Desktop/Tesi/Dataset/')
strategy <- read_csv('clean_dataset.csv')[,-1]

vars <- names(strategy)

na_perc <- sapply(strategy, function(x) mean(is.na(x)) * 100)

na_summary <- data.frame(
  variabile = names(na_perc),
  na_perc = na_perc
)

na_summary <- na_summary[order(na_summary$na_perc, decreasing = TRUE), ]

par(mar = c(10, 4, 4, 2))

bp <- barplot(na_summary$na_perc,
              names.arg = na_summary$variabile,
              col = "steelblue",
              las = 2,
              cex.names = 0.7)

abline(h = 50, col = "red", lwd = 1, lty = 2)


outcomes <- c('totale_mbi_t0_arm_1','mbi_t1','totale_mbi_fu_t2_arm_1','totale_mbi_fu_t3_arm_1',
              'fac_t0_arm_1', 'fac_t1', 'fac_fu_t2_arm_1','fac_fu_t3_arm_1',
               'totale_tct_t0_arm_1', 'tct_t1', 
               'totale_mmse_t0_arm_1','mmse_t1',
              "mrs_t0_arm_1", "mrs_t1", "mrs_fu_t2_arm_1", "mrs_fu_t3_arm_1",
              "totale_sppb_t0_arm_1", "sppb_t1"
)

summary_outcomes <- data.frame(
  N = sapply(strategy[,outcomes], function(x) sum(!is.na(x))),
  
  Mean = sapply(strategy[,outcomes], function(x)
    mean(x, na.rm = TRUE)),
  
  SD = sapply(strategy[,outcomes], function(x)
    sd(x, na.rm = TRUE)),
  
  Median = sapply(strategy[,outcomes], function(x)
    median(x, na.rm = TRUE)),
  
  IQR = sapply(strategy[,outcomes], function(x)
    IQR(x, na.rm = TRUE)),
  
  NA_perc = sapply(strategy[,outcomes], function(x)
    mean(is.na(x)) * 100)
)
summary_outcomes
