library(tableone)
library(readxl)
setwd('/Users/valerianovidetta/Desktop/Tesi/Dataset/')
strategy <- read_xlsx('strategy_firenze.xlsx')

vars <- names(strategy)

factor_vars <- c(
  
  "status",
  "sesso_E1",
  "convivi_dicot_E1",
  "centro_E1",
  "tipologia_ictus_E1",
  "sopratent_sottotent_E1",
  "recidiva_E1",
  "lato_3cat_E1",
  
  "ridotta_vigilanza_o_coma_E1",
  "instabilita_clinica_E1",
  "delirium_E1",
  "infezione_acuta_E1",
  "depresisone_E1",
  "disfagia_E1",
  "malnutrizione_E1",
  "ulcera_da_pressione_E1",
  "catetere_vescicale_E1",
  "catetere_venoso_centrale_E1",
  "tracheostomia_E1",
  "dolore_marc_E1",
  "sng_peg_E1",
  
  "ridotta_vigilanza_o_coma_E2",
  "instabilita_clinica_E2",
  "delirium_E2",
  "infezione_acuta_E2",
  "depresisone_E2",
  "disfagia_E2",
  "malnutrizione_E2",
  "ulcera_da_pressione_E2",
  "catetere_vescicale_E2",
  "catetere_venoso_centrale_E2",
  "tracheostomia_E2",
  "dolore_marc_E2",
  "sng_peg_E2",
  
  "recidiva_pre_E4",
  "recidiva_pre_E5",
  
  "convivi_dicot_E4",
  "convivi_dicot_E5",
  
  "scala_instabilit_clinica___1_E1",
  "scala_instabilit_clinica___2_E1",
  "scala_instabilit_clinica___3_E1",
  "scala_instabilit_clinica___4_E1",
  "scala_instabilit_clinica___5_E1",
  "scala_instabilit_clinica___0_E1",
  
  "scala_instabilit_clinica___1_E2",
  "scala_instabilit_clinica___2_E2",
  "scala_instabilit_clinica___3_E2",
  "scala_instabilit_clinica___4_E2",
  "scala_instabilit_clinica___5_E2",
  "scala_instabilit_clinica___0_E2",
  
  "accettazione_E1",
  "dimissione_presso_E2",
  
  "programma_riabilitativo___1_E2",
  "programma_riabilitativo___2_E2",
  "programma_riabilitativo___3_E2",
  "programma_riabilitativo___4_E2",
  "programma_riabilitativo___5_E2",
  "programma_riabilitativo___6_E2",
  "programma_riabilitativo___7_E2",
  "programma_riabilitativo___9_E2",
  "programma_riabilitativo___10_E2",
  "programma_riabilitativo___13_E2",
  
  "presc_ausili_E2",
  "trattamento_con_antidepres_E2",
  "trattamento_del_dolore_E2",
  "trattamento_nutrizionale_o_E2",
  "nutrizione_parenterale_E2",
  
  "edema_E1",
  
  "condizioni___1_E1",
  "condizioni___2_E1",
  "condizioni___3_E1",
  "condizioni___4_E1",
  
  "protesi_udito_E1",
  "protesi_udito_E2",
  "protesi_vista_E1",
  "protesi_vista_E2",
  
  "incontinenza_urinaria_E1",
  "incontinenza_urinaria_E2",
  
  "anemia_E1",
  "anemia_E2",
  
  "dialisi_E1",
  "dialisi_E2",
  
  "toast_E1",
  
  "infezione_urinaria_E2",
  "infezione_non_urinaria_E2",
  "caduta_E2",
  "contenzione_fisica_farmaco_E2"
) 

tab1 <- CreateTableOne(vars= vars[2:length(vars)], 
                       factorVars = factor_vars, 
                       data = strategy, 
                       includeNA = TRUE)
tab1_clean <- print(tab1, printToggle = FALSE, showAllLevels = TRUE)
tab1_clean
write.csv(tab1_clean, "table1_descriptives.csv")

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

abline(h = 15, col = "red", lwd = 1, lty = 2)


outcomes <- c('totale_mbi_E1','totale_mbi_E2','totale_mbi_fu_E4','totale_mbi_fu_E5',
              'fac_E1', 'fac_E2', 'fac_fu_E4','fac_fu_E5',
               'totale_tct_E1', 'totale_tct_E2', 
               'mmse_strategy_corretto_E1','mmse_strategy_corretto_E2'
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
