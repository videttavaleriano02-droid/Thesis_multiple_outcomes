                                      # ------------------ #
                                      #   Adjusting Data   #
                                      # ------------------ #
setwd('/Users/valerianovidetta/Desktop/Tesi/Dataset/') 

library(dplyr)
library(synthpop)

#' impute values of mbi,fac, ... at dimission
#' capire perchè ho meno variabili
#' creare variabile in cui si considera codice 26 se diverso da codice 56 
#' (paziente dimesso e riammesso con altro codice)
#' 

# ----------------

raw_data <- readxl::read_xlsx('strategy_valeriano_wide_solo_event.xlsx')

raw_data <- raw_data[raw_data$record_id >= 68, ] # per ora rimaniamo fino al 68


# ------

# MMSE resa numerica

raw_data$totale_mmse_t0_arm_1 <- as.numeric(raw_data$totale_mmse_t0_arm_1)
raw_data$totale_mmse_t1_dimissione_56_arm_1 <- as.numeric(raw_data$totale_mmse_t1_dimissione_56_arm_1)
raw_data$totale_mmse_t1_dimissione_ex_2_arm_1 <- as.numeric(raw_data$totale_mmse_t1_dimissione_ex_2_arm_1)


# --------

# rendere dicotomiche alcune variabili categoriche:

# ----
# tipologia_ictus_ischemico_2_t0_arm_1
# 1. sindromi lacunari 
# 2. sindromi circolo posteriore
# 3. sindrome completa circolo anteriore
# 4. sindrome parziale circolo anteriore

# prendo sindromi lacunari come reference 
raw_data$ictus_ischemico_circ_post_t0 <- ifelse(raw_data$tipologia_ictus_ischemico_2_t0_arm_1 == 2, 1, 0)
raw_data$ictus_ischemico_compl_circ_ant_t0 <- ifelse(raw_data$tipologia_ictus_ischemico_2_t0_arm_1 == 3, 1, 0)
raw_data$ictus_ischemico_parz_circ_ant_t0 <- ifelse(raw_data$tipologia_ictus_ischemico_2_t0_arm_1 == 4, 1, 0)

# ----
# lato_t0_arm_1
# 0. nessuno 
# 1. destro 
# 2. sinistra 
# 3. bilaterale

# nessun lato come reference
raw_data$lato_lesione_destro_t0  <- ifelse(raw_data$lato_t0_arm_1 == 1, 1, 0)
raw_data$lato_lesione_sinistro_t0 <- ifelse(raw_data$lato_t0_arm_1 == 2, 1, 0)
raw_data$lato_lesione_bilaterale_t0 <- ifelse(raw_data$lato_t0_arm_1 == 3, 1, 0)


# cammino_t0_arm_1
# ???


# Dividere in due la variabile: sede lesione
# 1 se tronco e cerello (sede da 7 a 10) <- lesione sotto tentoriale
# 1 tutto il resto <- lesione sopra tentoriale
raw_data$lesione_sopratentoriale <- ifelse(
  rowSums(raw_data[, paste0("sede_lesione___", 1:6, "_t0_arm_1")], na.rm = TRUE) > 0,
  1, 0
)
raw_data$lesione_sottotentoriale <- ifelse(
  rowSums(raw_data[, paste0("sede_lesione___", 7:10, "_t0_arm_1")], na.rm = TRUE) > 0,
  1, 0
)


# -----------

# unire variabili codici diversi

# alcuni pazienti passano da codice 56 a 26, quindi vengono dimessi e riammessi, poi ridimessi
# vanno unite le variabili prendendo il valore ultimo (quindi col codice 26)

# coalesce prende il primo valore non NA tra le variabili, quindi guarda prima se ho un valore al 
# codice 26 e poi sennò prende quello con codice 56

raw_data <- raw_data %>%
  mutate(
    
    mbi_t1   = coalesce(totale_mbi_t1_dimissione_ex_2_arm_1,   totale_mbi_t1_dimissione_56_arm_1),
    
    tct_t1   = coalesce(totale_tct_t1_dimissione_ex_2_arm_1,   totale_tct_t1_dimissione_56_arm_1),
    
    mmse_t1  = coalesce(totale_mmse_t1_dimissione_ex_2_arm_1,  totale_mmse_t1_dimissione_56_arm_1),
    
    fac_t1   = coalesce(fac_t1_dimissione_ex_2_arm_1,          fac_t1_dimissione_56_arm_1),
    
    sppb_t1  = coalesce(totale_sppb_t1_dimissione_ex_2_arm_1,  totale_sppb_t1_dimissione_56_arm_1),
    
    mrs_t1   = coalesce(mrs_t1_dimissione_ex_2_arm_1,          mrs_t1_dimissione_56_arm_1),
    
    scala_instabilita_clinica_1_t1 = coalesce(scala_instabilit_clinica___1_t1_dimissione_ex_2_arm_1,
                                              scala_instabilit_clinica___1_t1_dimissione_56_arm_1),
    
    scala_instabilita_clinica_2_t1 = coalesce(scala_instabilit_clinica___2_t1_dimissione_ex_2_arm_1,
                                              scala_instabilit_clinica___2_t1_dimissione_56_arm_1),
    
    dimissione_presso_t1 = coalesce(dimissione_presso_t1_dimissione_ex_2_arm_1,
                                    dimissione_presso_t1_dimissione_56_arm_1)
    
  )


# codebook summary of the variables
codebook.syn(raw_data)$tab 


# -----

# ripuliamo le variabili che non ci interessano più:

var_to_omit <- c( 'scala_instabilit_clinica___1_t1_dimissione_ex_2_arm_1','scala_instabilit_clinica___1_t1_dimissione_56_arm_1',
                  'scala_instabilit_clinica___2_t1_dimissione_ex_2_arm_1', 'scala_instabilit_clinica___2_t1_dimissione_56_arm_1',
                  'totale_mbi_t1_dimissione_ex_2_arm_1', 'totale_mbi_t1_dimissione_56_arm_1',
                  'mrs_t1_dimissione_ex_2_arm_1', 'mrs_t1_dimissione_56_arm_1',
                  'fac_t1_dimissione_ex_2_arm_1', 'fac_t1_dimissione_56_arm_1',
                  'totale_sppb_t1_dimissione_ex_2_arm_1', 'totale_sppb_t1_dimissione_56_arm_1',
                  'totale_mmse_t1_dimissione_ex_2_arm_1', 'totale_mmse_t1_dimissione_56_arm_1',
                  'lato_t0_arm_1', 'tipologia_ictus_ischemico_2_t0_arm_1',
                  "totale_tct_t1_dimissione_ex_2_arm_1", "totale_tct_t1_dimissione_56_arm_1",
                  "dimissione_presso_t1_dimissione_ex_2_arm_1", "dimissione_presso_t1_dimissione_56_arm_1",
                  "sede_lesione___1_t0_arm_1", "sede_lesione___2_t0_arm_1", "sede_lesione___3_t0_arm_1",
                  "sede_lesione___4_t0_arm_1", "sede_lesione___5_t0_arm_1", "sede_lesione___6_t0_arm_1",
                  "sede_lesione___7_t0_arm_1", "sede_lesione___8_t0_arm_1", "sede_lesione___9_t0_arm_1",
                  "sede_lesione___10_t0_arm_1"
)


# tolgo tutte le variabili non utili + elimino osservazioni con arrivo <0 o >40
clean_data <- raw_data %>%
  select(-all_of(var_to_omit)) %>%
  filter(is.na(arr_evento_t0_arm_1) | 
           (arr_evento_t0_arm_1 >= 0 & arr_evento_t0_arm_1 <= 40))


# --------------
# imputo valori:

id_imp <- c(85, 187, 378, 402, 411, 414, 422, 430, 444, 459, 489)
mbi_imp <- c(87, 91, 67, 67, 85, 100, 43, 95, 100, 98, 100)

for (i in seq_along(id_imp)){
  
  clean_data$mbi_t1[clean_data$record_id == id_imp[i]] <- mbi_imp[i]
  
}

# ---- 
# togliamo variabili con 50%+ di missing
perc_missing <- colMeans(is.na(clean_data)) * 100
vars_da_rimuovere <- names(perc_missing[perc_missing >= 50])

clean_data <- clean_data[, !(names(clean_data) %in% vars_da_rimuovere)]

ncol(strategy)
ncol(clean_data)


# DATASET PULITO
write.csv(clean_data, "clean_dataset.csv")


