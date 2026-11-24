                                      # ------------------ #
                                      #   Adjusting Data   #
                                      # ------------------ #
 

# ----------------

raw_data <- readxl::read_xlsx(paste0(dirinput,"/strategy_valeriano_wide_solo_event.xlsx"))

raw_data <- raw_data[raw_data$record_id >= 68, ] # per ora rimaniamo fino al 68

# ------

# numeric MMSE

raw_data$totale_mmse_t0_arm_1 <- as.numeric(raw_data$totale_mmse_t0_arm_1)
raw_data$totale_mmse_t1_dimissione_56_arm_1 <- as.numeric(raw_data$totale_mmse_t1_dimissione_56_arm_1)
raw_data$totale_mmse_t1_dimissione_ex_2_arm_1 <- as.numeric(raw_data$totale_mmse_t1_dimissione_ex_2_arm_1)

# --------

# stroke type:
raw_data$hemorragic_stroke <- ifelse(
  is.na(raw_data$tipologia_ictus_t0_arm_1),
  NA,
  ifelse(raw_data$tipologia_ictus_t0_arm_1 == 2, 1, 0)
)
# ----
# tipologia_ictus_ischemico_2_t0_arm_1
# 1. lacunar syndromes
# 2. posterior circulation syndromes
# 3. total anterior circulation syndrome
# 4. partial anterior circulation syndrome

# lacunar syndromes as baseline:
raw_data$ictus_ischemico_circ_post_t0 <- ifelse(raw_data$tipologia_ictus_ischemico_2_t0_arm_1 == 2, 1, 0)
raw_data$ictus_ischemico_compl_circ_ant_t0 <- ifelse(raw_data$tipologia_ictus_ischemico_2_t0_arm_1 == 3, 1, 0)
raw_data$ictus_ischemico_parz_circ_ant_t0 <- ifelse(raw_data$tipologia_ictus_ischemico_2_t0_arm_1 == 4, 1, 0)

# ----
# lato_t0_arm_1
# 0. none 
# 1. right 
# 2. left 
# 3. bilateral

# nessun lato come reference
raw_data$lato_lesione_destro_t0  <- ifelse(raw_data$lato_t0_arm_1 == 1, 1, 0)
raw_data$lato_lesione_sinistro_t0 <- ifelse(raw_data$lato_t0_arm_1 == 2, 1, 0)
raw_data$lato_lesione_bilaterale_t0 <- ifelse(raw_data$lato_t0_arm_1 == 3, 1, 0)

# divide in two the variable 'sede lesione'
# 1 if trunk and brain (site from 7 to 10) <- infratentorial lesion
# 1 the rest <- supratentorial region
raw_data$lesione_sopratentoriale <- ifelse(
  rowSums(raw_data[, paste0("sede_lesione___", 1:6, "_t0_arm_1")], na.rm = TRUE) > 0,
  1, 0
)
raw_data$lesione_sottotentoriale <- ifelse(
  rowSums(raw_data[, paste0("sede_lesione___", 7:10, "_t0_arm_1")], na.rm = TRUE) > 0,
  1, 0
)


# -----------

# unite different variable codes

# Some patients switch from code 56 to 26, so they are discharged and readmitted, then discharged again
# The variables need to be merged by taking the latest value (i.e. the one with code 26)

# coalesce takes the first non-NA value among the variables, so it first checks whether there is a
# value for code 26, and otherwise falls back on the one with code 56

raw_data <- raw_data %>%
  mutate(
    
    mbi_t1   = coalesce(totale_mbi_t1_dimissione_ex_2_arm_1, totale_mbi_t1_dimissione_56_arm_1),
    
    tct_t1   = coalesce(totale_tct_t1_dimissione_ex_2_arm_1, totale_tct_t1_dimissione_56_arm_1),
    
    mmse_t1  = coalesce(totale_mmse_t1_dimissione_ex_2_arm_1, totale_mmse_t1_dimissione_56_arm_1),
    
    fac_t1   = coalesce(fac_t1_dimissione_ex_2_arm_1, fac_t1_dimissione_56_arm_1),
    
    sppb_t1  = coalesce(totale_sppb_t1_dimissione_ex_2_arm_1, totale_sppb_t1_dimissione_56_arm_1),
    
    mrs_t1   = coalesce(mrs_t1_dimissione_ex_2_arm_1, mrs_t1_dimissione_56_arm_1),
    
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

# clean-up variables which are not of interest anymore:

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
                  "sede_lesione___10_t0_arm_1", "dimissione_presso_t1", "scala_instabilita_clinica_2_t1",
                  "scala_instabilita_clinica_1_t1", "tipologia_ictus_t0_arm_1"
)


# Drop all unneeded variables + remove observations with arrival < 0 or > 40
clean_data <- raw_data %>%
  select(-all_of(var_to_omit)) %>%
  filter(is.na(arr_evento_t0_arm_1) | 
           (arr_evento_t0_arm_1 >= 0 & arr_evento_t0_arm_1 <= 40))


# --------------
# impute values from clinical additional data:

id_imp <- c(85, 187, 378, 402, 411, 414, 422, 430, 444, 459, 489)
mbi_imp <- c(87, 91, 67, 67, 85, 100, 43, 95, 100, 98, 100)

for (i in seq_along(id_imp)){
  
  clean_data$mbi_t1[clean_data$record_id == id_imp[i]] <- mbi_imp[i]
  
}

# ---- 
# remove all variables with 50%+ missings
perc_missing <- colMeans(is.na(clean_data)) * 100
vars_da_rimuovere <- names(perc_missing[perc_missing >= 50])

clean_data <- clean_data[, !(names(clean_data) %in% vars_da_rimuovere)]

ncol(raw_data)
ncol(clean_data)

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

clean_data <- clean_data %>%
  mutate(across(any_of(cat_vars), as.factor))

# clean dataset
write.csv(clean_data, paste0(dirintermediate,"/clean_dataset.csv"), row.names = FALSE)


