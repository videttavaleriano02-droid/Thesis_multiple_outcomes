                                      # ------------------ #
                                      #   Adjusting Data   #
                                      # ------------------ #

#' impute values of mbi,fac, ... at dimission
#' capire perchè ho meno variabili
#' creare variabile in cui si considera codice 26 se diverso da codice 56 
#' (paziente dimesso e riammesso con altro codice)
#' 

# ----------------

raw_data <- readxl::read_xlsx('strategy_valeriano_wide_solo_event.xlsx')

raw_data <- raw_data[-raw_data$record_id<68] # capire fino a che id eliminare


# creating outcome variables:
