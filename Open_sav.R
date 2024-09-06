library(haven)
TX_tyrimo_darbinis_last_1_ <- read_sav("C:/Users/Legion/Desktop/pk Aurelija/TX_tyrimo_darbinis_last.sav")
View(TX_tyrimo_darbinis_last_1_)

dfexp<-as.data.frame(TX_tyrimo_darbinis_last_1_)
library("writexl")
write_xlsx(dfexp,"C:/Users/Legion/Desktop/pk Aurelija/case_data_R.xlsx")