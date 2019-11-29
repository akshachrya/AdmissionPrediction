library(googlesheets)
library(dplyr)

ttt <- gs_auth()

saveRDS(ttt,"ttt.rds")


Data <- gs_new("Admission Predict") %>% 
  gs_ws_rename(from = "Sheet1", to = "Admission Predict")   

Data <- Data %>% 
  gs_edit_cells(ws = "Admission Predict", input = cbind("Serial No.","GRE Score","TOEFL Score","University Rating","SOP","LOR","CGPA","Research","Chance of Admit" ),trim = T)  

