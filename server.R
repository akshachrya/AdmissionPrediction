library(shiny)
library(shinydashboard)
library(googlesheets)
library(DT)
library(dplyr)
library(ggplot2)
library(rpart)
library(randomForest)

cgpatogpa <- function(cgpa) {
  if (cgpa<4) {
    gpa=2
  }
  if (cgpa>=4 & cgpa<5) {
    gpa=3
  }
  if (cgpa>=5 & cgpa<6) {
    gpa=3.5
  } 
  if(cgpa>=6 & cgpa<=10) {
    gpa=4
  }
  return(gpa)
}

dataready <- function(x){
  df<-x
  colnames(df) <- c("Serial Number","GRE Scores","TOEFL Scores","University Ranking","SOP","LOR","CGPA","Research","Chance of Admit") 
  df <- select(df,-c("Serial Number"))
}

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}


shinyServer(function(input,output,session){

  
  
  googlesheets::gs_auth(token = "ttt.rds")
  sheet_key <- "1U96KL565SE1ALz1pKRzmMgyrTBmKb99MpfBSzSO60KE"
  
  data <- googlesheets::gs_key(sheet_key) %>%
    gs_read(ws = "Admission Predict", range = cell_rows(1:501))
 
  for (row in 1:nrow(data['CGPA'])) { 
    data$CGPA[row] <- cgpatogpa(data$CGPA[row]) 
  }
  observeEvent(input$refresh,{
    session$reload()
    
  })
  
  
  ###Algorithm Discussed
  getPage<-function() {
    return((browseURL('https://rpubs.com/akshachrya/admissionprediction')))
  }
  output$inc<-renderUI({
    getPage()
  })

  
  
  A <- eventReactive(input$predict1,{
      df <- dataready(data)
      # atest <- c(337,118,4,4.5,4.5,9.65,1,0)
      atest <- c(as.numeric(input$grescore),as.numeric(input$toeflscore),as.numeric(input$unversityranking),as.numeric(input$sop),as.numeric(input$lor),
                 as.numeric(input$cgpa),as.numeric(input$research),as.numeric(0))
      df<- rbind(df,atest)
      dfNorm <- df
      dfNorm <- as.data.frame(lapply(df, normalize))
      test <- dfNorm[501,]
      dfNorm <- dfNorm[1:500,]
      #80% of the sample size
      smp_size <- floor(1 * nrow(dfNorm))
      #Set the seed to make your partition reproducible
      set.seed(123)
      train_ind <- sample(seq_len(nrow(dfNorm)), size = smp_size)
      train <- dfNorm[train_ind, ]
      model1 = lm(Chance.of.Admit~., data=train)
      model2 = update(model1,~.-University.Ranking-SOP)
      pred1 <- predict(model2, newdata =test)
      value1 <- data.frame(select(test,-c("Chance.of.Admit")),pred1)
      colnames(value1) <- c("GRE Scores","TOEFL Scores","University Ranking","SOP","LOR","CGPA","Research","Prediction")
      value1$Prediction*100
    
      
  })
  
  B <- eventReactive(input$predict2,{
    df <- dataready(data)
    # atest <- c(337,118,4,4.5,4.5,9.65,1,0)
    atest <- c(as.numeric(input$grescore),as.numeric(input$toeflscore),as.numeric(input$unversityranking),as.numeric(input$sop),as.numeric(input$lor),
               as.numeric(input$cgpa),as.numeric(input$research),as.numeric(0))
    df<- rbind(df,atest)
    dfNorm <- as.data.frame(lapply(df, normalize))
    test <- dfNorm[501,]
    dfNorm <- dfNorm[1:500,]
    #80% of the sample size
    smp_size <- floor(0.8 * nrow(dfNorm))
    #Set the seed to make your partition reproducible
    set.seed(123)
    train_ind <- sample(seq_len(nrow(dfNorm)), size = smp_size)
    train <- dfNorm[train_ind, ]
    require(randomForest)
    df.rf=randomForest(Chance.of.Admit ~ . , data =train)
    pred2 <- predict(df.rf, newdata = test)
    value2 <- data.frame(select(test,-c("Chance.of.Admit")),pred2)
    colnames(value2) <- c("GRE Scores","TOEFL Scores","University Ranking","SOP","LOR","CGPA","Research","Prediction")
    value2$Prediction*100
    
  })
  
  C <- eventReactive(input$predict3,{
    df <- dataready(data)
    # atest <- c(337,118,4,4.5,4.5,9.65,1,0)
    atest <- c(as.numeric(input$grescore),as.numeric(input$toeflscore),as.numeric(input$unversityranking),as.numeric(input$sop),as.numeric(input$lor),
               as.numeric(input$cgpa),as.numeric(input$research),as.numeric(0))
    df<- rbind(df,atest)
    dfNorm <- as.data.frame(lapply(df, normalize))
    test <- dfNorm[501,]
    dfNorm <- dfNorm[1:500,]
    #80% of the sample size
    smp_size <- floor(0.8 * nrow(dfNorm))
    #Set the seed to make your partition reproducible
    set.seed(123)
    train_ind <- sample(seq_len(nrow(dfNorm)), size = smp_size)
    train <- dfNorm[train_ind, ]
    require(rpart)
    df.dt=rpart(Chance.of.Admit ~GRE.Scores+TOEFL.Scores+University.Ranking+SOP+LOR+CGPA+Research, method="anova",data =train)
    pred3 <- predict(df.dt, newdata = test)
    value3 <- data.frame(select(test,-c("Chance.of.Admit")),pred3)
    colnames(value3) <- c("GRE Scores","TOEFL Scores","University Ranking","SOP","LOR","CGPA","Research","Prediction")
    value3$Prediction*100
    
  })
  
  D <- eventReactive(input$predict4,{
    df <- dataready(data)
    # atest <- c(337,118,4,4.5,4.5,9.65,1,0)
    atest <- c(as.numeric(input$grescore),as.numeric(input$toeflscore),as.numeric(input$unversityranking),as.numeric(input$sop),as.numeric(input$lor),
               as.numeric(input$cgpa),as.numeric(input$research),as.numeric(0))
    df<- rbind(df,atest)
    dfNorm <- as.data.frame(lapply(df, normalize))
    test <- dfNorm[501,]
    dfNorm <- dfNorm[1:500,]
    #80% of the sample size
    smp_size <- floor(0.8 * nrow(dfNorm))
    #Set the seed to make your partition reproducible
    set.seed(123)
    train_ind <- sample(seq_len(nrow(dfNorm)), size = smp_size)
    train <- dfNorm[train_ind, ]
    
    model1 = lm(Chance.of.Admit~., data=train)
    model2 = update(model1,~.-University.Ranking-SOP)
    pred1 <- predict(model2, newdata =test)
    value1 <- data.frame(select(test,-c("Chance.of.Admit")),pred1)
    colnames(value1) <- c("GRE Scores","TOEFL Scores","University Ranking","SOP","LOR","CGPA","Research","Prediction")
    value1$Prediction*100
    
    require(randomForest)
    df.rf=randomForest(Chance.of.Admit ~ . , data =train)
    pred2 <- predict(df.rf, newdata = test)
    value2 <- data.frame(select(test,-c("Chance.of.Admit")),pred2)
    colnames(value2) <- c("GRE Scores","TOEFL Scores","University Ranking","SOP","LOR","CGPA","Research","Prediction")
    value2$Prediction
    
    require(rpart)
    df.dt=rpart(Chance.of.Admit ~GRE.Scores+TOEFL.Scores+University.Ranking+SOP+LOR+CGPA+Research, method="anova",data =train)
    pred3 <- predict(df.dt, newdata = test)
    value3 <- data.frame(select(test,-c("Chance.of.Admit")),pred3)
    colnames(value3) <- c("GRE Scores","TOEFL Scores","University Ranking","SOP","LOR","CGPA","Research","Prediction")
    value3$Prediction
    
    mean(value1$Prediction,value2$Prediction,value3$Prediction)
    
  })
  
  output$t1<- renderText({
    A()
    })
  output$t2<- renderText({
    B()
  })
  output$t3<- renderText({
    C()
  })
  
  output$t4<- renderText({
    D()
  })

  ###Final Dashboard (Data)
  output$mytable =DT::renderDataTable({
    mytable <-data.frame(data)
    colnames(mytable) <-c("Serial No.","GRE Score","TOEFL Score","University Rating","SOP","LOR","GPA","Research","Chance of Admit")
    mytable <- datatable(data=mytable)
  })                                                
  
  
})