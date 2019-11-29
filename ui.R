library(shiny)
library(shinydashboard)
library(googlesheets)
library(ggplot2)
library(dplyr)
library(DT)
library(markdown)

shinyUI(
  
  
  dashboardPage(title="Dashboard",skin="blue",
                dashboardHeader(title="Admission Predict"
                                
                ), #end of dashboard header
                dashboardSidebar(
                  
                  sidebarMenu(
                    
                    menuItem("Predictor",tabName = "dashboard2",icon=icon("robot")),
                    menuSubItem("Survey Data",tabName = "dashboard3",icon = icon("database")),
                    menuSubItem("Algorithm Discussed",tabName="dashboard",icon=icon("dashboard"))
                    
                    
                    
                  )
                ), #end of dashboard siderbar
                dashboardBody(
                  tabItems(
                   
                    
                    tabItem(tabName = "dashboard",
                            box(h1("Algorithm Discussed",align="center"),width=12),br(),
                            fluidPage(
                            box(h5("Algorithm discussed is published in RPubs.To view the document"),tags$a(href="https://rpubs.com/akshachrya/admissionprediction", "Click here!"),htmlOutput("inc"),width=12)
                          
                            )
                            
                    ),
                    
                    tabItem(tabName = "dashboard2",
                            box(h1("Predictor",align="center"),width=12),br(),
                            fluidPage(
                              box(
                                textInput("grescore","GRE Scores",placeholder = "Input your GRE Score",value = 0),
                                h6("GRE Score should be range from 260 to 340"),br(),
                                textInput("toeflscore","TOEFL Score",placeholder = "Input your TOEFL Score",value=0),
                                h6("TOEFL Score should be range from 92 to 120"),br(),
                                sliderInput("unversityranking", "Unversity Ranking",min = 1, max =5,value = 3,step=1),
                                textInput("sop","Statement of Purpose",placeholder = "Input your SOP",value=0),
                                h6("SOP should be range from 1 to 5"),br(),
                                textInput("lor","Letter of Recommendation Strength",placeholder = "Input your Letter of Recommendation Strength",value = 0),
                                h6("LOR should be range from 1 to 5"),br(),
                                textInput("cgpa","Undergraduate GPA",placeholder = "Input your GPA",value=0),
                                h6("GPA should be range from 0 to 4"),br(),
                                sliderInput("research", "Research Experience",min = 0, max =1,value = 0,step=1)
                                
                              ),
                              box(
                                h3("Multiple Linear Regression"),
                                actionButton("predict1","Predict by Multiple Linear Regression"),
                              textOutput("t1"),
                              h3("Random Forest Regression"),
                              actionButton("predict2","Predict by Random Forest Regression"),
                              textOutput("t2"),
                              h3("Decision Tree Regression"),
                              actionButton("predict3","Predict by Decision Tree Regression"),
                              textOutput("t3"),
                              h3("Average Prediction"),
                              actionButton("predict4","Average Predict by Regressor"),
                              textOutput("t4")
                              )
                              
                            )
                            
                    ),
                    tabItem(tabName = "dashboard3",
                            fluidPage(
                              box(width=12,dataTableOutput('mytable')),br()
                              
                            )
                            
                    )
                    
                    
                    
                  ) #end of tabItems
                  
                  
                )#end of dashboard body
  ) #end of dashboard page
) #end of shiny UI