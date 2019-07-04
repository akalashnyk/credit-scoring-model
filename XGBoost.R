rm(list = ls()) 
library(Hmisc)
library(xgboost)
library(class)
library(ElemStatLearn)
library(DT)
library(caret)
library(clusterSim)
library(dplyr)
library(OneR)
library(caTools)
library(e1071)
library(RcmdrMisc)
library(onehot)
library(BCA)

######Loading the data and Data preparation

setwd("~/Documents/Study/2nd sem/PWC/New data/Materials/")
df <- read.csv('final_data.csv', dec = '.')

#Converting df$Average_income into numeric
df$Average_income <- as.character(df$Average_income)
replaceCommas<-function(x){
  x<-as.numeric(gsub("\\,", "", x))
}
df$Average_income <- replaceCommas(df$Average_income)

df$Divorce_number_1000 <- as.character(df$Divorce_number_1000)
df$Divorce_number_1000 <- scan(text=df$Divorce_number_1000, dec=",", sep=".")
df$Divorce_number_1000 <- as.numeric(df$Divorce_number_1000)
#df$Divorce_number_1000 <- trunc(df$Divorce_number_1000)
#hist(df$Divorce_number_1000)

test <- df[is.na(df$DefFlag),]
mydata <- df[!is.na(df$DefFlag),]
set.seed(123)

rand <- sample(1:nrow(mydata), 0.8*nrow(mydata))
train <- mydata[rand,]
valid <- mydata[-rand,]

train <- within(train, {
  Home_status <- Recode(Home_status, '"Owner"=1; "Rental"=0', as.factor=FALSE)
  Car_status <- Recode(Car_status, '"Owner"=1; "No"=0', as.factor=FALSE )
})
Application_ID_train <- train$Application_ID

hot <- onehot(train, stringsAsFactors = TRUE, addNA = FALSE, max_levels = 4)
data <- predict(hot, train)
data <- as.data.frame(data)
train <- data
train$Application_ID <- Application_ID_train

valid <- within(valid, {
  Home_status <- Recode(Home_status, '"Owner"=1; "Rental"=0', as.factor=FALSE)
  Car_status <- Recode(Car_status, '"Owner"=1; "No"=0', as.factor=FALSE )
})
Application_ID_valid <- valid$Application_ID

hot <- onehot(valid, stringsAsFactors = TRUE, addNA = FALSE, max_levels = 4)
data <- predict(hot, valid)
data <- as.data.frame(data)
valid <- data
valid$Application_ID <- Application_ID_valid

test <- within(test, {
  Home_status <- Recode(Home_status, '"Owner"=1; "Rental"=0', as.factor=FALSE)
  Car_status <- Recode(Car_status, '"Owner"=1; "No"=0', as.factor=FALSE )
})
Application_ID_test <- test$Application_ID

hot <- onehot(test, stringsAsFactors = TRUE, addNA = FALSE, max_levels = 4)
data <- predict(hot, test)
data <- as.data.frame(data)
test <- data
test$Application_ID <- Application_ID_test

mydata <- within(mydata, {
  Home_status <- Recode(Home_status, '"Owner"=1; "Rental"=0', as.factor=FALSE)
  Car_status <- Recode(Car_status, '"Owner"=1; "No"=0', as.factor=FALSE )
})
Application_ID_mydata <- mydata$Application_ID

hot <- onehot(mydata, stringsAsFactors = TRUE, addNA = FALSE, max_levels = 4)
data <- predict(hot, mydata)
data <- as.data.frame(data)
mydata <- data
mydata$Application_ID <- Application_ID_mydata


#Droping Charachter from DF
test$Application_ID <- NULL
train$Application_ID <- NULL
valid$Application_ID <- NULL
mydata$Application_ID <- NULL

####Building the model (XGBoost)

model <- xgboost(data = as.matrix(train[-99]), label = train$DefFlag, nrounds =36)

pred <- predict(model, newdata = as.matrix(valid[-99]))
pred_v <- predict(model, newdata = as.matrix(mydata[-99]))

####GINI Function

Gini_value <- function(
  score, #prediction from model
  target #target binary variable (0/1)
){
  
  default <- ifelse(target==0, 'G','B')
  d <- data.frame(FY = default, SCORE = score)
  s <- table(d[,2],d[,1])
  sHeader <- colnames(s)
  s <- cbind(s,apply(s,2,cumsum))
  colnames(s) <- c(sHeader,paste(sHeader,"_cum",sep=""))
  s <- cbind(s , s[,"B_cum"]/max(s[,"B_cum"]) , s[,"G_cum"]/max(s[,"G_cum"]),
             diff(c(0,s[,"G_cum"]))/max(s[,"G_cum"]))
  colnames(s)<-c(sHeader,
                 paste(sHeader,"_cum",sep=""),
                 c("%cum_bad","%cum_good","%good"))
  p <- 1:nrow(s)
  s <- cbind(s, c( s[1,7] , s[p[-1],7]+s[(p-1)[-1],7]) ) 
  s <- cbind(s, c(0,s[1:(nrow(s)-1),"%cum_bad"]))
  colnames(s)[length(colnames(s))] <- "%cum_bad_prev"
  auc <- sum(s[,"%good"]*(s[,"%cum_bad"]+s[,"%cum_bad_prev"])*0.5)
  gini_value <- abs( 2 * ( auc - 0.5 ) )
  return(gini_value)
  
}
Gini_value(pred, valid$DefFlag)
Gini_value(pred_v, mydata$DefFlag)
#Creating score prediction
mypred <- predict(model, newdata = as.matrix(test[-99]))
mypred2 <- predict(model, newdata = as.matrix(mydata[-99]))

#Bringing back Application_ID to a data Frame
test$Application_ID <- Application_ID_test 
mydata$Application_ID <- Application_ID_mydata

#Creating prediction DF for 70% and for 30% of data
Prediction <- data.frame(Application_ID = test$Application_ID, Score = mypred)
Prediction2 <- data.frame(Application_ID = mydata$Application_ID, Score = mypred2)

#Combaning both predictions 
pred_final <- rbind(Prediction,Prediction2)

#save file
write.csv(pred_final, "Output.csv", row.names = F)


