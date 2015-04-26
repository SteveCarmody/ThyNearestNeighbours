setwd('/Users/StephenCarmody/Documents/Machine Learning Competition/ThyNearestNeighbours')
rm( list=ls() )

if (!require("class")) install.packages("class")
library(class)

if (!require("foreach")) install.packages("foreach")
library(foreach)

if (!require("doSNOW")) install.packages("doSNOW")
library(doSNOW)

if (!require("deepnet")) install.packages("deepnet")
library(deepnet)

if (!require("randomForest")) install.packages("randomForest")
library(randomForest)

if (!require("e1071")) install.packages("e1071")
library(e1071)

library(data.table)

type <- "train" # predict | train


# read in intial data
data.raw <- read.csv("Kaggle_Covertype_training.csv", head = TRUE, sep = ',')
data.raw.test<- matrix()

if(type == "predict")
{
  data.raw.test <- read.table("Kaggle_Covertype_test.csv", head = TRUE, sep = ',')
}

# Limit data if necesary
#data.raw <- head(data.raw, 50000)
sampleIndex <- sample(45000,5000)

# Perfrom feature construction
data.raw <- cbind(data.raw, EVDtH = data.raw$elevation-data.raw$ver_dist_hyd)
data.raw <- cbind(data.raw, EHDtH = data.raw$elevation-(data.raw$hor_dist_hyd*0.2))
data.raw <- data.raw[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                        31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,58,56)]

if(type == "predict")
{
  data.raw.test <- cbind(data.raw.test, EVDtH = data.raw$elevation-data.raw$ver_dist_hyd)
  data.raw.test <- cbind(data.raw.test, EHDtH = data.raw$elevation-(data.raw$hor_dist_hyd*0.2))
}


scaleData <- function(data){
    
  data.scaled <- data.frame()
  
  for(i in 2:11){
    data.raw[,i] <- as.double(scale(as.numeric(data.raw[,i]), center = TRUE, scale = TRUE))
  }
  data.scaled <- data.raw
  
  return(data.scaled)
}


if(type == "train"){
  data.train <- data.raw[-sampleIndex,]
  data.test <- data.raw[sampleIndex,]
  
  data.train.X <- data.train[,2:(ncol(data.train)-1)]
  data.train.Y <- data.train[,ncol(data.train)]
  
  data.test.X <- data.test[,2:(ncol(data.test)-1)]
  data.test.Y <- data.test[,ncol(data.test)] 
  
}else{
  if(type == "predict"){
  
  data.test <- (data.raw.test)
  data.train <- data.raw
    
  data.train.X <- data.train[,2:(ncol(data.train)-1)]
  data.train.Y <- data.train[,ncol(data.train)]
  
  data.test.X <- data.test[,2:(ncol(data.test))]
  }
}


testPredictions <- knn(train=data.train.X, test=data.test.X, cl=data.train.Y, k=1)

if(type == "train")
{
  testError <- mean(testPredictions != data.test.Y)
  print(testError)
}

id <- seq(from = 50001, to = 150000)
Cover_Type <- testPredictions
testPredictions.FC <- cbind(id, Cover_Type)
write.csv(testPredictions.FC ,"knnPredictions_FC.csv",row.names = FALSE)





# # for prediction
# 
# # Split data and label
# data.train.X <- data.train[,2:(ncol(data.train)-1)]
# data.train.Y <- data.train[,ncol(data.train)]
# 
# # Test data for training
# data.test.X <- data.test[,2:(ncol(data.test)-1)]
# data.test.Y <- data.test[,ncol(data.test)]
# 
# # Test data fro prediction
# data.test.X <- data.test[,2:(ncol(data.test))]



# # Deepnet
# dnn <- dbn.dnn.train(data.train.X, data.train.Y, c(1))
# dnn.results <- nn.test(dnn, data.test.X, data.test.Y, t = 0.5)
# predictedClasses.Dnn <- nn.predict(dnn, data.test.X)
# errorTest.Dnn <- mean(predictedClasses.Dnn!=data.test.Y)
# 
# 
# svm.model <- svm(x = data.train.X, y = factor(data.train.Y), cost = 2, gamma = 1, scale = FALSE)
# svm.pred <- predict(svm.model, data.test.X)
# errorTestSVM <- mean(svm.pred!=data.test.Y,type = "class")
# 
# # writes scaled training data and error results to csv files
# write.table(data.scaled ,"training_Data_scaled.csv", sep = ",")
# write.table(results ,"classification_error_estimate.csv", sep = ",")
# write.csv(testPredictions ,"knnPredictions.csv", sep = ",")
# 




