# Optimization of k parameter for kNN

# Loading libraries

library(foreach)
library(snow)
library(doSNOW)
library(randomForest)
library(class)
library(data.table)

# Import and prepare data

setwd("C:/Users/Monika/Documents/Master/MachineLearning/Competition")
training <- read.csv("Kaggle_Covertype_training.csv", header = TRUE, sep = ",")

# Perform feature construction
training <- cbind(training, EVDtH_1 = training$elevation-training$ver_dist_hyd)
training <- cbind(training, EHDtH_2 = training$elevation-(training$hor_dist_hyd*0.2))
training <- training[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,58,56)]

# random subsetting of the data
set.seed(1234)
subset <- sample(seq(1,40000,1),40000)

# creating the training set
xTrain <- training[subset,2:(ncol(training)-1)]
xTrain <- sapply(xTrain,as.numeric)
yTrain <- training[subset,58]
yTrain <- factor(yTrain)

# creating the test set
xTest <- training[-subset,2:(ncol(training)-1)]
xTest <- sapply(xTest,as.numeric)
yTest <- training[-subset,58]
yTest <- factor(yTest)

##################################################
# INITIALIZE PARALLEL COMPUTATION

noCores <- 3
cl <- makeCluster(noCores, type="SOCK", outfile="")
registerDoSNOW(cl)

# choosing parameters to optimize
kOpt <- c(1,3,5,7,9,11,13,15,17,19,23,27,29,31,37,41,43,47,51)

resultsKNN <- foreach(k = kOpt,
                     .combine = rbind, .packages = "class") %dopar% {
                       
                       cat("k = ", k, fill = TRUE)
                       
                       # kNN results
                       predictedClasses <- knn(train=xTrain, test=xTest, cl=yTrain, k=k)
                       
                       predictedClasses <- factor(predictedClasses, levels=c(1,2,3,4,5,6,7))
                       yTest <- factor(yTest, levels=c(1,2,3,4,5,6,7))
                       
                       predictionError <- mean(predictedClasses != yTest)
                       
                       result <- c(k, predictionError)
                     }

# puts results in a data table
colnames(resultsKNN) <- c("k", "Error")
resultsKNN <- data.table(resultsKNN)
resultsKNN <- resultsKNN[with(resultsKNN, order(Error)), ]

# save results in .csv file
write.csv(resultsKNN,"knn_testing.csv", row.names=F)

stopCluster(cl)