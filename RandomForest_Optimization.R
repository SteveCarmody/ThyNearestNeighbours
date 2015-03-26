### RANDOM FOREST PARAMETERS OPTIMIZATION USING PARALLEL

# load libraries
library(foreach)
library(snow)
library(doSNOW)
library(randomForest)
library(class)
library(data.table)

# loading the data
setwd("C:/Users/Monika/Documents/Master/MachineLearning/Competition")
training <- read.csv("Kaggle_Covertype_training.csv", header = TRUE, sep = ",")

## Preparing the data

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

# INITIALIZE PARALLEL COMPUTATION

noCores <- 10
cl <- makeCluster(noCores, type="SOCK", outfile="")
registerDoSNOW(cl)

# choosing parameters to optimize
nTreeOpt <- seq(100, 1500, 100)
mTryOpt <- seq(3, 30, 3) 

# equal the length of each parameter vector
nTreeList <- rep(nTreeOpt, length(1:(length(mTryOpt))))
mTryList <- rep(mTryOpt, length(1:(length(nTreeOpt))))
i <- 1 # starting the iterations

# do a foreach optimization of parameters
resultsRF <- foreach(n = nTreeList,
                     m = mTryList,
                     .combine = rbind, .packages = "randomForest") %dopar% {
                       print(i)
                       cat("ntree = ", n, " mtry = ", m, fill = TRUE)                     
                       
                       training_rf <- randomForest(x = xTrain, y = yTrain, xtest = NULL, ytest = NULL,
                                                   ntree = n,
                                                   mtry = m,
                                                   replace = TRUE,
                                                   nodesize = 1,
                                                   keep.forest = TRUE)
                       i <- i + 1
                       
                       predictedClasses <- predict(training_rf, xTest, type ="response")
                       
                       predictionError <- mean(predictedClasses != yTest)
                       
                       result <- c(n, m, predictionError)
                       
                     }

# putting the results in a data table
colnames(resultsRF) <- c("ntree", "mtry", "Error")
resultsRF <- data.table(resultsRF)

# ordering the table by ascending prediction error
resultsRF <- resultsRF[with(resultsRF, order(Error)), ]

# save results in .csv file
write.csv(resultsRF,"randomforest_testing.csv", row.names=F)

stopCluster(cl)