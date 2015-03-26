##### Thy Nearest Neighbours #####

### TESTING AND PREDICTING THE CLASSES USING RANDOM FOREST AND K-NN ###

# loading te data
setwd("C:/Users/Monika/Documents/Master/MachineLearning/Competition")

training <- read.csv("Kaggle_Covertype_training.csv", header = TRUE, sep = ",")
test <- read.csv("Kaggle_Covertype_test.csv", header = TRUE, sep = ",")

# loading necessary packages
 
library(foreach)
library(snow)
library(doSNOW)
library(randomForest)
library(class)
library(data.table)


## Preparing the data

# Perform feature construction in the training set
training <- cbind(training, EVDtH_1 = training$elevation-training$ver_dist_hyd)
training <- cbind(training, EHDtH_2 = training$elevation-(training$hor_dist_hyd*0.2))
training <- training[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                  31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,58,56)]

# Perform feature construction in the test set
test <- cbind(test, EVDtH_1 = test$elevation-test$ver_dist_hyd)
test <- cbind(test, EHDtH_2 = test$elevation-(test$hor_dist_hyd*0.2))

############################################################################
############################################################################

##### RANDOM FOREST ####

### TRAINING AND TESTING THE MODEL

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

##############################################################################

### PREDICTING THE CLASSES

# creating the training set
xTrain <- training[,2:(ncol(training)-1)]
xTrain <- sapply(xTrain,as.numeric)
yTrain <- training[,58]
yTrain <- factor(yTrain)

# creating the test set
xTest <- test[,(2:(ncol(test)))]
xTest <- sapply(xTest,as.numeric)

##################################################

# predicting the classes using Random Forest Model with optimal parameters
# (the ones that gave the lowest prediction error)

training_rf <- randomForest(x = xTrain, y = yTrain, xtest = NULL, ytest = NULL,
                            ntree = 1350,
                            mtry = 27,
                            replace = TRUE,
                            nodesize = 1,
                            keep.forest = TRUE)

predictedClasses <- predict(training_rf, xTest, type ="response")

# save results in .csv file
write.csv(predictedClasses,"rf_prediction.csv", row.names=F)

############################################################################
############################################################################

### K-NN

### TRAINING AND TESTING THE MODEL

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

# Initialize parallel computation

noCores <- 3
cl <- makeCluster(noCores, type="SOCK", outfile="")
registerDoSNOW(cl)

# choosing parameters to optimize
kOpt <- c(1,3,5,7,9,11,13,15,17,19,23,27,29,31,37,41,43,47,51)

# do a foreach optimization of parameters
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

# putting the results in a data table
colnames(resultsKNN) <- c("k", "Error")
resultsKNN <- data.table(resultsKNN)

# ordering the results by ascending prediction error
resultsKNN <- resultsKNN[with(resultsKNN, order(Error)), ]

# save results in .csv file
write.csv(resultsKNN,"knn_testing.csv", row.names=F)

stopCluster(cl)

##############################################################################

### PREDICTING THE CLASSES

# creating the training set
xTrain <- training[,2:(ncol(training)-1)]
xTrain <- sapply(xTrain,as.numeric)
yTrain <- training[,58]
yTrain <- factor(yTrain)

# creating the test set
xTest <- test[,(2:(ncol(test)))]
xTest <- sapply(xTest,as.numeric)

##################################################

# predicting the classes using k-NN Model with optimal parameters
# (the ones that gave the lowest prediction error)

predictedClasses <- knn(train=xTrain, test=xTest, cl=yTrain, k=1)

# save results in .csv file
write.csv(predictedClasses,"knn_prediction.csv", row.names=F)