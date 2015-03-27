##### Thy Nearest Neighbours #####

### TESTING AND PREDICTING THE CLASSES USING RANDOM FOREST AND K-NN ###
###
### PLEASE PLACE THIS SCRIPT AND THE DATA SET IN THE SAME DIRECTORY
### AND SET THE WORKING DIRECTORY TO SOURCE FILE LOCATION

# loading the data
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
training <- cbind(training, NewF4 = training$elevation-(training$hor_dist_road*-0.15))
training <- training[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                     21,28,39,48,57,58,59,56)] # select features for the analysis

# Perform feature construction in the test set
test <- cbind(test, EVDtH_1 = test$elevation-test$ver_dist_hyd)
test <- cbind(test, EHDtH_2 = test$elevation-(test$hor_dist_hyd*0.2))
test <- cbind(test, NewF4 = test$elevation-(test$hor_dist_road*-0.15))
test <- test[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                21,28,39,48,56,57,58)] # select features for the analysis

# random subsetting of the data for testing
set.seed(1234)
subset <- sample(seq(1,50000,1),40000)

# creating the training set
xTrain <- training[subset,2:(ncol(training)-1)]
xTrain <- sapply(xTrain,as.numeric)
yTrain <- training[subset,23]
yTrain <- factor(yTrain)

# creating the test set
xTest <- training[-subset,2:(ncol(training)-1)]
xTest <- sapply(xTest,as.numeric)
yTest <- training[-subset,23]
yTest <- factor(yTest)

# preparing the test set for prediction
xTest <- test[,(2:(ncol(test)))]
xTest <- sapply(xTest,as.numeric)

############################################################################
############################################################################

##### RANDOM FOREST ####

### TRAINING AND TESTING THE MODEL

# initialising parallel computation
noCores <- 3
cl <- makeCluster(noCores, type="SOCK", outfile="")
registerDoSNOW(cl)

# choosing parameters to optimize
nTreeOpt <- seq(50, 1550, 100)
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

# select the optimal parameters based on test error
best_ntree <- resultsRF$ntree[1]
best_mtry <- resultsRF$mtry[1]

# do the prediction
training_rf <- randomForest(x = xTrain, y = yTrain, xtest = NULL, ytest = NULL,
                            ntree = best_ntree,
                            mtry = best_mtry,
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

# select the optimal parameters based on test error
best_k <- resultsKNN$k[1]

# do the prediction
predictedClasses <- knn(train=xTrain, test=xTest, cl=yTrain, k=best_k)

# save results in .csv file
write.csv(predictedClasses,"knn_prediction.csv", row.names=F)


