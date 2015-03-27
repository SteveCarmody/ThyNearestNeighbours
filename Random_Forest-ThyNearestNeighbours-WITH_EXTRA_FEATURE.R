# Load libraries

library(foreach)
library(snow)
library(doSNOW)
library(randomForest)
library(class)
library(data.table)

# Import and prepare data

data <- read.csv("Kaggle_Covertype_training.csv", header = TRUE, sep = ",")
data1 <- data

# Perfrom feature construction
data1 <- cbind(data1, EVDtH = data1$elevation-data1$ver_dist_hyd)         # 57
data1 <- cbind(data1, EHDtH = data1$elevation-(data1$hor_dist_hyd*0.2))   # 58
data1 <- cbind(data1, NewF4 = data1$elevation-(data1$hor_dist_road*-0.15))  # 59

data1 <- data1[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                  21,28,39,48,57,58,59,56)]

set.seed(1234)
subset <- sample(seq(1,50000,1),40000) # random subsetting of training data

xTrain <- data1[subset,2:(ncol(data1)-1)]
xTrain <- sapply(xTrain,as.numeric)
yTrain <- data1[subset,23]
yTrain <- factor(yTrain)

xTest <- data1[-subset,2:(ncol(data1)-1)]
xTest <- sapply(xTest,as.numeric)
yTest <- data1[-subset,23]
yTest <- factor(yTest)

##################################################

noCores <- 4
cl <- makeCluster(noCores, type="SOCK", outfile="")
registerDoSNOW(cl)

nTreeOpt <- seq(50, 1550, 100)
mTryOpt <- seq(2, 15, 1) 
nSizeOpt <- seq(1, 1, 1)  
nTreeList <- rep(nTreeOpt, length(1:(length(mTryOpt)*length(nSizeOpt))))
mTryList <- rep(mTryOpt, length(1:(length(nTreeOpt)*length(nSizeOpt))))
nSizeList <- rep(nSizeOpt, length(1:(length(mTryOpt)*length(nTreeOpt))))
i <- 1

resultsRF <- foreach(n = nTreeList,
                     m = mTryList,
                     s = nSizeList,
                     .combine = rbind, .packages = "randomForest") %dopar% {
                       print(i)
                       cat("ntree = ", n, " mtry = ", m, " nodesize = ", s, fill = TRUE)                     
                       
                       training_rf <- randomForest(x = xTrain, y = yTrain, xtest = NULL, ytest = NULL,
                                                   ntree = n,
                                                   mtry = m,
                                                   replace = TRUE,
                                                   nodesize = s,
                                                   keep.forest = TRUE)
                       i <- i + 1
                       
                       predictedClasses <- predict(training_rf, xTest, type ="response")
                       
                       predictionError <- mean(predictedClasses != yTest)
                       
                       result <- c(n, m, s, predictionError)
                       
                     }

# puts results in a data table
colnames(resultsRF) <- c("ntree", "mtry", "nodesize", "Error")
results2 <- data.table(resultsRF)
results2 <- results2[with(results2, order(Error)), ]

# save results in .csv file
write.csv(results2,"randomforest_testing.csv", row.names=F)

stopCluster(cl)
