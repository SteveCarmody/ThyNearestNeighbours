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

# Perform feature construction
data1 <- cbind(data1, EVDtH_1 = data1$elevation-data1$ver_dist_hyd)
data1 <- cbind(data1, EHDtH_2 = data1$elevation-(data1$hor_dist_hyd*0.2))
data1 <- data1[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                  31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,58,56)]


set.seed(1234)
subset <- sample(seq(1,40000,1),40000) # random subsetting of training data

xTrain <- data1[subset,2:(ncol(data1)-1)]
xTrain <- sapply(xTrain,as.numeric)
yTrain <- data1[subset,58]
yTrain <- factor(yTrain)

xTest <- data1[-subset,2:(ncol(data1)-1)]
xTest <- sapply(xTest,as.numeric)
yTest <- data1[-subset,58]
yTest <- factor(yTest)

##################################################

noCores <- 10
cl <- makeCluster(noCores, type="SOCK", outfile="")
registerDoSNOW(cl)

nTreeOpt <- seq(100, 1500, 100)
mTryOpt <- seq(3, 30, 3) 
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
