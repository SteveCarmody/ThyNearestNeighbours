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

set.seed(1234)
subset <- sample(seq(1,40000,1),40000) # random subsetting of training data

xTrain <- data1[subset,2:(ncol(data1)-1)]
xTrain <- sapply(xTrain,as.numeric)
yTrain <- data1[subset,56]
yTrain <- factor(yTrain)

xTest <- data1[-subset,2:(ncol(data1)-1)]
xTest <- sapply(xTest,as.numeric)
yTest <- data1[-subset,56]
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
