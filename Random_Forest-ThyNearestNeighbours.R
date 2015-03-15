library(foreach)
library(snow)
library(doSNOW)
library(randomForest)
library(class)
library(data.table)

data <- read.csv("Kaggle_Covertype_training.csv", header = TRUE, sep = ",")
data1 <- data

training <- data1[1:40000,2:(ncol(data1)-1)]
training <- sapply(training,as.numeric)
xTrain <- training
test <- data1[40001:50000,2:(ncol(data1)-1)]
test <- sapply(test,as.numeric)
xTest <- test
yTrain <- data1[1:40000,56]
yTrain <- factor(yTrain)
yTest <- data1[40001:50000,56]
yTest <- factor(yTest)

##################################################

noCores <- 35
cl <- makeCluster(noCores, type="SOCK", outfile="")
registerDoSNOW(cl)

nTreeOpt <- seq(501, 2000, 200)
mTryOpt <- seq(20, 50, 5) 
nSizeOpt <- seq(1, 100, 20)  
nTreeList <- rep(nTreeOpt, length(1:(length(mTryOpt)*length(nSizeOpt))))
mTryList <- rep(mTryOpt, length(1:(length(nTreeOpt)*length(nSizeOpt))))
i <- 1

resultsRF <- foreach(n = nTreeList,
                     m = mTryList,
                     .combine = rbind, .packages = "randomForest") %dopar% {
                       print(i)
                       cat("ntree = ", n, " mtry = ", m, " nodesize = ", 1, fill = TRUE)                     
                       
                       training_rf <- randomForest(x = xTrain, y = yTrain, xtest = NULL, ytest = NULL,
                                                   ntree = n,
                                                   mtry = m,
                                                   replace = TRUE,
                                                   nodesize = 1,
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

stopCluster(cl)
