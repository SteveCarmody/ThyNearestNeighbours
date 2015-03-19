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

noCores <- 3
cl <- makeCluster(noCores, type="SOCK", outfile="")
registerDoSNOW(cl)

kOpt <- c(1,3,5,7,9,11,13,15,17,19,23,27,29,31,37,41,43,47,51)

resultsRF <- foreach(k = kOpt,
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
colnames(resultsRF) <- c("k", "Error")
results2 <- data.table(resultsRF)
results2 <- results2[with(results2, order(k)), ]

# save results in .csv file
write.csv(results2,"knn_testing.csv", row.names=F)

stopCluster(cl)