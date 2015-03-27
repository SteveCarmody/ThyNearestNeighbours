library(foreach)
library(snow)
library(doSNOW)
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

noCores <- 8
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
write.csv(results2,"knn_testing-EXTRA_FEATURE.csv", row.names=F)

stopCluster(cl)
