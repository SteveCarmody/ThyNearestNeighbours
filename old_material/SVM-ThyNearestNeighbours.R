# Load libraries

library(foreach)
library(snow)
library(doSNOW)
library(e1071)
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

# Scale data
for(i in 2:11){
  data1[,i] <- as.double(scale(as.numeric(data1[,i]), center = TRUE, scale = TRUE))
}
for(i in 56:57){
  data1[,i] <- as.double(scale(as.numeric(data1[,i]), center = TRUE, scale = TRUE))
}

# Calculate class weights
wts <- summary(factor(data1[,ncol(data1)]))
wts <- wts / nrow(data1)

set.seed(1234)
#subset <- sample(seq(1,40000,1),40000) # random subsetting of training data
subset.train <- seq(1,5000,1)
subset.test <- seq(5001,5100,1)

Train <- data1[subset.train,2:(ncol(data1))]
Train <- sapply(Train,as.numeric)

Test <- data1[subset.test,2:(ncol(data1))]
Test <- sapply(Test,as.numeric)

##################################################

noCores <- 3
cl <- makeCluster(noCores, type="SOCK", outfile="")
registerDoSNOW(cl)

costs <- c(0.1,0.5,1,2,5,10)
gammas <- c(0.5,1,2,3,4)
costsList <- rep(costs, length(1:(length(costs)*length(gammas))))
gammasList <- rep(gammas, length(1:(length(costs)*length(gammas))))
i <- 1

resultsSVM <- foreach(c = costsList,
                      g = gammasList,
                       .combine = rbind, .packages = "e1071") %dopar% {
                       print(i)
                       cat("cost = ", c, fill = TRUE)                     
                       
                       svmfit <- svm(factor(Cover_Type) ~ ., 
                                     data=Train, 
                                     type='C-classification', 
                                     kernel = "polynomial", 
                                     cost = c, 
                                     gamma = g,
                                     class.weights = wts,
                                     scale=FALSE)
                       
                       i <- i + 1
                       
                       predictedClasses <- predict(svmfit, Test)
                       
                       predictionError <- mean(predictedClasses != Test[,57])
                       
                       result <- c(c, g, predictionError)
                       
                     }

# puts results in a data table
colnames(resultsSVM) <- c("cost", "gamma", "Error")
results2 <- data.table(resultsSVM)
results2 <- results2[with(results2, order(Error)), ]

# save results in .csv file
write.csv(results2,"SVM_testing.csv", row.names=F)

stopCluster(cl)
