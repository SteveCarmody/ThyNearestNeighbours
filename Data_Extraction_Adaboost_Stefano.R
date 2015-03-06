setwd("~/Dropbox/MSc Data Science/Courses/006 - Machine Learning (14D005)/Competition/ThyNearestNeighbours")
rm( list=ls() )

if (!require("class")) install.packages("class")
library(class)

if (!require("foreach")) install.packages("foreach")
library(foreach)

if (!require("doSNOW")) install.packages("doSNOW")
library(doSNOW)

# read in intial data
D <- read.table("Kaggle_Covertype_training.csv", head = TRUE, stringsAsFactors=FALSE)

# grab column names
colNames <- names(D)

# convert column names from one joined string to seperate column names
colNames <- strsplit(colNames, '[.]')
colNames <- unlist(colNames)

# create empty data frame
data.raw <- data.frame()

# each row is a long string. Here we split up each variable and put it its appropriate column
data.raw <- t(data.frame(strsplit(D[,], ','), stringsAsFactors = FALSE))
rownames(data.raw) <- NULL
data.raw <- data.frame(data.raw, stringsAsFactors = FALSE)
names(data.raw) <- colNames

dataCorrectMat <- data.matrix(data.raw)

# Scales data for non boolean values
data.scaled <- data.frame()

for(i in 2:11){
  data.raw[,i] <- as.double(scale(as.numeric(data.raw[,i]), center = TRUE, scale = TRUE))
}
data.scaled <- data.raw

# Limit amount of data used to save compute time
data <- head(data.scaled, 5000)

numFolds <- 10
kOpt <- c(1, 3, 5, 15, 45, 60, 90)
foldList <- 1:(numFolds*length(kOpt))
kList <- rep(kOpt, length(1:numFolds))

system.time(
  resultsKnn<- foreach(i = foldList, k = kList,
                      .combine=rbind, .packages=c("class", "dplyr")) %dopar% {
                        
                        # some helpful debugging messages
                        cat("Sample ", i, " is the current test set! k = ", k, fill = TRUE)
                        
                        # random choice of row numbers for monte carlo cross validation
                        sampleIndex <- sample(1:nrow(data), size = (nrow(data)*.9), replace = FALSE, prob = NULL)
                        
                        # select training data
                        Xtrain <- data[sampleIndex,(2:ncol(data)-1)] 
                        Ytrain <- data[sampleIndex,ncol(data)] 
                        
                        # select testing data
                        Xtest <- data[-sampleIndex,(2:ncol(data)-1)]
                        Ytest <- data[-sampleIndex,ncol(data)] 
                        
                        # kNN results
                        testPredictions <- knn(train=Xtrain, test=Xtest, cl=Ytrain, k=k)
                        testError <- mean(testPredictions != Ytest)
                        
                        # last thing is returned
                        result <- c(i, k, testError)
                      }
)

# puts results in a data table
results <- data.table(resultsKnn)
results
melt(results)

# writes scaled training data and error results to csv files
write.table(data.scaled ,"training_Data_scaled.csv", sep = ",")
write.table(results ,"classification_error_estimate.csv", sep = ",")


