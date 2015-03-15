library(e1071)

setwd("~/Dropbox/MSc Data Science/Courses/006 - Machine Learning (14D005)/Competition/ThyNearestNeighbours")
D <- read.table("Kaggle_Covertype_training.csv", head = TRUE, sep=",")

data <- D[,-1] # Remove ID column
data$Cover_Type <- as.factor(data$Cover_Type)
set.seed(1234)
subset <- sample(seq(1,45000,1),45000)
training <- data[subset,]
test <- data[-subset,]

svmfit <- svm(Cover_Type ~ ., data=training,
              type="C-classification", kernel = "radial", 
              cost = 0.1, gamma = 0.5, scale=FALSE) 

predictions <- predict(svmfit, test)
error <- mean(predictions != test[,ncol(test)])


