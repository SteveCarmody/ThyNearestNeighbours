library(adabag)

setwd("~/Dropbox/MSc Data Science/Courses/006 - Machine Learning (14D005)/Competition/ThyNearestNeighbours")
D <- read.table("Kaggle_Covertype_training.csv", head = TRUE, sep=",")

for(i in 2:11){
  D[,i] <- as.double(scale(as.numeric(D[,i]), center = TRUE, scale = TRUE))
}
data.scaled <- D

data <- D[1:5000,-1]
data$Cover_Type = factor(data$Cover_Type)
train <- data[1:4500,]
test <- data[4501:5000,]

# Trying with vanilla adabag package

adafit <- boosting(Cover_Type ~ ., data = train)
adapred <- predict.boosting(adafit, newdata = test)
adaerror <- mean(adapred$class != test[,ncol(test)])
adaerror

# Trying with caret

library(caret)
library(doMC)
registerDoMC(cores = 3)
## All subsequent models are then run in parallel
cvCtrl <- trainControl(method = "repeatedcv", repeats = 1) # do 10-fold CV 1 time
adafit <- train(Cover_Type ~ ., data = train, method = "AdaBoost.M1",trControl = cvCtrl)
adapred <- predict.train(adafit, train)
adaerror.c <- mean(adapred$class != test[,ncol(test)])
adaerror.c


