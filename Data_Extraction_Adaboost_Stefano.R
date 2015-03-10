library(adabag)

setwd("~/Dropbox/MSc Data Science/Courses/006 - Machine Learning (14D005)/Competition/ThyNearestNeighbours")
D <- read.table("Kaggle_Covertype_training.csv", head = TRUE, sep=",")

for(i in 2:11){
  D[,i] <- as.double(scale(as.numeric(D[,i]), center = TRUE, scale = TRUE))
}
data.scaled <- D

data <- D[,-1]
data$Cover_Type = factor(data$Cover_Type)
train <- data[1:47000,]
test <- data[47001:50000,]

# Trying with adabag package - BOOSTING

adafit.boo <- boosting(Cover_Type ~ ., data = train)
adapred.boo <- predict.boosting(adafit.boo, newdata = test)
adaerror.boo <- mean(adapred.boo$class != test[,ncol(test)])
adaerror.boo

# Trying with adabag package - BOOSTING vith CV (10-fold)

adafit.boo.cv <- boosting(Cover_Type ~ ., data = train)
adapred.boo.cv <- predict.boosting(adafit.boo.cv, newdata = test)
adaerror.boo.cv <- mean(adapred.boo.cv$class != test[,ncol(test)])
adaerror.boo.cv

# Trying with adabag package - BAGGING

adafit.bag <- bagging(Cover_Type ~ ., data = train)
adapred.bag <- predict.bagging(adafit.bag, newdata = test)
adaerror.bag <- mean(adapred.bag$class != test[,ncol(test)])
adaerror.bag

# Trying with adabag package - BAGGING with CV (10-fold)

adafit.bag.cv <- bagging.cv(Cover_Type ~ ., data = data)
adafit.bag.cv$error




