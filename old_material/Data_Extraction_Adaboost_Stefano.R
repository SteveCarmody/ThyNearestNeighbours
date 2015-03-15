library(adabag)

setwd("~/Dropbox/MSc Data Science/Courses/006 - Machine Learning (14D005)/Competition/ThyNearestNeighbours")
D <- read.table("Kaggle_Covertype_training.csv", head = TRUE, sep=",")

data <- D[,-1] # Remove ID column
data$Cover_Type <- as.factor(data$Cover_Type)

# Trying with adabag package - BOOSTING vith CV (10-fold)

adafit.boo.cv <- boosting.cv(Cover_Type ~ ., data = data)
adafit.boo.cv$error

# Trying with adabag package - BAGGING

adafit.bag <- bagging(Cover_Type ~ ., data = train)
adapred.bag <- predict.bagging(adafit.bag, newdata = test)
adaerror.bag <- mean(adapred.bag$class != test[,ncol(test)])
adaerror.bag

# Trying with adabag package - BAGGING with CV (10-fold)

adafit.bag.cv <- bagging.cv(Cover_Type ~ ., data = data)
adafit.bag.cv$error




