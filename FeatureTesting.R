setwd('/Users/StephenCarmody/Documents/Machine Learning Competition/')
rm( list=ls() )

if (!require("class")) install.packages("class")
library(class)

if (!require("foreach")) install.packages("foreach")
library(foreach)

if (!require("doSNOW")) install.packages("doSNOW")
library(doSNOW)

if (!require("ggplot2")) install.packages("ggplot2")
library(ggplot2)

if (!require("caret")) install.packages("caret")
library(caret)

if (!require("mlbench")) install.packages("mlbench")
library(mlbench)

library(data.table)

library(class)


noCores <- 2
cl <- makeCluster(noCores, type="SOCK", outfile="")
registerDoSNOW(cl)

type <- "train" # predict | train

# read in intial data
data.read <- read.csv("Kaggle_Covertype_training.csv", head = TRUE, sep = ',')

data.raw <- data.read


data.raw <- cbind(data.raw, EVDtH = data.raw$elevation-data.raw$ver_dist_hyd*.2)         # 57
data.raw <- cbind(data.raw, EHDtH = data.raw$elevation-(data.raw$hor_dist_hyd*0.2))   # 58
data.raw <- cbind(data.raw, NewF4 = data.raw$elevation-(data.raw$hor_dist_road*-0.15))  # 59

data.raw <- data.raw[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                        31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,58,59,56)]

# Reduce data used
data.raw <- head(data.raw, 500)



data.train.X <- data.raw[,2:(ncol(data.raw)-1)]
data.train.Y <- data.raw[,ncol(data.raw)]


# I just used a out of the box random forest
model <- train(data.train.Y~., data=data.train.X, method="cforest")

importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

# Should set number to 10 for highter CrossValidation Accuracy
control <- rfeControl(functions=rfFuncs, method="cv", number=3)

results <- rfe(data.train.X, data.train.Y , sizes=c(1:57), rfeControl=control)

stopCluster(cl)

# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))

#aspect2 <- vector()
#aspect2 <- ifelse(data.raw$aspect[] + 180, aspect2 <- data.raw$aspect[] - 180, aspect2 <- data.raw$aspect[] + 180 )


