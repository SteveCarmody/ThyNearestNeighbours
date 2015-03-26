### CLASS PREDICTION USING RANDOM FOREST

# Import and prepare data

library(randomForest)

setwd("C:/Users/Monika/Documents/Master/MachineLearning/Competition")

training <- read.csv("Kaggle_Covertype_training.csv", header = TRUE, sep = ",")
test <- read.csv("Kaggle_Covertype_test.csv", header = TRUE, sep = ",")

# Perfrom feature construction in training set
training <- cbind(training, EVDtH_1 = training$elevation-training$ver_dist_hyd)
training <- cbind(training, EHDtH_2 = training$elevation-(training$hor_dist_hyd*0.2))
training <- training[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                  31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,58,56)]

# Perform feature construction in test set
test <- cbind(test, EVDtH_1 = test$elevation-test$ver_dist_hyd)
test <- cbind(test, EHDtH_2 = test$elevation-(test$hor_dist_hyd*0.2))

### Prediction of the classes

# creating the training set
xTrain <- training[,2:(ncol(training)-1)]
xTrain <- sapply(xTrain,as.numeric)
yTrain <- training[,58]
yTrain <- factor(yTrain)

# creating the test set
xTest <- test[,(2:(ncol(test)))]
xTest <- sapply(xTest,as.numeric)

##################################################

training_rf <- randomForest(x = xTrain, y = yTrain, xtest = NULL, ytest = NULL,
                            ntree = 1350,
                            mtry = 27,
                            replace = TRUE,
                            nodesize = 1,
                            keep.forest = TRUE)


predictedClasses <- predict(training_rf, xTest, type ="response")
             
# save results in .csv file
write.csv(predictedClasses,"rf_prediction.csv", row.names=F)
