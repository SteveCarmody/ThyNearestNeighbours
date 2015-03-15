### MACHINE LEARNING COMPETITION ###
###
### TEAM: Thy Nearest Neighbours (Monika, Stefano, Stephen) ###
###

# This file contains the code that runs our preferred option for each of the
# techniques we have considered for this exercise. For each technique, we report
# the cross-validation or test error we have obtained. We use this result to motivate
# the choice of the technique underpinning our final submission. The technical report
# accompanying our submission provides details on our testing results and the decision
# process we have followed.

# The Github repository also contains the code for implementing the analysis for the 
# techniques we have shortlisted (one file each)

### 1 - PREPARATION ###

## Load the required libraries

library(adabag)
library(randomForest)
library(class)
# [ADD ALL THE LIBRARIES WE'VE USED]

## Load the training data
D <- read.table("Kaggle_Covertype_training.csv", head = TRUE, sep=",")
training <- D[,-1] # Remove ID column
training$Cover_Type <- as.factor(training$Cover_Type)

### 2 - LEARNING ###

## K-NN

# {STEPHEN TO POPULATE}

## SVM

# {STEFANO TO POPULATE}

## Boosting and bagging

# {STEFANO TO POPULATE}

## Random Forest

# {MONIKA TO POPULATE}

### 3 - RESULTS ###

# Here we summarise the results of our analysis and motivate the choice
# underpinning our final submission

### 4 - PREDICTION ###

# Here we make the prediction which we chose as our final submission to Kaggle

## Load the test data
D <- read.table("Kaggle_Covertype_test.csv", head = TRUE, sep=",")
test <- D[,-1] # Remove ID column

## Use preferred model for prediction
labels <- predict( # insert preferred model here )
ids <- seq(50001,150000,1)
submission <- cbind(ids,labels)
colnames(submission) <- c("id","Cover_Type")

## Save the results as CSV file in the required format
write.csv(submission,"ThyNearestNeighbour-submission.csv",row.names=F)


