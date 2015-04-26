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
data.raw <- read.csv("Kaggle_Covertype_training.csv", head = TRUE, sep = ',')
data.raw.test<- matrix()

if(type == "predict")
{
  data.raw.test <- read.table("Kaggle_Covertype_test.csv", head = TRUE, sep = ',')
}

# Plot all non binary variables against each other

# for(i in 2:11)
# {
#   for(j in 2:11)
#   {
#     myplot <- ggplot(data.raw, aes(x = data.raw[,i], y = data.raw[,j])) + geom_point(aes(colour = factor(data.raw$Cover_Type)))    
#     ggsave(myplot,filename=paste("myplot",i,j,".png",sep=""))
#   }
# }


data.raw <- read.csv("Kaggle_Covertype_training.csv", head = TRUE, sep = ',')
data.raw <- head(data.raw, 5000)


data.raw <- cbind(data.raw, EVDtH = data.raw$elevation-data.raw$ver_dist_hyd)         # 57
data.raw <- cbind(data.raw, EHDtH = data.raw$elevation-(data.raw$hor_dist_hyd*0.2))   # 58
data.raw <- cbind(data.raw, NewF4 = data.raw$elevation-(data.raw$hor_dist_road*0.3))  # 59

data.raw <- data.raw[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                        31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,58,56)]





if(type == "predict")
{
  data.raw.test <- cbind(data.raw.test, EVDtH = data.raw$elevation-data.raw$ver_dist_hyd)
  data.raw.test <- cbind(data.raw.test, EHDtH = data.raw$elevation-(data.raw$hor_dist_hyd*0.2))
}


scaleData <- function(data){
  
  data.scaled <- data.frame()
  
  for(i in 2:11){
    data.raw[,i] <- as.double(scale(as.numeric(data.raw[,i]), center = TRUE, scale = TRUE))
  }
  data.scaled <- data.raw
  
  return(data.scaled)
}

numFolds <- 10
kOpt <- c(1)
foldList <- 1:(numFolds*length(kOpt))
kList <- rep(kOpt, length(1:numFolds))


resultsKnn<- foreach(i = foldList, k = kList,
                     .combine=rbind, .packages=c("class", "dplyr")) %dopar% {
                       
                       cat("Sample ", i, " is the current test set! k = ", k, fill = TRUE)
                       
                       # random choice of row numbers for monte carlo cross validation
                       sampleIndex <- sample( (nrow(data.raw)*.90) , (nrow(data.raw)*.10) )
                       
                       if(type == "train"){
                         data.train <- data.raw[-sampleIndex,]
                         data.test <- data.raw[sampleIndex,]
                         
                         data.train.X <- data.train[,2:(ncol(data.train)-1)]
                         data.train.Y <- data.train[,ncol(data.train)]
                         
                         data.test.X <- data.test[,2:(ncol(data.test)-1)]
                         data.test.Y <- data.test[,ncol(data.test)] 
                         
                       }else{
                         if(type == "predict"){
                           
                           data.test <- (data.raw.test)
                           data.train <- data.raw
                           
                           data.train.X <- data.train[,2:(ncol(data.train)-1)]
                           data.train.Y <- data.train[,ncol(data.train)]
                           
                           data.test.X <- data.test[,2:(ncol(data.test))]
                         }
                       }
                       
                       # kNN results
                       
                       testPredictions <- knn(train=data.train.X, test=data.test.X, cl=data.train.Y, k=k)
                       
                       if(type == "train")
                       {
                         testError <- mean(testPredictions != data.test.Y)
                       }
                       
                       # return results
                       result <- c(i, k, testError)
                     }

# puts results in a data table
results <- data.frame(resultsKnn)
results <- aggregate(results$X3 ~ results$X2, results, mean)
colnames(results) <- c("K", "Error")
results

ggplot(results) +
geom_line(aes(x = results$K, y =results$Error)) +
xlab("K Nearest Neighbuors") +
ylab("Error") +
ggtitle("Relation of K and Error Estimate")

# writes scaled training data and error results to csv files
write.csv(results ,"classification_error_estimate_KNN.csv",row.names=F)

if(type == "predict")
{
  id <- seq(from = 50001, to = 150000)
  Cover_Type <- testPredictions
  testPredictions.FC <- cbind(id, Cover_Type)
  write.csv(testPredictions.FC ,"knnPredictions_FC.csv",row.names = FALSE)
}

stopCluster(cl)


scaleData <- function(data){
  
  data.scaled <- data.frame()
  
  for(i in 2:11){
    data.raw[,i] <- as.double(scale(as.numeric(data.raw[,i]), center = TRUE, scale = TRUE))
  }
  data.scaled <- data.raw
  
  return(data.scaled)
}

data.raw <- scaleData(data.raw)

# Principle Component Analysis
data.PCA <- data.raw[,2:(ncol(data.raw)-1)]
data.PCA <- prcomp(data.PCA, center = TRUE, scale = FALSE)
plot(data.PCA, type = "b")
plot(cumsum(res$sdev^2/sum(res$sdev^2)))

summary(data.PCA)


setwd('/Users/StephenCarmody/Documents/Machine Learning Competition/')
rm( list=ls() )

noCores <- 2
cl <- makeCluster(noCores, type="SOCK", outfile="")
registerDoSNOW(cl)

type <- "train" # predict | train

# read in intial data
data.raw <- read.csv("Kaggle_Covertype_training.csv", head = TRUE, sep = ',')

data.raw <- cbind(data.raw, EVDtH = data.raw$elevation-data.raw$ver_dist_hyd)         # 57
data.raw <- cbind(data.raw, EHDtH = data.raw$elevation-(data.raw$hor_dist_hyd*0.2))   # 58
data.raw <- cbind(data.raw, NewF4 = data.raw$elevation-(data.raw$hor_dist_road*0.3))  # 59

data.raw <- data.raw[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                        31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,58,56)]

data.raw <- head(data.raw, 5000)



data.train.X <- data.raw[,2:(ncol(data.raw)-1)]
data.train.Y <- data.raw[,ncol(data.raw)]

model <- train(data.train.Y~., data=data.train.X, method="cforest")
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

control <- rfeControl(functions=rfFuncs, method="cv", number=3)

results <- rfe(data.train.X, data.train.Y , sizes=c(1:57), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))

write.csv((results$results) ,"Best_Feature_Selection_Results.csv",row.names=F)
write.csv((predictors(results)) ,"Best_Feature_Selection_Predictors.csv",row.names=F)









