#Christopher Han
#Created on: 10/19/2019
#ConocoPhillips Challenge for TAMU Datathon 2019
#Last run: 10/20/2019
#Challenge Question: Predict surface and down-hole failures using the data set provided

#load useful packages
library(dplyr)

#Load training set
setwd("C:/Users/Chris Han/Datathon")
data <- read.csv("C:/Users/Chris Han/Datathon/equipfails/equip_failures_training_set.csv",
                 na.strings = "na")
summary(data)
sum(data$target) #1000/60000 = about 1.67% failure rate
data$target <- as.factor(data$target) #turn target column into factor

#remove features with near zero variance
nzv <- nearZeroVar(data[,3:172])
data <- data[,-(nzv + 2)]

#remove features that are highly correlated (r = 0.9 or more)
sensor_cor <- cor(data[,-c(1,2)], method = "spearman", use = "complete.obs")
highly_cor <- findCorrelation(sensor_cor, cutoff = 0.9)
df <- data[,-(highly_cor + 2)]

#remove features that have more than 5% missing values 
missing <- sapply(df, function(x) sum(is.na(x))/nrow(df))
sort(missing, decreasing = TRUE) #43 and 2 have significant amount of NA, remove
df <- df[,missing <= 0.05] #try 0.05 and 0.1
summary(df)

#impute the rest using median
for(i in 3:ncol(df)){
    df[is.na(df[,i]), i] <- median(df[,i], na.rm = TRUE)
}

#create balanced dataset by randomly sampling 1000 surface failures
#combined with 1000 down-hole failures data available to us
#very crude way, needs improvement
set.seed(2512)
random_1000 <- sample(df[df$target == 0,1], 1000)
df_adjusted <- rbind(df[random_1000,], df[df$target == 1,])

##import testing set and clean it the same way as the training data
testing_set <- read.csv("C:/Users/Chris Han/Datathon/equipfails/equip_failures_test_set.csv",
                        na.strings = "na")
#remove all the columns that doesn't exist in training data
testing_set <- testing_set[,names(df)[-2]]

#impute missing values using median
for(i in 2:ncol(testing_set)){
    testing_set[is.na(testing_set[,i]), i] <- median(testing_set[,i], na.rm = TRUE)
}

#clean up environment
rm(sensor_cor, highly_cor, i, missing, nzv, random_1000)

