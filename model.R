library(caret)
#### practice data fit

#bootstrap fitcontrol
fitControl <- trainControl(method = "boot",
                           #number = 10,
                           allowParallel = TRUE
)

#function to calculate the f1 score
f1 <- function(fit, t){
    precision <- posPredValue(fit, t$target, positive = "1")
    
    recall <- sensitivity(fit, t$target, positive = "1")
    
    F1 <- (2*precision*recall)/(precision+recall)
    F1
}


### functions for convenience
#LDA function
#x = training dataset
#t = testing dataset

get_lda <- function(x){
    set.seed(20051)
    inTrain <- createDataPartition(x$id, p = 0.75, list = FALSE)
    training <- x[inTrain,]
    testing <- x[-inTrain,]
    
    set.seed(21500)
    model_lda <- train(target ~.,
                       data = training[,-1],
                       trControl = fitControl,
                       #preProcess = c("BoxCox"),
                       method = "lda"
    )
    model_lda
}

get_rf <- function(x){
    set.seed(20051)
    inTrain <- createDataPartition(x$id, p = 0.75, list = FALSE)
    training <- x[inTrain,]
    testing <- x[-inTrain,]
    
    set.seed(21500)
    model_rf <- train(target ~.,
                      data = training[,-1],
                      trControl = fitControl,
                      method = "rf"
    )
    model_rf
}

get_glm <- function(x){
    set.seed(20051)
    inTrain <- createDataPartition(x$id, p = 0.75, list = FALSE)
    training <- x[inTrain,]
    testing <- x[-inTrain,]
    
    set.seed(21500)
    model_glm <- train(target ~.,
                       data = training[,-1],
                       trControl = fitControl,
                       method = "glm"
    )
    model_glm
}

get_xgbTree <- function(x){
    set.seed(20051)
    inTrain <- createDataPartition(x$id, p = 0.75, list = FALSE)
    training <- x[inTrain,]
    testing <- x[-inTrain,]
    
    set.seed(21500)
    model_xgb <- train(target ~.,
                       data = training[,-1],
                       trControl = fitControl,
                       method = "xgbTree"
    )
    model_xgb
}

get_gbm <- function(x){
    set.seed(20051)
    inTrain <- createDataPartition(x$id, p = 0.75, list = FALSE)
    training <- x[inTrain,]
    testing <- x[-inTrain,]
    
    set.seed(21500)
    model_gbm <- train(target ~.,
                       data = training[,-1],
                       trControl = fitControl,
                       method = "gbm"
    )
    model_gbm
}

get_bag <- function(x){
    set.seed(20051)
    inTrain <- createDataPartition(x$id, p = 0.75, list = FALSE)
    training <- x[inTrain,]
    testing <- x[-inTrain,]
    
    set.seed(21500)
    model_bag <- train(target ~.,
                       data = training[,-1],
                       trControl = fitControl,
                       method = "treebag"
    )
    model_bag
}


get_lda_f1 <- function(x){
    set.seed(20051)
    inTrain <- createDataPartition(x$id, p = 0.75, list = FALSE)
    training <- x[inTrain,]
    testing <- x[-inTrain,]
    
    set.seed(21500)
    model_lda <- train(target ~.,
                       data = training[,-1],
                       trControl = fitControl,
                       #preProcess = c("BoxCox"),
                       method = "lda"
    )
    pred.lda <- predict(model_lda, newdata = testing)
    f1(pred.lda, testing)
    
}

get_rf_f1 <- function(x){
    set.seed(20051)
    inTrain <- createDataPartition(x$id, p = 0.75, list = FALSE)
    training <- x[inTrain,]
    testing <- x[-inTrain,]
    
    set.seed(21500)
    model_rf <- train(target ~.,
                       data = training[,-1],
                       trControl = fitControl,
                       method = "rf"
    )
    pred.rf <- predict(model_rf, newdata = testing)
    f1(pred.rf, testing)
}

get_glm_f1 <- function(x){
    set.seed(20051)
    inTrain <- createDataPartition(x$id, p = 0.75, list = FALSE)
    training <- x[inTrain,]
    testing <- x[-inTrain,]
    
    set.seed(21500)
    model_glm <- train(target ~.,
                       data = training[,-1],
                       trControl = fitControl,
                       method = "glm"
    )
    pred.glm <- predict(model_glm, newdata = testing)
    f1(pred.glm, testing)
}

get_xgbTree_f1 <- function(x){
    set.seed(20051)
    inTrain <- createDataPartition(x$id, p = 0.75, list = FALSE)
    training <- x[inTrain,]
    testing <- x[-inTrain,]
    
    set.seed(21500)
    model_xgb <- train(target ~.,
                       data = training[,-1],
                       trControl = fitControl,
                       method = "xgbTree"
    )
    pred.xgb <- predict(model_xgb, newdata = testing)
    f1(pred.xgb, testing)
}

get_gbm_f1 <- function(x){
    set.seed(20051)
    inTrain <- createDataPartition(x$id, p = 0.75, list = FALSE)
    training <- x[inTrain,]
    testing <- x[-inTrain,]
    
    set.seed(21500)
    model_gbm <- train(target ~.,
                       data = training[,-1],
                       trControl = fitControl,
                       method = "gbm"
    )
    pred.gbm <- predict(model_gbm, newdata = testing)
    f1(pred.gbm, testing)
}

get_bag_f1 <- function(x){
    set.seed(20051)
    inTrain <- createDataPartition(x$id, p = 0.75, list = FALSE)
    training <- x[inTrain,]
    testing <- x[-inTrain,]
    
    set.seed(21500)
    model_bag <- train(target ~.,
                       data = training[,-1],
                       trControl = fitControl,
                       method = "treebag"
    )
    pred.bag <- predict(model_bag, newdata = testing)
    f1(pred.bag, testing)
}

