source("cleaning.R")
source("model.R")

f1_lda <- get_lda_f1(df_adjusted)
f1_bag <- get_bag_f1(df_adjusted)
f1_gbm <- get_gbm_f1(df_adjusted)
f1_glm <- get_glm_f1(df_adjusted)
f1_rf <- get_rf_f1(df_adjusted)
f1_xgb <- get_xgbTree_f1(df_adjusted)


### Export results
#lda
lda.pred <- predict(get_lda(df_adjusted), 
                     newdata = testing_set
)
lda.result <- data.frame(id = seq(1,16001), target = lda.pred)
write.csv(lda.result, "C:/Users/Chris Han/Datathon/lda_result.csv", row.names = FALSE)

#bag
bag.pred <- predict(get_bag(df_adjusted), 
                    newdata = testing_set
)
bag.result <- data.frame(id = seq(1,16001), target = bag.pred)
write.csv(bag.result, "C:/Users/Chris Han/Datathon/bag_result.csv", row.names = FALSE)

#gbm
gbm.pred <- predict(get_gbm(df_adjusted), 
                    newdata = testing_set
)
gbm.result <- data.frame(id = seq(1,16001), target = gbm.pred)
write.csv(gbm.result, "C:/Users/Chris Han/Datathon/gbm_result.csv", row.names = FALSE)

#glm
glm.pred <- predict(get_glm(df_adjusted), 
                    newdata = testing_set
)
glm.result <- data.frame(id = seq(1,16001), target = glm.pred)
write.csv(glm.result, "C:/Users/Chris Han/Datathon/glm_result.csv", row.names = FALSE)

#rf
rf.pred <- predict(get_rf(df_adjusted), 
                   newdata = testing_set
)
rf.result <- data.frame(id = seq(1,16001), target = rf.pred)
write.csv(rf.result, "C:/Users/Chris Han/Datathon/rf_result.csv", row.names = FALSE)

#xgbTree
xgb.pred <- predict(get_xgbTree(df_adjusted), 
                   newdata = testing_set
)
xgb.result <- data.frame(id = seq(1,16001), target = xgb.pred)
write.csv(xgb.result, "C:/Users/Chris Han/Datathon/xgb_result.csv", row.names = FALSE)

#table of results from kaggle
lda_perf <- 0.97446
bag_perf <- 0.93732
gbm_perf <- 0.94357
glm_perf <- 0.96044
rf_perf <- 0.93750
xgb_perf <- 0.94517

finalresults <- data.frame(c(f1_bag, f1_gbm, f1_glm, f1_lda, f1_rf, f1_xgb),
          c(bag_perf, gbm_perf, glm_perf, lda_perf, rf_perf, xgb_perf), 
          row.names = c("Bagged Tree", "GBM", "Logistic","LDA", "RF", "XGB"),
          )
names(finalresults) <- c("Training F1", "Testing F1")
write.csv(finalresults, "C:/Users/Chris Han/Datathon/finalresults.csv")

lda_final <- get_lda(df_adjusted)
lda_final
