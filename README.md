# Predicting Downhole Equipment Failures

Christopher Han
10/20/2019

## Objective
When costs go up, cash goes down. The objective is to accurately predict the failures of downhole equipment in order to proactively identify possible problems and minimize costs associated with these failures.

### Background
Stripper wells are an important source of steady cash flow as they have low operational costs and low capital intensity. In fact, stripper wells account for 80% of producing oil wells in the United States due to this reason among others. In this challenge, we aim to train a model which can distinguish surface failures from downhole failures from a feature set of sensor data. Our main goal is to maximize the Mean F1-Score for predicting downhole failures.

## Data Preparation
The file “equip_failures_training_set” contains 60000 observations of 172 variables. “id” identifies the observations and “target” identifies whether it is a surface equipment failure (0) or downhole equipment failure (1). Initial exploration of the data reveals that only 1000 of 60000 observations, or about 1.67%, are downhole failures, resulting in a heavily imbalanced data. Imbalanced data decreases the performance of many machine learning models and we tackle this problem in later steps of data cleaning. First, we factorize the target column.

A major challenge with this dataset is the high dimensional feature set. For better performance in our machine learning algorithms as well as for interpretability, we try to extract only the useful features. We remove near zero variance features as well as features that are highly correlated with one another. Subsequently, we notice that some features have an extremely high proportion of missing (NA) values. We remove all features that contain more than 5% missing data. Resulting data set contains 59 variables of which 57 are sensor data (66.5% feature reduction).

There still remains a significant number of NA values across the data set. We impute the missing values with the median of each column. It is worth noting that we chose median instead of mean because nearly every feature is heavily left skewed.

As mentioned earlier, we need to resolve the issue of imbalanced results. An easy way is to sample 1000 random rows from the surface failure group and use all 1000 rows of the downhole failure group, resulting in 50:50 distribution of the result.
As last step in the data preparation, we clean the testing set in the same manner as the training set.

## Machine Learning
### Method
The training set containing 2000 rows are divided into 75% training and 25% testing. Each model is trained with the caret package with bootstrapping. Mean F1-Score is evaluated with a custom function for each algorithm.

We test the following six machine learning algorithms: Bagged Trees, Gradient Boosting Machine, Logistic Model, Linear Discriminant Analysis, Random Forest, XGB Tree. Each have their own benefits and disadvantages and we will compare the results. The algorithm for Linear Discriminant Analysis (LDA) is shown below. Each subsequent models were calculated and the scores were calculated in similar manner.

## Results
Each of the six algorithm produced a Mean F1-Score and a predicted target results for the testing set. The predicted target results were uploaded to Kaggle and the score was obtained. In the table below, we show the performance of each model on the training set as well as the testing set.

We can see that LDA performed the best in the testing set with a mean F1-Score of 0.97446 although other models performed better in the training set. Since LDA gives us the best results as well as better interpretability over some other models, we will use LDA in our final analysis.

## Conclusion
As the main objective was to identify the downhole equipment failures, we recommend the use of an LDA trained model with the above 57 sensors. The feature reduction of 66.5% will be helpful in identifying the specific sensor data that may contribute the most to the failure. With this data, we can proactively send out a crew to check on the 57 sensor readings to identify which part of the equipment is most prone to failure. In addition, correctly identifying whether it is a downhole equipment failure or surface equipment failure will save time and money in addresing the issues by providing the crew with accurate information on where to look and what equipment they require.

### Short-comings and Possible Improvements
A major short-coming of our model is the highly left skewed distributions of our feature set. Failure to address this issue possibly decreased the accuracy of our prediction. Box-Cox transformation or similar transformation of the sort to diminish the effect of non-normality may help improve the model in the future. Another short-coming of our process was in how we handled the imbalanced data. Randomly sampling 1000 values from the surface failures group is not an elegant solution and may have overestimated the accuracy of the model in the training process. In the future, we would need to address this issue through methods such as ROSE and SMOTE. Lastly, in the future, we should experiment with the tuning parameters for each algorithm to see if it improves the accuracy.
