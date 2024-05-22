#Importing all the libraries needs for the classification models
library(caret)
library(tidyverse)
library(rpart.plot)
library(rsample)
library(MASS)
library(dplyr)


#Loading the Diabetes dataset 
diab_data<-read.csv("C:/Users/duapr/OneDrive/Documents/Conestoga College Study/Predictive Analytics/Assignments records/Multivariate/Project 2/Diabetes_Classification/diabetes_classification.csv")
str(diab_data)

#Splitting training set and testing set after the initial_split above
sampleData2<-initial_split(diab_data,prop = 0.8,strata = Age)
trainset <- training(sampleData2)
testset <- testing(sampleData2)

#Converting Outcome category of 1 as "Yes", Outcome category of 0 as "No"
trainset$Outcome <- factor(trainset$Outcome, labels = c("No","Yes"))
testset$Outcome <- factor(testset$Outcome, labels = c("No","Yes"))


##################   KNN CLASSIFICATION MODEL ##################


#Setting grid values for k
k_grid <- expand.grid(k = seq(1,5))

#Training KNN with the training data
KNN_model<- train(
  Outcome ~ .,
  data = trainset,
  method = "knn",
  preProcess = c("center","scale"),
  tuneGrid = k_grid,
  trControl = trainControl(method = "cv", number = 10)
)

#Calling the trained model to show optimal model was selected during training
KNN_model

#Plotting KNN model to compare accuracy amongst the 15 models generated
plot(KNN_model)

#Running Predictions for KNN model
KNN_prediction<- predict(KNN_model, testset)
KNN_prediction

#Displaying Prediction Summary for KNN 
confusionMatrix(KNN_prediction,testset$Outcome)


######### Decision tree Classification Model ##################

#Training Decision Tree Model with the training data
tree_diabetes <- train(Outcome~.,
                        data = trainset,
                        method = "rpart",
                        trControl = trainControl(method = "cv", number = 10)
)

tree_diabetes

#Plotting the final chosen model for Decision Tree
rpart.plot(tree_diabetes$finalModel)

#Running Predictions for Decision Tree Model
tree_predictions <- predict(tree_diabetes, newdata = testset)

#Displaying Prediction Summary for Decision Tree 
confusionMatrix(tree_predictions, testset$Outcome)


######### Logistic Classification Model #####################

#Training Logistic Regression Model with the training data
Logistic_Diab<-train(Outcome~.,
                     data = trainset,
                     method = "glm",
                     family = "binomial",
                     trControl = trainControl(method = "cv", number = 10)
)

Logistic_Diab
summary(Logistic_Diab)
#Running Predictions for Logistic model
log_predictions <- predict(Logistic_Diab,testset)
log_predictions

#Displaying Prediction Summary for Logisitic Regression 
confusionMatrix(log_predictions, testset$Outcome)

######### Linear Discriminant Analysis Classification Model #####################

# Separate predictors (features) and outcome variable in the training data
X_train <- subset(trainset, select = -Outcome)
Y_train <- trainset$Outcome

# Fit LDA model
lda_model <- lda(Outcome ~ ., data = trainset)

# Summary of the LDA model
lda_model

plot(lda_model)

#Make Predictions on Test Data

# Separate predictors (features) from the test data
X_test <- subset(testset, select = -Outcome)

# Predict using the LDA model
lda_predictions <- predict(lda_model, newdata = X_test)
lda_predictions

# View the predicted classes
lda_predicted_classes <- lda_predictions$class

# Print confusion matrix
confusionMatrix(predicted_classes, testset$Outcome)


######### Random Forest Classification Model #####################

#Training Random Forest with the training data
model_random <- train(Outcome ~ .,
                      data = trainset,
                      method = "ranger",
                      trControl = trainControl(method="cv", number = 5, verboseIter = TRUE, classProbs = TRUE),
                      num.trees = 100,
                      importance = "impurity"
)

#Calling the trained model to show optimal model was selected during training
model_random

#Plotting the model for Random Forest
plot(model_random)

#Running Predictions for Random Forest Model
random_model_prediction<-predict(model_random, newdata = testset)
random_model_prediction


#Displaying Prediction Summary for Random forest 
confusionMatrix(random_model_prediction,testset$Outcome)
