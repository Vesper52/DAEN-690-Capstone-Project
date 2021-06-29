library(caret)
library(ggplot2)
library(randomForest)
library(dplyr)

#setwd('C:/Users/prahi/Desktop/DAEN 690 - Capstone/Final Project - GitHub Repo/DAEN-690-Capstone-Project/Data')

#### Data Load and Subset ####
setwd('C:/Users/jeres/Documents/GitHub/DAEN-690-Capstone-Project/Data')
data_nhanes <- read.csv('NHANES_Data.csv', header = T, sep = ',')
colnames(data_nhanes)

nhanes_trim <- subset(data_nhanes, select=-c(ï..Measured_Diabetes_A1c, Measured_Diabetes, Dr_Diabetes_Binary, Dr_Diabetes, Pre_Dia, SEQN, AHH_Income,
                                              FAM_Income, Weight_kg, Height_cm, Len_Leg, Len_Arm, Waist, Fasting_Glucose, Private_Insurance))
colnames(nhanes_trim)

zeroVarpred <- nearZeroVar(nhanes_trim)
colnames(nhanes_trim[,zeroVarpred])

names <- c('Measured_Diabetes_x2', 'Gender','Race','Birth_Country','Citizenship','Edu_Adult','Marital_Status','Health_Insurance')
nhanes_trim[,names] <- lapply(nhanes_trim[,names], as.factor)

levels(nhanes_trim$Gender) <- c('Male','Female')
levels(nhanes_trim$Race) <- c('Mexican','Hispanic','White','Black','Asian','Other')
levels(nhanes_trim$Birth_Country) <- c('USA','Other')
levels(nhanes_trim$Citizenship) <- c('Citizen','Non_Citizen')
levels(nhanes_trim$Edu_Adult) <- c('Below9th','Between9-11','HS_Grad','Some_College','College_Grad')
levels(nhanes_trim$Marital_Status) <- c('Married','Widowed','Divorced','Separated','Never_Married','Living_with_partner')
levels(nhanes_trim$Health_Insurance) <- c('Yes','No')

rm_nas <- na.omit(nhanes_trim)

non_AA <- subset(nhanes_trim, Race!='Black')
table(non_AA$Race)

only_AA <- subset(nhanes_trim, Race == 'Black')
table(only_AA$Race)

response_non_AA <- non_AA[,'Measured_Diabetes_x2']
# response_non_AA <- factor(response_non_AA)
# levels(response_non_AA) <- c('No Risk', 'Risk')

response_only_AA <- only_AA[,'Measured_Diabetes_x2']
# response_only_AA <- factor(response_only_AA)
# levels(response_only_AA) <- c('No Risk', 'Risk')

non_AA <- non_AA[,-1]
only_AA <- only_AA[,-1]
#### end ####


#### 5%  African American ####

# 5%  African American
set.seed(0)
index_AA5 <- createDataPartition(response_only_AA, p=0.05, list=FALSE)
predictorTraining_only_AA5 <- only_AA[index_AA5,]
predictorTesting_only_AA5 <- only_AA[-index_AA5,]
responseTraining_only_AA5 <- response_only_AA[index_AA5]
responseTesting_only_AA5 <- response_only_AA[-index_AA5]

# 5% Predictor Training
predictorTraining_all5 <- rbind(predictorTraining_only_AA5, predictorTraining_non_AA)
nrow(predictorTraining_only_AA5)
nrow(predictorTraining_non_AA)
nrow(predictorTraining_all5)

# 5% Predictor Testing
predictorTesting_all5 <- rbind(predictorTesting_only_AA5, predictorTesting_non_AA)
nrow(predictorTesting_only_AA5)
nrow(predictorTesting_non_AA)
nrow(predictorTesting_all5)

# 5% Response Training
responseTraining_all5 <- c(responseTraining_only_AA5, responseTraining_non_AA)
length(responseTraining_only_AA5)
length(responseTraining_non_AA)
length(responseTraining_all5)

# 5% Response Testing
responseTesting_all5 <- c(responseTesting_only_AA5, responseTesting_non_AA)
length(responseTesting_only_AA5)
length(responseTesting_non_AA)
length(responseTesting_all5)


responseTraining_all5 <- factor(responseTraining_all5)
responseTesting_all5 <- factor(responseTesting_all5)
levels(responseTraining_all5) <- c('No_Risk', 'Risk')
levels(responseTesting_all5) <- c('No_Risk', 'Risk')


# 5% Training Random Forest 
set.seed(0)
RFmodel <- train(predictorTraining_all5,responseTraining_all5,method="rf",
                 trControl = RFparams,
                 tuneGrid = RFGrid)

RFmodel
RFmodel$bestTune
RFmodel$results[3,] #these are the optimal model params
RFmerge <- merge(RFmodel$pred,  RFmodel$bestTune)

RFTest <- data.frame(Method="RF",Y=responseTesting_all5,
                     X=predict(RFmodel,predictorTesting_all5))

#5% Random Forest Predict
RFPredictions <- predict(RFmodel, newdata=predictorTesting_all5)
RFAssess <- data.frame(obs=responseTesting_all5, pred = RFPredictions)
defaultSummary(RFAssess)
confusionMatrix(RFPredictions, reference = responseTesting_all5)

varImp(RFmodel)

test <- predictorTesting_all5
test['predictions'] <- RFPredictions
test['actual_vals'] <- responseTesting_all5

test <- test %>% mutate(Results = if_else(predictions ==actual_vals, 1, 0))

table(test$Race, test$Results)

#### end ####


#### 10%  African American ####

# 10%  African American
set.seed(0)
index_AA10 <- createDataPartition(response_only_AA, p=0.1, list=FALSE)
predictorTraining_only_AA10 <- only_AA[index_AA10,]
predictorTesting_only_AA10 <- only_AA[-index_AA10,]
responseTraining_only_AA10 <- response_only_AA[index_AA10]
responseTesting_only_AA10 <- response_only_AA[-index_AA10]

# 10% Predictor Training
predictorTraining_all10 <- rbind(predictorTraining_only_AA10, predictorTraining_non_AA)
nrow(predictorTraining_only_AA10)
nrow(predictorTraining_non_AA)
nrow(predictorTraining_all10)

# 10% Predictor Testing
predictorTesting_all10 <- rbind(predictorTesting_only_AA10, predictorTesting_non_AA)
nrow(predictorTesting_only_AA10)
nrow(predictorTesting_non_AA)
nrow(predictorTesting_all10)

# 10% Response Training
responseTraining_all10 <- c(responseTraining_only_AA10, responseTraining_non_AA)
length(responseTraining_only_AA10)
length(responseTraining_non_AA)
length(responseTraining_all10)

# 10% Response Testing
responseTesting_all10 <- c(responseTesting_only_AA10, responseTesting_non_AA)
length(responseTesting_only_AA10)
length(responseTesting_non_AA)
length(responseTesting_all10)

responseTraining_all10 <- factor(responseTraining_all10)
responseTesting_all10 <- factor(responseTesting_all10)
levels(responseTraining_all10) <- c('No_Risk', 'Risk')
levels(responseTesting_all10) <- c('No_Risk', 'Risk')

# 10% Training Random Forest
set.seed(0)
RFmodel <- train(predictorTraining_all10,responseTraining_all10,method="rf",
                 trControl = RFparams,
                 tuneGrid = RFGrid)

RFmodel
RFmodel$bestTune
RFmodel$results[3,] #these are the optimal model params
RFmerge <- merge(RFmodel$pred,  RFmodel$bestTune)

RFTest <- data.frame(Method="RF",Y=responseTesting_all10,
                     X=predict(RFmodel,predictorTesting_all10))

# 10% Random Forest Predict
RFPredictions <- predict(RFmodel, newdata=predictorTesting_all10)
RFAssess <- data.frame(obs=responseTesting_all10, pred = RFPredictions)
defaultSummary(RFAssess)
confusionMatrix(RFPredictions, reference = responseTesting_all10)

varImp(RFmodel)

test <- predictorTesting_all10
test['predictions'] <- RFPredictions
test['actual_vals'] <- responseTesting_all10

test <- test %>% mutate(Results = if_else(predictions ==actual_vals, 1, 0))

table(test$Race, test$Results)
#### end ####


#### 15%  African American ####
# 15%  African American
set.seed(0)
index_AA15 <- createDataPartition(response_only_AA, p=0.15, list=FALSE)
predictorTraining_only_AA15 <- only_AA[index_AA15,]
predictorTesting_only_AA15 <- only_AA[-index_AA15,]
responseTraining_only_AA15 <- response_only_AA[index_AA15]
responseTesting_only_AA15 <- response_only_AA[-index_AA15]

# 15% Predictor Training
predictorTraining_all15 <- rbind(predictorTraining_only_AA15, predictorTraining_non_AA)
nrow(predictorTraining_only_AA15)
nrow(predictorTraining_non_AA)
nrow(predictorTraining_all15)

# 15% Predictor Testing
predictorTesting_all15 <- rbind(predictorTesting_only_AA15, predictorTesting_non_AA)
nrow(predictorTesting_only_AA15)
nrow(predictorTesting_non_AA)
nrow(predictorTesting_all15)

# 15% Response Training
responseTraining_all15 <- c(responseTraining_only_AA15, responseTraining_non_AA)
length(responseTraining_only_AA15)
length(responseTraining_non_AA)
length(responseTraining_all15)

# 15% Response Testing
responseTesting_all15 <- c(responseTesting_only_AA15, responseTesting_non_AA)
length(responseTesting_only_AA15)
length(responseTesting_non_AA)
length(responseTesting_all15)

responseTraining_all15 <- factor(responseTraining_all15)
responseTesting_all15 <- factor(responseTesting_all15)
levels(responseTraining_all15) <- c('No_Risk', 'Risk')
levels(responseTesting_all15) <- c('No_Risk', 'Risk')

# 15% Training Random Forest 
set.seed(0)
RFmodel <- train(predictorTraining_all15,responseTraining_all15,method="rf",
                 trControl = RFparams,
                 tuneGrid = RFGrid)

RFmodel
RFmodel$bestTune
RFmodel$results[3,] #these are the optimal model params
RFmerge <- merge(RFmodel$pred,  RFmodel$bestTune)

RFTest <- data.frame(Method="RF",Y=responseTesting_all15,
                     X=predict(RFmodel,predictorTesting_all15))

# 15% Random Forest Predict
RFPredictions <- predict(RFmodel, newdata=predictorTesting_all15)
RFAssess <- data.frame(obs=responseTesting_all15, pred = RFPredictions)
defaultSummary(RFAssess)
confusionMatrix(RFPredictions, reference = responseTesting_all15)

varImp(RFmodel)

test <- predictorTesting_all15
test['predictions'] <- RFPredictions
test['actual_vals'] <- responseTesting_all15

test <- test %>% mutate(Results = if_else(predictions ==actual_vals, 1, 0))

table(test$Race, test$Results)
#### end ####


#### 20%  African American ####
# 20%  African American
set.seed(0)
index_AA20 <- createDataPartition(response_only_AA, p=0.2, list=FALSE)
predictorTraining_only_AA20 <- only_AA[index_AA20,]
predictorTesting_only_AA20 <- only_AA[-index_AA20,]
responseTraining_only_AA20 <- response_only_AA[index_AA20]
responseTesting_only_AA20 <- response_only_AA[-index_AA20]

# 20% Predictor Training
predictorTraining_all20 <- rbind(predictorTraining_only_AA20, predictorTraining_non_AA)
nrow(predictorTraining_only_AA20)
nrow(predictorTraining_non_AA)
nrow(predictorTraining_all20)

# 20% Predictor Testing
predictorTesting_all20 <- rbind(predictorTesting_only_AA20, predictorTesting_non_AA)
nrow(predictorTesting_only_AA20)
nrow(predictorTesting_non_AA)
nrow(predictorTesting_all20)

# 20% Response Training
responseTraining_all20 <- c(responseTraining_only_AA20, responseTraining_non_AA)
length(responseTraining_only_AA20)
length(responseTraining_non_AA)
length(responseTraining_all20)

# 20% Response Testing
responseTesting_all20 <- c(responseTesting_only_AA20, responseTesting_non_AA)
length(responseTesting_only_AA20)
length(responseTesting_non_AA)
length(responseTesting_all20)

responseTraining_all20 <- factor(responseTraining_all20)
responseTesting_all20 <- factor(responseTesting_all20)
levels(responseTraining_all20) <- c('No_Risk', 'Risk')
levels(responseTesting_all20) <- c('No_Risk', 'Risk')

# 20% Training Random Forest 
set.seed(0)
RFmodel <- train(predictorTraining_all20,responseTraining_all20,method="rf",
                 trControl = RFparams,
                 tuneGrid = RFGrid)

RFmodel
RFmodel$bestTune
RFmodel$results[3,] #these are the optimal model params
RFmerge <- merge(RFmodel$pred,  RFmodel$bestTune)

RFTest <- data.frame(Method="RF",Y=responseTesting_all20,
                     X=predict(RFmodel,predictorTesting_all20))

#20% Random Forest Predict
RFPredictions <- predict(RFmodel, newdata=predictorTesting_all20)
RFAssess <- data.frame(obs=responseTesting_all20, pred = RFPredictions)
defaultSummary(RFAssess)
confusionMatrix(RFPredictions, reference = responseTesting_all20)

varImp(RFmodel)

test <- predictorTesting_all20
test['predictions'] <- RFPredictions
test['actual_vals'] <- responseTesting_all20

test <- test %>% mutate(Results = if_else(predictions ==actual_vals, 1, 0))

table(test$Race, test$Results)
#### end ####

# 70%
set.seed(0)
index_non_AA <- createDataPartition(response_non_AA, p=0.7, list=FALSE)
predictorTraining_non_AA <- non_AA[index_non_AA,]
predictorTesting_non_AA <- non_AA[-index_non_AA,]
responseTraining_non_AA <- response_non_AA[index_non_AA]
responseTesting_non_AA <- response_non_AA[-index_non_AA]



#params
RFGrid <- expand.grid(.mtry = 3:6)
RFparams <- trainControl(method = 'cv', number = 10, classProbs = TRUE, savePredictions = TRUE) 





####model without subsampling AAs#####
response_all <- rm_nas[,'Measured_Diabetes_x2']
pred_all <- rm_nas[,-1]

set.seed(0)
index_all <- createDataPartition(response_all, p=0.7, list=FALSE)
pred_train_all <- pred_all[index_all,]
pred_test_all <- pred_all[-index_all,]
resp_train_all <- response_all[index_all]
resp_test_all <- response_all[-index_all]

resp_train_all <- factor(resp_train_all)
resp_test_all <- factor(resp_test_all)
levels(resp_train_all) <- c('No_Risk', 'Risk')
levels(resp_test_all) <- c('No_Risk', 'Risk')

set.seed(0)
RFmodel_all <- train(pred_train_all,resp_train_all,method="rf",
                     trControl = RFparams,
                     tuneGrid = RFGrid)
RFmodel_all
RFmodel_all$bestTune
RFmodel_all$results[2,] #these are the optimal model params
RFmerge_all <- merge(RFmodel_all$pred,  RFmodel_all$bestTune)

RFPredictions_all <- predict(RFmodel_all, newdata=pred_test_all)
RFAssess_all <- data.frame(obs=resp_test_all, pred = RFPredictions_all)
defaultSummary(RFAssess_all)
confusionMatrix(RFPredictions_all, reference = resp_test_all)

test_all <- pred_test_all
test_all['predictions'] <- RFPredictions_all
test_all['actual_vals'] <- resp_test_all
test_all <- test_all %>% mutate(Results = if_else(predictions ==actual_vals, 1, 0))
table(test_all$Race, test_all$Results)
#### end ####