#Load libraries
library(caret)
library(ggplot2)
library(randomForest)
library(dplyr)

#Set Working directory and import data
setwd('C:\Users\johnn\OneDrive\Documents\GitHub\DAEN-690-Capstone-Project\Data')

data_nhanes <- read.csv('NHANES_Data.csv', header = T, sep = ',')

#Select needed columns and rename
colnames(data_nhanes)
nhanes_trim <- subset(data_nhanes, select=-c(ï..Measured_Diabetes_A1c, Measured_Diabetes, Dr_Diabetes_Binary, Dr_Diabetes, Pre_Dia, SEQN, AHH_Income,
                               FAM_Income, Weight_kg, Height_cm, Len_Leg, Len_Arm, Waist, Fasting_Glucose, Private_Insurance))
colnames(nhanes_trim)

zeroVarpred <- nearZeroVar(nhanes_trim)
colnames(nhanes_trim[,zeroVarpred])

names <- c('Measured_Diabetes_x2', 'Gender','Race','Birth_Country','Citizenship','Edu_Adult','Marital_Status','Health_Insurance')

#Define categorical features and levels
nhanes_trim[,names] <- lapply(nhanes_trim[,names], as.factor)

levels(nhanes_trim$Gender) <- c('Male','Female')
levels(nhanes_trim$Race) <- c('Mexican','Hispanic','White','Black','Asian','Other')
levels(nhanes_trim$Birth_Country) <- c('USA','Other')
levels(nhanes_trim$Citizenship) <- c('Citizen','Non_Citizen')
levels(nhanes_trim$Edu_Adult) <- c('Below9th','Between9-11','HS_Grad','Some_College','College_Grad')
levels(nhanes_trim$Marital_Status) <- c('Married','Widowed','Divorced','Separated','Never_Married','Living_with_partner')
levels(nhanes_trim$Health_Insurance) <- c('Yes','No')

#Omit NA values
rm_nas <- na.omit(nhanes_trim)

#Separate African American and Non-African American datasets
non_AA <- subset(nhanes_trim, Race!='Black')
table(non_AA$Race)

only_AA <- subset(nhanes_trim, Race == 'Black')
table(only_AA$Race)

#Separate target variable
response_non_AA <- non_AA[,'Measured_Diabetes_x2']
# response_non_AA <- factor(response_non_AA)
# levels(response_non_AA) <- c('No Risk', 'Risk')

response_only_AA <- only_AA[,'Measured_Diabetes_x2']
# response_only_AA <- factor(response_only_AA)
# levels(response_only_AA) <- c('No Risk', 'Risk')

non_AA <- non_AA[,-1]
only_AA <- only_AA[,-1]

#Partition data to exaggerate bias and define training and testing data
set.seed(0)
index_AA <- createDataPartition(response_only_AA, p=0.2, list=FALSE)
predictorTraining_only_AA <- only_AA[index_AA,]
predictorTesting_only_AA <- only_AA[-index_AA,]
responseTraining_only_AA <- response_only_AA[index_AA]
responseTesting_only_AA <- response_only_AA[-index_AA]

set.seed(0)
index_non_AA <- createDataPartition(response_non_AA, p=0.8, list=FALSE)
predictorTraining_non_AA <- non_AA[index_non_AA,]
predictorTesting_non_AA <- non_AA[-index_non_AA,]
responseTraining_non_AA <- response_non_AA[index_non_AA]
responseTesting_non_AA <- response_non_AA[-index_non_AA]

#Append African American and Non-African American datasets (Training and testing datasets)
predictorTraining_all <- rbind(predictorTraining_only_AA, predictorTraining_non_AA)
nrow(predictorTraining_only_AA)
nrow(predictorTraining_non_AA)
nrow(predictorTraining_all)

predictorTesting_all <- rbind(predictorTesting_only_AA, predictorTesting_non_AA)
nrow(predictorTesting_only_AA)
nrow(predictorTesting_non_AA)
nrow(predictorTesting_all)

responseTraining_all <- c(responseTraining_only_AA, responseTraining_non_AA)
length(responseTraining_only_AA)
length(responseTraining_non_AA)
length(responseTraining_all)

responseTesting_all <- c(responseTesting_only_AA, responseTesting_non_AA)
length(responseTesting_only_AA)
length(responseTesting_non_AA)
length(responseTesting_all)

#Define categorical features and levels
responseTraining_all <- factor(responseTraining_all)
responseTesting_all <- factor(responseTesting_all)
levels(responseTraining_all) <- c('No_Risk', 'Risk')
levels(responseTesting_all) <- c('No_Risk', 'Risk')

#Random Forest Model Parameters
RFGrid <- expand.grid(.mtry = 3:6)
RFparams <- trainControl(method = 'cv', number = 10, classProbs = TRUE, savePredictions = TRUE) 

#Train Random Forest Model
set.seed(0)
RFmodel <- train(predictorTraining_all,responseTraining_all,method="rf",
                 trControl = RFparams,
                 tuneGrid = RFGrid)

RFmodel
RFmodel$bestTune
RFmodel$results[3,] #these are the optimal model parameters
RFmerge <- merge(RFmodel$pred,  RFmodel$bestTune)

RFTest <- data.frame(Method="RF",Y=responseTesting_all,
                     X=predict(RFmodel,predictorTesting_all))

#Random Forest Predictions
RFPredictions <- predict(RFmodel, newdata=predictorTesting_all)
RFAssess <- data.frame(obs=responseTesting_all, pred = RFPredictions)
defaultSummary(RFAssess)
confusionMatrix(RFPredictions, reference = responseTesting_all)

varImp(RFmodel)

test <- predictorTesting_all
test['predictions'] <- RFPredictions
test['actual_vals'] <- responseTesting_all

test <- test %>% mutate(Results = if_else(predictions ==actual_vals, 1, 0))

table(test$Race, test$Results)

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
