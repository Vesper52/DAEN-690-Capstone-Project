library(caret)
library(ggplot2)
library(randomForest)
library(dplyr)
library(fastDummies)

setwd("~/GitHub/DAEN-690-Capstone-Project/Data")
setwd("C:/Users/prahi/Desktop/DAEN 690 - Capstone/Final Project - GitHub Repo/DAEN-690-Capstone-Project/Data")

#### Data Load and Subset ####
#setwd('C:/Users/jeres/Documents/GitHub/DAEN-690-Capstone-Project/Data')
data_nhanes <- read.csv('NHANES_Data.csv', header = T, sep = ',')

nhanes_trim <- subset(data_nhanes, select=-c(�..Measured_Diabetes_A1c, Measured_Diabetes, Dr_Diabetes_Binary, Dr_Diabetes, Pre_Dia, AHH_Income,
                                              FAM_Income, Weight_kg, Height_cm, Len_Leg, Len_Arm, Waist, Fasting_Glucose, Private_Insurance, SEQN))

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
response_only_AA <- only_AA[,'Measured_Diabetes_x2']
non_AA <- non_AA[,-1]
only_AA <- only_AA[,-1]
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

# 30%  African American
set.seed(0)
index_AA30 <- createDataPartition(response_only_AA, p=0.3, list=FALSE)
predictorTraining_only_AA30 <- only_AA[index_AA30,]
predictorTesting_only_AA30 <- only_AA[-index_AA30,]
responseTraining_only_AA30 <- response_only_AA[index_AA30]
responseTesting_only_AA30 <- response_only_AA[-index_AA30]

# 30% Predictor Training
predictorTraining_all30 <- rbind(predictorTraining_only_AA30, predictorTraining_non_AA)
nrow(predictorTraining_only_AA30)
nrow(predictorTraining_non_AA)
nrow(predictorTraining_all30)

# 30% Predictor Testing
predictorTesting_all30 <- rbind(predictorTesting_only_AA30, predictorTesting_non_AA)
nrow(predictorTesting_only_AA30)
nrow(predictorTesting_non_AA)
nrow(predictorTesting_all30)

# 30% Response Training
responseTraining_all30 <- c(responseTraining_only_AA30, responseTraining_non_AA)
length(responseTraining_only_AA30)
length(responseTraining_non_AA)
length(responseTraining_all30)

# 30% Response Testing
responseTesting_all30 <- c(responseTesting_only_AA30, responseTesting_non_AA)
length(responseTesting_only_AA30)
length(responseTesting_non_AA)
length(responseTesting_all30)

responseTraining_all30 <- factor(responseTraining_all30)
responseTesting_all30 <- factor(responseTesting_all30)
levels(responseTraining_all30) <- c('No_Risk', 'Risk')
levels(responseTesting_all30) <- c('No_Risk', 'Risk')

# 50%  African American
set.seed(0)
index_AA50 <- createDataPartition(response_only_AA, p=0.5, list=FALSE)
predictorTraining_only_AA50 <- only_AA[index_AA50,]
predictorTesting_only_AA50 <- only_AA[-index_AA50,]
responseTraining_only_AA50 <- response_only_AA[index_AA50]
responseTesting_only_AA50 <- response_only_AA[-index_AA50]

# 50% Predictor Training
predictorTraining_all50 <- rbind(predictorTraining_only_AA50, predictorTraining_non_AA)
nrow(predictorTraining_only_AA50)
nrow(predictorTraining_non_AA)
nrow(predictorTraining_all50)

# 50% Predictor Testing
predictorTesting_all50 <- rbind(predictorTesting_only_AA50, predictorTesting_non_AA)
nrow(predictorTesting_only_AA50)
nrow(predictorTesting_non_AA)
nrow(predictorTesting_all50)

# 50% Response Training
responseTraining_all50 <- c(responseTraining_only_AA50, responseTraining_non_AA)
length(responseTraining_only_AA50)
length(responseTraining_non_AA)
length(responseTraining_all50)

# 50% Response Testing
responseTesting_all50 <- c(responseTesting_only_AA50, responseTesting_non_AA)
length(responseTesting_only_AA50)
length(responseTesting_non_AA)
length(responseTesting_all50)

responseTraining_all50 <- factor(responseTraining_all50)
responseTesting_all50 <- factor(responseTesting_all50)
levels(responseTraining_all50) <- c('No_Risk', 'Risk')
levels(responseTesting_all50) <- c('No_Risk', 'Risk')

####model without subsampling AAs#####
response_all <- nhanes_trim[,'Measured_Diabetes_x2']
pred_all <-nhanes_trim[,-1]

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

##############################################################################################################################
### GLMnet Modeling ###

##create dummy cols for 5% AA
dummyPredictorsTrain5 <- dummy_cols(predictorTraining_all5, select_columns=c('Gender', 'Race', 'Birth_Country', 'Citizenship', 'Edu_Adult','Marital_Status',
                                                                             'HH_Numb','Health_Insurance'))

dummyPredictorsTrain5 <- dummyPredictorsTrain5[,-c(1,4,5,6,7,8,12)]

dummyPredictorsTest5 <- dummy_cols(predictorTesting_all5, select_columns=c('Gender', 'Race', 'Birth_Country', 'Citizenship', 'Edu_Adult','Marital_Status',
                                                                           'HH_Numb','Health_Insurance'))
dummyPredictorsTest5 <- dummyPredictorsTest5[,-c(1,4,5,6,7,8,12)]

# 5% AA
set.seed(0)
glmnetModel <- train(dummyPredictorsTrain5[,-2],responseTraining_all5,method="glmnet",
                     trControl = RFparams, preProcess = c('center', 'scale'))

glmnetModel
glmnetModel$bestTune
glmnetModel$results[7,] #these are the optimal model params
glmnetMerge <- merge(glmnetModel$pred,  glmnetModel$bestTune)

#glmnetTest <- data.frame(Method="RF",Y=responseTesting_all5,
# X=predict(RFmodel,predictorTesting_all5))

#5% GLMnet Predict
glmnetPredictions <- predict(glmnetModel, newdata=dummyPredictorsTest5[,-2])
glmnetAssess <- data.frame(obs=responseTesting_all5, pred = glmnetPredictions)
defaultSummary(glmnetAssess)
confusionMatrix(glmnetPredictions, reference = responseTesting_all5, positive='Risk')

test <- dummyPredictorsTest5
test['Race'] <- dummyPredictorsTest5[,2]
test['predictions'] <- glmnetPredictions
test['actual_vals'] <- responseTesting_all5

test <- test %>% mutate(Results = if_else(predictions ==actual_vals, 1, 0))

table(test$Race, test$Results)

## END ##

#30% GLMnet predictions
dummyPredictorsTrain30 <- dummy_cols(predictorTraining_all30, select_columns=c('Gender', 'Race', 'Birth_Country', 'Citizenship', 'Edu_Adult','Marital_Status',
                                                                               'HH_Numb','Health_Insurance'))

dummyPredictorsTrain30 <- dummyPredictorsTrain30[,-c(1,4,5,6,7,8,12)]

dummyPredictorsTest30 <- dummy_cols(predictorTesting_all30, select_columns=c('Gender', 'Race', 'Birth_Country', 'Citizenship', 'Edu_Adult','Marital_Status',
                                                                             'HH_Numb','Health_Insurance'))
dummyPredictorsTest30 <- dummyPredictorsTest30[,-c(1,4,5,6,7,8,12)]

# 30% AA
set.seed(0)
glmnetModel <- train(dummyPredictorsTrain30[,-2],responseTraining_all30,method="glmnet",
                     trControl = RFparams, preProcess = c('center', 'scale'))

glmnetModel
glmnetModel$bestTune
glmnetModel$results[7,] #these are the optimal model params
glmnetMerge <- merge(glmnetModel$pred,  glmnetModel$bestTune)

#30% GLMnet Predict
glmnetPredictions <- predict(glmnetModel, newdata=dummyPredictorsTest30[,-2])
glmnetAssess <- data.frame(obs=responseTesting_all30, pred = glmnetPredictions)
defaultSummary(glmnetAssess)
confusionMatrix(glmnetPredictions, reference = responseTesting_all30, positive='Risk')

test <- dummyPredictorsTest30
test['Race'] <- dummyPredictorsTest30[,2]
test['predictions'] <- glmnetPredictions
test['actual_vals'] <- responseTesting_all30

test <- test %>% mutate(Results = if_else(predictions ==actual_vals, 1, 0))

table(test$Race, test$Results)

## END ##


#50% GLMnet predictions
dummyPredictorsTrain50 <- dummy_cols(predictorTraining_all50, select_columns=c('Gender', 'Race', 'Birth_Country', 'Citizenship', 'Edu_Adult','Marital_Status',
                                                                               'HH_Numb','Health_Insurance'))

dummyPredictorsTrain50 <- dummyPredictorsTrain50[,-c(1,4,5,6,7,8,12)]

dummyPredictorsTest50 <- dummy_cols(predictorTesting_all50, select_columns=c('Gender', 'Race', 'Birth_Country', 'Citizenship', 'Edu_Adult','Marital_Status',
                                                                             'HH_Numb','Health_Insurance'))
dummyPredictorsTest50 <- dummyPredictorsTest50[,-c(1,4,5,6,7,8,12)]

# 50% AA
set.seed(0)
glmnetModel <- train(dummyPredictorsTrain50[,-2],responseTraining_all50,method="glmnet",
                     trControl = RFparams, preProcess = c('center', 'scale'))

glmnetModel
glmnetModel$bestTune
glmnetModel$results[7,] #these are the optimal model params
glmnetMerge <- merge(glmnetModel$pred,  glmnetModel$bestTune)

#glmnetTest <- data.frame(Method="RF",Y=responseTesting_all5,
# X=predict(RFmodel,predictorTesting_all5))

#50% GLMnet Predict
glmnetPredictions <- predict(glmnetModel, newdata=dummyPredictorsTest50[,-2])
glmnetAssess <- data.frame(obs=responseTesting_all50, pred = glmnetPredictions)
defaultSummary(glmnetAssess)
confusionMatrix(glmnetPredictions, reference = responseTesting_all50, positive='Risk')

test <- dummyPredictorsTest50
test['Race'] <- dummyPredictorsTest50[,2]
test['predictions'] <- glmnetPredictions
test['actual_vals'] <- responseTesting_all50

test <- test %>% mutate(Results = if_else(predictions ==actual_vals, 1, 0))

table(test$Race, test$Results)

## END ##

#GLMNET No Subsample
dummyPredictorsTrainall <- dummy_cols(pred_train_all, select_columns=c('Gender', 'Race', 'Birth_Country', 'Citizenship', 'Edu_Adult','Marital_Status',
                                                                       'HH_Numb','Health_Insurance'))

dummyPredictorsTrainall <- dummyPredictorsTrainall[,-c(1,4,5,6,7,8,12)]

dummyPredictorsTestall <- dummy_cols(pred_test_all, select_columns=c('Gender', 'Race', 'Birth_Country', 'Citizenship', 'Edu_Adult','Marital_Status',
                                                                     'HH_Numb','Health_Insurance'))
dummyPredictorsTestall <- dummyPredictorsTestall[,-c(1,4,5,6,7,8,12)]

# No Subsample
set.seed(0)
glmnetModel <- train(dummyPredictorsTrainall[,-2],resp_train_all,method="glmnet",
                     trControl = RFparams, preProcess = c('center', 'scale'))

glmnetModel
glmnetModel$bestTune
glmnetModel$results[7,] #these are the optimal model params
glmnetMerge <- merge(glmnetModel$pred,  glmnetModel$bestTune)

#No Subsample GLMnet predict
glmnetPredictions <- predict(glmnetModel, newdata=dummyPredictorsTestall[,-2])
glmnetAssess <- data.frame(obs=resp_test_all, pred = glmnetPredictions)
defaultSummary(glmnetAssess)
confusionMatrix(glmnetPredictions, reference = resp_test_all, positive='Risk')

test <- dummyPredictorsTestall
test['Race'] <- dummyPredictorsTestall[,2]
test['predictions'] <- glmnetPredictions
test['actual_vals'] <- resp_test_all

test <- test %>% mutate(Results = if_else(predictions ==actual_vals, 1, 0))

table(test$Race, test$Results)

### Syn Data
synData50 <- read.csv('New_Synthetic_Data_50.csv', header = T, sep = ',')

synData50[,names] <- lapply(synData50[,names], as.factor)

levels(synData50$Gender) <- c('Male','Female')
levels(synData50$Race) <- c('Mexican','Hispanic','White','Black','Asian','Other')
levels(synData50$Birth_Country) <- c('USA','Other')
levels(synData50$Citizenship) <- c('Citizen','Non_Citizen')
levels(synData50$Edu_Adult) <- c('Below9th','Between9-11','HS_Grad','Some_College','College_Grad')
levels(synData50$Marital_Status) <- c('Married','Widowed','Divorced','Separated','Never_Married','Living_with_partner')
levels(synData50$Health_Insurance) <- c('Yes','No')

synResponse50 <- synData50[,'Measured_Diabetes_x2']
synResponse50 <- factor(synResponse50)
levels(synResponse50) <- c('No_Risk', 'Risk')

syn_real_response_50 <- c(responseTraining_all30, synResponse50)
length(responseTraining_all30)
length(synResponse50)
length(syn_real_response_50)

syn_real_response_50 <- factor(syn_real_response_50)
levels(syn_real_response_50) <- c('No_Risk','Risk')

synTrain50 <- synData50[,-13]
syn_real_train_50 <- rbind(predictorTraining_all30, synTrain50)
nrow(predictorTraining_all30)
nrow(synTrain50)
nrow(syn_real_train_50)

dummyPredictorsTrain30_syn50 <- dummy_cols(syn_real_train_50, select_columns=c('Gender', 'Race', 'Birth_Country', 'Citizenship', 'Edu_Adult','Marital_Status',
                                                                               'HH_Numb','Health_Insurance'))

dummyPredictorsTrain30_syn50 <- dummyPredictorsTrain30_syn50[,-c(1,4,5,6,7,8,12)]

# 30% AA
set.seed(0)
glmnetModel_syn <- train(dummyPredictorsTrain30_syn50[,-2],syn_real_response_50,method="glmnet",
                     trControl = RFparams, preProcess = c('center', 'scale'))

glmnetModel_syn
glmnetModel_syn$bestTune
glmnetModel_syn$results[7,] #these are the optimal model params
glmnetMerge <- merge(glmnetModel_syn$pred,  glmnetModel_syn$bestTune)

#GLMnet Predict Syn Data
glmnetPredictions <- predict(glmnetModel_syn, newdata=dummyPredictorsTest30[,-2])
glmnetAssess <- data.frame(obs=responseTesting_all30, pred = glmnetPredictions)
defaultSummary(glmnetAssess)
confusionMatrix(glmnetPredictions, reference = responseTesting_all30, positive='Risk')

test <- dummyPredictorsTest30
test['Race'] <- dummyPredictorsTest30[,2]
test['predictions'] <- glmnetPredictions
test['actual_vals'] <- responseTesting_all30

test <- test %>% mutate(Results = if_else(predictions ==actual_vals, 1, 0))

table(test$Race, test$Results)

### Syn 70 Data

synData70 <- read.csv('New_Synthetic_Data_90.csv', header = T, sep = ',') #change CSV file name here based on which synthetic data file is needed

synData70[,names] <- lapply(synData70[,names], as.factor)

levels(synData70$Gender) <- c('Male','Female')
levels(synData70$Race) <- c('Mexican','Hispanic','White','Black','Asian','Other')
levels(synData70$Birth_Country) <- c('USA','Other')
levels(synData70$Citizenship) <- c('Citizen','Non_Citizen')
levels(synData70$Edu_Adult) <- c('Below9th','Between9-11','HS_Grad','Some_College','College_Grad')
levels(synData70$Marital_Status) <- c('Married','Widowed','Divorced','Separated','Never_Married','Living_with_partner')
levels(synData70$Health_Insurance) <- c('Yes','No')

synResponse70 <- synData70[,'Measured_Diabetes_x2']
synResponse70 <- factor(synResponse70)
levels(synResponse70) <- c('No_Risk', 'Risk')

syn_real_response_70 <- c(responseTraining_all30, synResponse70)
length(responseTraining_all30)
length(synResponse70)
length(syn_real_response_70)

syn_real_response_70 <- factor(syn_real_response_70)
levels(syn_real_response_70) <- c('No_Risk','Risk')

synTrain70 <- synData70[,-13]

syn_real_train_70 <- rbind(predictorTraining_all30, synTrain70)
nrow(predictorTraining_all30)
nrow(synTrain70)
nrow(syn_real_train_70)

dummyPredictorsTrain30_syn70 <- dummy_cols(syn_real_train_70, select_columns=c('Gender', 'Race', 'Birth_Country', 'Citizenship', 'Edu_Adult','Marital_Status',
                                                                               'HH_Numb','Health_Insurance'))

dummyPredictorsTrain30_syn70 <- dummyPredictorsTrain30_syn70[,-c(1,4,5,6,7,8,12)]

# 30% AA + Syn
set.seed(0)
glmnetModel_syn <- train(dummyPredictorsTrain30_syn70[,-2],syn_real_response_70,method="glmnet",
                         trControl = RFparams, preProcess = c('center', 'scale'))

glmnetModel_syn
glmnetModel_syn$bestTune
glmnetModel_syn$results[1,] #these are the optimal model params
glmnetMerge <- merge(glmnetModel_syn$pred,  glmnetModel_syn$bestTune)

#GLMnet Predict Syn Data
glmnetPredictions <- predict(glmnetModel_syn, newdata=dummyPredictorsTest30[,-2])
glmnetAssess <- data.frame(obs=responseTesting_all30, pred = glmnetPredictions)
defaultSummary(glmnetAssess)
confusionMatrix(glmnetPredictions, reference = responseTesting_all30, positive='Risk')

test <- dummyPredictorsTest30
test['Race'] <- dummyPredictorsTest30[,2]
test['predictions'] <- glmnetPredictions
test['actual_vals'] <- responseTesting_all30

test <- test %>% mutate(Results = if_else(predictions ==actual_vals, 1, 0))

table(test$Race, test$Results)
