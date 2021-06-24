library(caret)
library(ggplot2)
library(randomForest)
library(dplyr)

setwd('C:/Users/prahi/Desktop/DAEN 690 - Capstone/Final Project - GitHub Repo/DAEN-690-Capstone-Project/Data')

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

set.seed(0)
index_AA <- createDataPartition(response_only_AA, p=0.1, list=FALSE)
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

responseTraining_all <- factor(responseTraining_all)
responseTesting_all <- factor(responseTesting_all)
levels(responseTraining_all) <- c('No_Risk', 'Risk')
levels(responseTesting_all) <- c('No_Risk', 'Risk')

#params
RFGrid <- expand.grid(.mtry = 3:6)
RFparams <- trainControl(method = 'cv', number = 10, classProbs = TRUE, savePredictions = TRUE) 

set.seed(0)
RFmodel <- train(predictorTraining_all,responseTraining_all,method="rf",
                 trControl = RFparams,
                 tuneGrid = RFGrid)

RFmodel
RFmodel$bestTune
RFmodel$results[3,] #these are the optimal model params
RFmerge <- merge(RFmodel$pred,  RFmodel$bestTune)

RFTest <- data.frame(Method="RF",Y=responseTesting_all,
                     X=predict(RFmodel,predictorTesting_all))

#RF Predict
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
