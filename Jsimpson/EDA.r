library(AppliedPredictiveModeling)
library(caret)
library(factoextra)
library(MASS)
library(pamr)
library(pls)
library(pROC)
library(rms)
library(sparseLDA)
library(e1071)
library(nnet)
library(klaR)
library(dplyr)
library(ggplot2)
library(fastDummies)
library(glmnet)
library(plotROC)
library(corrplot)
library(GGally)
library(corrgram)

#### Read in Data
setwd('C:/Users/jeres/Documents/GitHub/DAEN-690-Capstone-Project/Data')
DemographicData <- read.csv('Demographic Jsimpson.csv', header = T, sep = ',')
BodyMeasureData <- read.csv('BodyMeasures Jsimpson.csv', header = T, sep = ',')
DiabetesData <- read.csv('Diabetes Jsimpson.csv', header = T, sep = ',')

colnames(DiabetesData)
head(DiabetesData)

### Exploratory Data Analysis


corrgram(DiabetesData, 
         order=TRUE, 
         lower.panel=panel.shade, 
         upper.panel=panel.pie, 
         text.panel=panel.txt)

corrgram(BodyMeasureData, 
         order=TRUE, 
         lower.panel=panel.shade, 
         upper.panel=panel.pie, 
         text.panel=panel.txt)

corrgram(DemographicData, 
         order=TRUE, 
         lower.panel=panel.shade, 
         upper.panel=panel.pie, 
         text.panel=panel.txt)

PlotData <- cor(DiabetesData)
corrplot.mixed(PlotData,
               lower.col = "black",
               number.cex = .7)


ggpairs(DiabetesData)

plot(DiabetesData, pch=20, cex=1.5, col="#69b3a2")

CorData <- DiabetesData
##[ , c(5, 7, 9, 11, 21, 22, 29)]
PlotData <- cor(CorData)
PlotData
class(PlotData)

corrplot.mixed(PlotData)

corrplot.mixed(PlotData, 
               lower.col = "black", 
               number.cex = .7, 
               main = 'Correlogram for Income, Home Owner, Interest Rate, Home Ownership, and Loan Amount', 
               mar=c(0,0,2,0))


# find the distribution of the gender by income
values = DemographicData %>%
  group_by(Gender) %>%
  summarise_at(vars(FAM.Income), funs(min(.,na.rm=TRUE),mean(., na.rm=TRUE),max(.,na.rm=TRUE)))

value_df = as.data.frame(values)

# Transaction Type Histogram
table(DemographicData$Marital.Status)
ggplot(data=DemographicData,aes(x=Marital.Status)) +
  geom_bar(stat="count", fill="#69b3a2", color="#e9ecef") + 
  ggtitle("Distribution")

# Transaction Type Histogram
table(DemographicData$Race)
ggplot(data=DemographicData,aes(x=race)) +
  geom_bar(stat="count", fill="#69b3a2", color="#e9ecef") + 
  ggtitle(" Distribution")

library(tidyverse)
library(hrbrthemes)

# plot
p <- DemographicData %>%
  ggplot( aes(x=TimeUS)) +
  geom_histogram( binwidth=1, fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  ggtitle("Bin size = 3") +
  theme_ipsum() +
  theme(
    plot.title = element_text(size=15)
  )
p

DemographicData [ , c(5, 7, 9, 11, 21, 22, 29)]
class(DemographicData)
#CorData <- DemographicData[ , c(5, 7, 10, 11, 13, 15, 21)]

CorData <- DemographicData [ , c(5, 7, 9, 11, 21, 22, 29)]
PlotData <- cor(CorData)
PlotData
class(PlotData)

corrplot.mixed(PlotData)

corrplot.mixed(PlotData, 
               lower.col = "black", 
               number.cex = .7, 
               main = 'Correlogram for Income, Home Owner, Interest Rate, Home Ownership, and Loan Amount', 
               mar=c(0,0,2,0))
