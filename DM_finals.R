# install and load libraries
#install.packages(c("readr", "tidyr", "dplyr", "plyr", "caret", "tidyverse", "caTools", "rpart", "rattle", "rpart.plot", "RColorBrewer", "GGally", "ggplot2"))
#remotes::install_github("wilkelab/cowplot")
library(readr)
library(tidyr)
library(dplyr)
library(plyr)
library(caret)
library(tidyverse)
library(caTools)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(GGally)
library(ggplot2)
library(cowplot)

# Importing the dataset
setwd('C:/Users/user/Downloads/ITS61504 Data Mining')
dataset = read.csv("heart_failure_clinical_records_dataset V1.03.csv")
head(dataset)
glimpse(dataset)

unique(dataset$age) # identify inconsistencies in data type
count(duplicated(dataset) == TRUE) # identify duplicated records
dataset[!complete.cases(dataset),] # identify the records with missing values
summary(dataset) # identify impossible or extreme values, and count the missing values

# boxplots of the numerical attribute Age before its data pre-processing
boxplot(dataset$age, xlab = "Age", ylab = "Number of Years", main = "Boxplot of Age")

# Pearson's product-moment correlation coefficients using correlation matrix
datasetCorCoef = dataset
datasetCorCoef$heart_attack = as.numeric(datasetCorCoef$heart_attack)
cor(datasetCorCoef)

# pair plots
ggpairs(datasetCorCoef, columns = 1:6, 
        ggplot2::aes(colour=as.factor(dataset$heart_attack), alpha=9))

# percentage stacked bar plots relative to 'heart_attack'
ggplot(dataset, aes(x = factor(sex), fill=factor(heart_attack))) + 
  geom_bar(position="fill") #bar plot
ggplot(dataset, aes(x = factor(anemia), fill=factor(heart_attack))) +
  geom_bar(position="fill")
ggplot(dataset, aes(x = factor(diabetes), fill=factor(heart_attack))) +
  geom_bar(position="fill")
ggplot(dataset, aes(x = factor(high_blood_pressure), fill=factor(heart_attack))) +
  geom_bar(position="fill")
ggplot(dataset, aes(x = factor(smoking), fill=factor(heart_attack))) +
  geom_bar(position="fill")

# data cleaning
# round up inconsistent 'age' values
dataset$age = round(dataset$age)
unique(dataset$age)

# fill in missing values with mode
val <- unique(dataset$diabetes[!(is.na(dataset$diabetes) == is.na(NA))])
modeDiabetes <- val[which.max(tabulate(match(dataset$diabetes, val)))]
print(modeDiabetes)
val <- unique(dataset$high_blood_pressure[!(is.na(dataset$high_blood_pressure) == is.na(NA))])
modeHBP <- val[which.max(tabulate(match(dataset$high_blood_pressure, val)))]
print(modeHBP)
val <- unique(dataset$smoking[!(is.na(dataset$smoking) == is.na(NA))])
modeSmoking <- val[which.max(tabulate(match(dataset$smoking, val)))]
print(modeSmoking)
dataset$diabetes <- replace(dataset$diabetes, is.na(dataset$diabetes) == is.na(NA), 
                            modeDiabetes)
dataset$high_blood_pressure <- replace(dataset$high_blood_pressure, 
                                       is.na(dataset$high_blood_pressure) == is.na(NA), 
                                       modeHBP)
dataset$smoking <- replace(dataset$smoking, is.na(dataset$smoking) == is.na(NA), 
                           modeSmoking)
count(is.na(dataset) == TRUE) #identify missing values

# data transformation : discretise 'age' into two discrete levels
dataset$age <- ifelse(dataset$age < 70, 0, 1)
summary(dataset)

# model development
# Splitting the dataset into the Training set and Test set
set.seed(111) #list of random numbers starting from position 111
split = sample.split(dataset$heart_attack, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE) #75% of dataset used as training subset
test_set = subset(dataset, split == FALSE) #25% of dataset used as training subset

# Fitting Decision Tree Classification model through information gain to the Training set
# Specify model formula (heart_attack is identified as the class) & model data
classifier <- rpart(formula = heart_attack ~., data = training_set, method="class",
                    control = rpart.control(minsplit = 4, minbucket = 2, 
                                            maxdepth = 7, xval=20, cp=0.01),
                    parms = list(split = "information")) # information gain
print(classifier)

# Plot tree
plot(classifier)

#add text inside plotting area
text(classifier)

# Beautify tree font size, and positioning and colour 
prp(classifier, cex = 0.55, faclen = 0, extra = 1, border.col = 'maroon') 

# classification rules
rpart.rules(classifier, roundint=FALSE, clip.facs=TRUE)

# model performance evaluation
# Predicting the Test set results, return prediction for each class
y_pred <- predict(classifier, newdata = test_set[-7], type = 'class')
head(y_pred)

# confusion Matrix to compare original (test set) against prediction (y_pred)
table(test_set[, 7], y_pred)
# for class 1:
# 53/75 = 70.6667% accuracy
# 4/24 = 16.6667% recall
# 4/6 = 66.6667% precision

# model predictions using decision tree classification
prediction <- data.frame(age = 0, sex = 1, anemia = 0, diabetes = 1, 
                         high_blood_pressure = 0, smoking = 1)
predict(classifier, prediction, type = "prob")
predict(classifier, prediction, type = "class")

# min max normalisation
prep = preProcess(as.data.frame(dataset$age), method = c("range"))
dataset$age = predict(prep, as.data.frame(dataset$age))
summary(dataset)

# model development
# Fitting Logistic Regression Machine Learning Model
glmLog <- glm(heart_attack ~ ., data = training_set, family = binomial)
summary(glmLog)

# feature selection using backward elimination based on significance level
glmLog <- glm(heart_attack ~ age + anemia, data = training_set, family = binomial)
summary(glmLog)

# model performance evaluation
# confusion Matrix to compare original (test set) against prediction (y_pred)
pred = predict(glmLog, newdata=test_set[-7], type = "response")
pred = ifelse(pred >= 0.5, 1, 0) # set 0.5 as boundary
cmLog = table(test_set[,7], pred)
confusionMatrix(cmLog)
# for class 1:
# 53/75 = 70.6667% accuracy
# 4/24 = 16.6667% recall
# 4/6 = 66.6667% precision