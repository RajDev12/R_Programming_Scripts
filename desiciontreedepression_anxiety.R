data <- read.csv('depression_anxiety_data.csv')
View(data)

str(data)
summary(data)

sum(is.na(data))
data <- na.omit(data)
sum(is.na(data))

head(data,3)

library(caTools)
set.seed(123)
split_data <- sample.split(data$depression_severity, SplitRatio = 0.8)
training_data <- subset(data, split_data == TRUE)
training_data
testing_data <- subset(data, split_data == FALSE)
testing_data

library(rpart)
model_tree <- rpart(depression_severity ~ school_year + age + bmi + phq_score, data = training_data)
library(rpart.plot)
rpart.plot(model_tree)

prediction <- predict(model_tree, newdata = testing_data, type = "class")
prediction
library(caret)
confusionmatrix <- table(Actual = testing_data$depression_severity, prediction)
confusionmatrix
acc <- sum(diag(confusionmatrix)) / sum(confusionmatrix)
acc

library(e1071)