#------------------------------------------------------#
#28/september/2023

#height predictor vector
x <- c(4.1, 5.5, 5.8, 6.1, 6.4, 6.7, 6.4, 6.1, 5.10, 2.7)

#weight response vector
y <- c(63, 66, 69,72, 75, 78, 75, 72, 69, 66)

relation <- lm(y~x)

summary(relation)

#find weight of a person with given height

a <- data.frame(x = 6.3)

result <- predict(relation, a)
print(result)
#-------------------------------------------------------#
# 29/september/2023

# simple linear regression
getwd()
#importing the dataset
dataset <- read.csv('Salary_data.csv')
View(dataset)

#Splitting the dataset into the train set and test set
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

#Feature Scaling
#training_set <- scale(training_set)
#test_set <- scale(test_set)

#Fiiting simple Linear regression to the training set
regressor <- lm(formula = Salary ~ YearsExperience, data = training_set)

#Predicting the test set results
y_pred = predict(regressor, newdata = test_set)

# to Visualize the training subset using scatterplot
plot(x= training_set$YearsExperience, y= training_set$Salary, 
     main = "Scatterpot of Years of Experience VS Salary",
     xlab = "Years of Experience",
     ylab = "Salary")

#Visulising the training set results
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary VS Experience (training_set)' ) +
  xlab('Years of Experience') +
  ylab('Salary')


#05/october/2023 - thursday 02-04pm

#apply simple regression on given dataset
dataset <- read.csv('Position_Salaries.csv')
View(dataset)

model <- lm(Salary ~ Level, data = dataset)

summary(model)


#applying polynomial regression
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4

poly_reg <- lm(formula = Salary ~ ., data = dataset)
View(dataset)

#Visulising the simple regression results
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(model, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth VS Bluff (Linear regression)' ) +
  xlab('Level') +
  ylab('Salary')

#visualising the poynomial regression results
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth VS Bluff (Linear regression)' ) +
  xlab('Level') +
  ylab('Salary')
#-----------------------------------------------------------#
#predicting a new result with Linear Regression
predict(model, data.frame(Level = 6.5))

#Predicting a new result with polynomial Regression
predict(poly_reg, data.frame(Level = 6.5,
                             Level2 = 6.5^2,
                             Level3 = 6.5^3,
                             Level4 = 6.5^4))
#---------------------------------------------------------#
#Applying simple linear regression on cars inbuilt dataset

dataset <- cars
View(cars)

model <- lm(formula = dist ~ speed, data = dataset)
summary(model)

#Visulising the simple regression results
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$speed, y = dataset$dist),
             colour = 'red') +
  geom_line(aes(x = dataset$speed, y = predict(model, newdata = dataset)),
            colour = 'blue') +
  ggtitle('cars (Linear regression)' ) +
  xlab('speed') +
  ylab('level')