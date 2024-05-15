data("cars")
View(cars)
# Simple Linear Regression

# Importing the dataset


# Splitting the dataset into the Training set and Test set
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(cars$speed, SplitRatio = 2/3)
training_set = subset(cars, split == TRUE)
test_set = subset(cars, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Simple Linear Regression to the Training set
regressor = lm(formula = dist ~ speed,
               data = training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

#to visualise the training subset using scatterplot 
plot(x=training_set$speed, y=training_set$dist,
     main = "Scatterplot of dist  vs. speed",
     xlab = "speed",
     ylab = "dist")


# Visualising the Training set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$speed, y = training_set$dist),
             colour = 'red') +
  geom_line(aes(x = training_set$speed, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('dist vs speed (Training set)') +
  xlab('speed') +
  ylab('dist')

# Visualising the Test set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$speed, 
                 y = test_set$dist),
             colour = 'red') +
  geom_line(aes(x = training_set$speed, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('dist vs speed (Test set)') +
  xlab('speed') +
  ylab('dist')  
