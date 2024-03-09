#11/08/2023 - friday

usedcars <- read.csv("usedcars.csv", stringsAsFactors = FALSE)
str(usedcars)
getwd()
summary(usedcars$year)
setwd('C:/Users/rjnat/OneDrive/Documents/R scripts')
summary(usedcars[c("price","mileage")])
c=c(5,10)
diff(c)
##diff func gives the difference between the 2 elements whihc are present in the vector
help(diff)
?sort
??knn
??caret
range(usedcars$price)
diff(range(usedcars$price))
diff(c)
table(usedcars$year)
sort(unique(usedcars$year))
table(usedcars$model)
table(usedcars$color)
model_table <- table(usedcars$model)
typeof(model_table)
class(model_table)

prop.table(model_table)
color_pct <- table(usedcars$color)
color_pct <- prop.table(color_pct) * 100
round(color_pct, digits = 1)

getwd()

17/Aug/2023 -- 02:04

getwd()

#data proceesing

#importing the dataset
dataset = read.csv("Data.csv")
#dataset=read.csv(file.choose())
View(dataset)
str(dataset)
summary(dataset)

#labelling the variables
names(dataset)

names(dataset) <- c("Country","Age","Salary")

str(dataset)
View(dataset)

#checking missing values
is.na(dataset)

sum(is.na(dataset))
View(dataset)
#to check how many na values each column is having
colSums(is.na(dataset))
#to check how many na values each row is having
rowSums(is.na(dataset))

dataset[1,]

missingdata <- dataset[!complete.cases(dataset),]

missingdata

#sum(is.na(missingdata))
View(dataset)

#taking care of missing values
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm=TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm=TRUE)),
                     dataset$Salary)
sum(is.na(dataset))
View(dataset)

str(dataset)

#handling the categorical values
dataset$Country= factor(dataset$Country,
                        levels= c("France","Spain","Germany"),
                        labels=c(1,2,3))

sum(is.na(dataset))

#18/08/2023

#preprocessing for sales data
s0 <- read.csv(file.choose(),header = TRUE, stringsAsFactors = TRUE)
str(s0)
summary(s0)

#Delete all the rows with missing data and name the new fataset as s1
s1 <- na.omit(s0)
View(s1)

summary(s1)
#Replace the missing values with mean vaue for each variable.
s0$Sales[is.na(s0$Sales)] <- mean(s0$Sales, na.rm = TRUE)
s0$Profit[is.na(s0$Profit)] <- mean(s0$Profit, na.rm = TRUE)
s0$Unit.Price[is.na(s0$Unit.Price)] <- mean(s0$Unit.Price, na.rm = TRUE)
summary(s0)



#########or other method###########

#Replacing the missing values for numerical variables

s0$Sales[is.na(s0$Sales)] <- runif(n = sum(is.na(s0$Sales)),
                                   min = min(s0$Sales, na.rm = TRUE),
                                   max = max(s0$Sales, na.rm = TRUE))

s0$Profit[is.na(s0$Profit)] <- runif(n = sum(is.na(s0$Profit)),
                                   min = min(s0$Profit, na.rm = TRUE),
                                   max = max(s0$Profit, na.rm = TRUE))
s0$Unit.Price[is.na(s0$Unit.Price)] <- runif(n = sum(is.na(s0$Unit.Price)),
                                   min = min(s0$Unit.Price, na.rm = TRUE),
                                   max = max(s0$Unit.Price, na.rm = TRUE))
summary(s0)

######------end-------#########

s0$Order.Priority[is.na(s0$Order.Priority)] <- sample(levels(s0$Order.Priority),
                                                      size = sum(is.na(s0$Order.Priority)),
                                                      replace = TRUE)

s0$Ship.Mode[is.na(s0$Ship.Mode)] <- sample(levels(s0$Ship.Mode),
                                                      size = sum(is.na(s0$Ship.Mode)),
                                                      replace = TRUE)

s0$Customer.Name[is.na(s0$Customer.Name)] <- sample(levels(s0$Customer.Name),
                                                      size = sum(is.na(s0$Customer.Name)),
                                                      replace = TRUE)

summary(s0)


# 24/August/2023

getwd()
wbcd <- read.csv("wisc_bc_data.csv", stringsAsFactors = FALSE)
str(wbcd)
wbcd <- wbcd[-1]
View(wbcd)
table(wbcd$diagnosis)
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"), labels = c("Benign", "Malignant"))
View(wbcd)
round(prop.table(table(wbcd$diagnosis)) * 100, digits = 1)

summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])

normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
} 
wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
View(wbcd_n)
summary(wbcd_n$area_mean)
wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]
wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

install.packages("class")
library(class)
wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, k= 21)

install.packages("gmodels")
library(gmodels)
CrossTable(x = wbcd_test_labels, y= wbcd_test_pred, prop.chisq = FALSE)
aa <- table(wbcd_test_labels, wbcd_test_pred)

#25/Aug/2023

getwd()
data("iris")
str(iris)
View(iris)
table(iris$Species)

round(prop.table(table(iris$Species)) * 100, digits = 1)
summary(iris)

normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

iris_norm <- as.data.frame(lapply(iris[1:4], normalize))
summary(iris_norm)
set.seed(1234)
ind <- sample(2, nrow(iris), replace = TRUE, prob = c(0.70, 0.30))
ind 
table(ind)
prop.table(table(ind))
round(prop.table(table(ind)) * 100, digits = 1)
iris_training <- iris[ind == 1, 1:4]
iris_training

iris_test <- iris[ind== 2, 1:4]
iris_test
iris.trainlabels <- iris[ind==1, 5]
iris.trainlabels
iris.testlables <- iris[ind== 2, 5]
iris.testlables

library(class)

k <- 3

predict_labels <- knn(iris_training, iris_test, iris.trainlabels, k)

library(gmodels)
CrossTable(x = iris.testlables, y = predict_labels, prop.chisq = FALSE)

tab <- table(predict_labels, iris.testlables)
tab


accuracy <- sum(predict_labels == iris.testlables) / length(iris.testlables) * 100

cat("Accuracy:", accuracy, "\n")

  
#31/aug and 01/sept

getwd()
sms_raw <- read.csv("sms_spam.csv", stringsAsFactors = FALSE)
str(sms_raw)
sms_raw$type <- factor(sms_raw$type)
str(sms_raw$type)
table(sms_raw$type)



install.packages("tm")
library(tm)
sms_corpus <- VCorpus(VectorSource(sms_raw$text))
print(sms_corpus)
inspect(sms_corpus[1])

as.character(sms_corpus[[1]])
lapply(sms_corpus[1:2], as.character)
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
as.character(sms_corpus[[1]])

as.character(sms_corpus_clean[[1]])

sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

install.packages("SnowballC")
library(SnowballC)
wordStem(c("learns", "learned", "learning", "learns"))
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

inspect(sms_dtm)
sms_dtm_train <- sms_dtm[1:4169, ]
inspect(sms_dtm_train)
sms_dtm_test <- sms_dtm[4170:5559, ]

sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4170:5559, ]$type

install.packages("wordcloud")
library(wordcloud)
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)
findFreqTerms(sms_dtm_train, 5)

sms_freq_words <- findFreqTerms(sms_dtm_train, 5)

str(sms_freq_words)
sms_dtm_freq_train <- sms_dtm_train[ ,  sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]
inspect(sms_dtm_freq_train)
convert_counts <- function(x) {
  x <- ifelse(x >0, "Yes", "No")
}
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

View(sms_train)
install.packages("e1071")
library(e1071)

sms_classifier <- naiveBayes(sms_train, sms_train_labels)

sms_test_pred <- predict(sms_classifier, sms_test)
head(sms_test_pred)
a <- table(sms_test_pred, sms_test_labels)
a
library(gmodels)
CrossTable(sms_test_pred, sms_test_labels, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted', 'actual'))

#8/September/2023

#Exploring iris data using naiveBayes algo#

install.packages("e1071")
data(iris)

install.packages("caTools")
library(caTools)
set.seed(1)
split = sample.split(iris, SplitRatio = 0.6)
train = subset(iris, split == TRUE)
test = subset(iris, split == FALSE)


model = naiveBayes(Species ~ ., data = train)

predictions = predict(model, test)

accuracy = sum(predictions == test$Species) / length(predictions)

accuracy

##14-15/September/2023

#Applying decision tree on iris dataset using rpart packages

data("iris")
str(iris)
View(iris)
install.packages("rpart")
library("rpart")

#this is used to pick 110 random number values from total of 150
indexes = sample(150,110)
indexes

iris_train = iris[indexes, ]
iris_train

iris_test = iris[-indexes, ]
iris_test

target = Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width 
target

tree = rpart(target, data = iris_train, method = "class")
tree

install.packages("rpart.plot")
library("rpart.plot")
rpart.plot(tree)
predictions = predict(tree, iris_test)
predictions

#---------------------------------------------------------------#
#applying decision tree on new dataset i.e. cardiotocographic

getwd()
cardio <- read.csv("cardiotocographic.csv", stringsAsFactors = FALSE)
str(cardio)
View(cardio)
install.packages("rpart")
library("rpart")

indexes = sample(2126,1500)
indexes

cardio_train = cardio[indexes, ]
cardio_train

cardio_test = cardio[-indexes, ]
cardio_test

target = NSP ~ LB + AC + FM
target

tree = rpart(target, data = cardio_train, method = "class")

install.packages("rpart.plot")
library("rpart.plot")
rpart.plot(tree)
predictions = predict(tree, cardio_test)
predictions

#----------------------------------------------#
#Another methos using party
#Decision Tree using Party Package

data <- read.csv("cardiotocographic.csv", stringsAsFactors = FALSE)
str(data)
data$NSPF <- factor(data$NSP)
str(data)

#Training and testing data
set.seed(1234)
pd <- sample(2, nrow(data), replace = TRUE, prob = c(0.8,0.2))
train <- data[pd==1,]
test <- data[pd==2,]

#Decision tree
install.packages("party")
library(party)
tree<- ctree(NSPF~LB+AC+FM,data = train, controls = ctree_control(mincriterion = 0.90, minsplit = 200))
tree

plot(tree)

#Prediction
predict(tree,test,type="prob")
predict(tree,test)

library(rpart)
tree1 <- rpart(NSPF ~ LB + AC + FM, train)
library(rpart.plot)
rpart.plot(tree1)

#misclassification error for training data
table <- table(predict(tree), train$NSPF)
table

1-sum(diag(table)/sum(table)) 
#amount of misclassification error

#misclassification error for testing data
predtest <- predict(tree, test)
table <- table(predtest, test$NSPF)
table
1-sum(diag(table)/sum(table))

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


#Multiple Linear Regression

install.packages("datarium")
library(datarium)
data("marketing")
View(marketing)
str(marketing)

#install.packages("ggplot2")
#library(ggplot2)
#ggpairs(marketing)
splitratio = 0.75
set.seed(101)
library(caTools)
sample = sample.split(marketing, SplitRatio = splitratio)
train = subset(marketing, sample == TRUE)
test = subset(marketing, sample == FALSE)
#train_size = dim(train)
#test_size = dim(test)
model <- lm(sales ~ youtube+facebook+newspaper, data = marketing)

pred <- predict(model, test)
pred

##applying multiple linear regression for 50_Startups

#multiple linear regression
# importing the dataset
dataset=read.csv("50_Startups.csv")
str(dataset)
View(dataset)
# checking for null values
sum(is.na(dataset))
colnames(dataset)[1]<-"Research"
colnames(dataset)[3]<-"Marketing"
install.packages('DataExplorer')
library('DataExplorer')
plot_correlation(dataset)
#create_report(dataset)
#handling categorical data
dataset$State=factor(dataset$State,levels=c('New York','Florida','California'),
                     labels=c(1,2,3))
View(dataset)

library(caTools)


#splitting the dataset
split=sample.split(dataset$Profit,SplitRatio=0.8)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

#fitting multiple linear regression model
regressor=lm(formula = Profit ~ Research+Administration+Marketing,
             data=training_set)
y_pred=predict(regressor,newdata=test_set)
y_pred

#prediction
df <- data.frame(Research = 165349.2 , Administration=136897.80 , Marketing= 471784.1)
result <- predict(regressor,df)
print(result)


library(ggplot2)
ggplot(training_set, aes(Administration, Profit)) +
  geom_smooth(method="lm") +
  geom_point(size=3) +
  theme_bw() + 
  xlab("R.D.Spend") +
  ylab("Profit") +
  ggtitle("Administration vs Profit")

#--------------------------------------------------------#

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

#-------------------------------------------------------------#
#06/october/2023
#Ann algo

getwd()
concrete = read.csv("concrete_Data.csv")
View(concrete)
str(concrete)

normalize <- function(x) { return((x-min(x)) / (max(x) - min(x)))}
concrete_form <- as.data.frame(lapply(concrete, normalize))

#now all the values are in the range of zero and one
summary(concrete_form$strength)

#actual values are larger than the normalized values
summary(concrete$strength)

concrete_train <- concrete_form[1:773,]
#75% training data

concrete_test <- concrete_form[774:1030, ]
#25% testing data

install.packages("neuralnet")
library(neuralnet)

concrete_model <- neuralnet(strength ~ ., data = concrete_train)

plot(concrete_model)

model_results <- compute(concrete_model, concrete_test[1:8])

predicted_strength <- model_results$net.result

cor(predicted_strength, concrete_test$strength)

concrete_model2 <- neuralnet(strength ~ ., data = concrete_train, hidden = 5)

plot(concrete_model2)

model_results2 <- compute(concrete_model2, concrete_test[1:8])

predicted_strength2 <- model_results2$net.result

cor(predicted_strength2, concrete_test$strength)
#______________________________________________________________#
# Load the necessary libraries
library(rpart)
library(caret)
library(e1071)

# Load the dataset (assuming you have 'mushrooms.csv' in your working directory)
mushrooms <- read.csv("mushrooms.csv")

# Check the structure of the dataset
str(mushrooms)

# Preprocess the data (if needed)
# For this example, we assume the dataset is already preprocessed.

# Split the data into training (80%) and testing (20%)
set.seed(123)  # Set seed for reproducibility
index <- createDataPartition(mushrooms$class, p = 0.8, list = FALSE)
train_data <- mushrooms[index, ]
test_data <- mushrooms[-index, ]

# Train a decision tree model
tree_model <- rpart(class ~ ., data = train_data, method = "class")

# Make predictions on the test data
predictions <- predict(tree_model, test_data, type = "class")

# Create a confusion matrix
confusion_matrix <- table(Actual = test_data$class, Predicted = predictions)

# Calculate the accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Print the confusion matrix and accuracy
print(confusion_matrix)
cat("Accuracy: ", accuracy, "\n")

# Plot the decision tree
library(rpart.plot)
rpart.plot(tree_model)

# Conclusion
# Analyze the decision tree and its splits to draw conclusions about the features that are important for classifying mushrooms as poisonous or edible.


#----------------------------------------------------------------#
##########SVM algorithm
install.packages("caret")
library(caret)
install.packages("tidyverse")
library(tidyverse)


social_data <- read.csv("social.csv")


social_data$purchased <- factor(social_data$purchased, levels = c(0, 1))

set.seed(123)
split <- sample.split(social_data$purchased, SplitRatio = 0.75)
train_set <- subset(social_data, split == TRUE)
test_set <- subset(social_data, split == FALSE)

# Scale the data
train_set[-5] <- scale(train_set[-5])
test_set[-5] <- scale(test_set[-5])

# Train the SVM model
svm_model <- svm(formula = purchased ~ ., data = train_set, type = "C-classification", kernel = "linear")

# Make predictions on the test set
test_predictions <- predict(svm_model, newdata = test_set[-5])

# Calculate the accuracy
accuracy <- mean(test_predictions == test_set$purchased)
print(accuracy)
#---------------------------------------------------------#
19/oct/2023
#CLustering using hierachical Clustering 
iris
iris1 = iris
iris1
iris1
d = dist(iris1, method = "euclidean")
hfit = hclust(d, method = "average")
plot(hfit)
grps = cutree(hfit, k=2)
grps
rect.hclust(hfit, k=2, border = "red")
#----------------------------------------------------#
######Clustering using k means
# Installing Packages
install.packages("arules")
install.packages("cluster")

# Loading package
#library(ClusterR)
library(cluster)
#??ClusterR
# Removing initial label of
# Species from original dataset
iris_1 <- iris[, -5]

# Fitting K-Means clustering Model
# to training dataset
set.seed(240) # Setting seed
kmeans.re <- kmeans(iris_1, centers = 3, nstart= 20)
#nstart means initial random number of centroids
#centers means no of clusters
kmeans.re

# Cluster identification for
# each observation
kmeans.re$cluster
kmeans.re$centers
# Confusion Matrix
cm <- table(iris$Species, kmeans.re$cluster)
cm

# Model Evaluation and visualization
#plot(iris_1[c("Sepal.Length", "Sepal.Width")])
#plot(iris_1[c("Sepal.Length", "Sepal.Width")],col = kmeans.re$cluster)
plot(iris_1[c("Sepal.Length", "Sepal.Width")],
     col = kmeans.re$cluster,
     main = "K-means with 3 clusters")

## Plotiing cluster centers
kmeans.re$centers
kmeans.re$centers[, c("Sepal.Length", "Sepal.Width")]

# cex is font size, pch is symbol
points(kmeans.re$centers[, c("Sepal.Length", "Sepal.Width")],
       col = 1:3, pch = 8, cex = 3)

## Visualizing clusters
y_kmeans <- kmeans.re$cluster
clusplot(iris_1[, c("Sepal.Length", "Sepal.Width")],
         y_kmeans,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste("Cluster iris"),
         xlab = 'Sepal.Length',
         ylab = 'Sepal.Width')
#lines =0 no distance lines between the elipses will be there
#shade = TRUE means elipses are shaded in relation to their intensity
#color = TRUE means eplises colored wrt density
#labels = 2 all points and ellipses are labelled in the plot
#plotchar= TRUE, then the plotting symbols differ for points 
#belonging to different clusters.
#span = TRUE each cluster is represented by the ellipse with smallest
#area containing all its points.
#---------------------------------------------------------------------------#
install.packages("ggplot2")
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$heart.disease, y = dataset$smoking),
             colour = 'red') +
  geom_line(aes(x = dataset$heart.disease, y = predict(model, newdata = dataset)),
            colour = 'blue') +
  ggtitle('heart disease (Linear regression)' ) +
  xlab('Reason') +
  ylab('level')
