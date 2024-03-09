a <- 2
b<-3
print(a+b)
print("Hello")
print(a*b)
print(a/b)

vector1<-c(1,23,4,5,5)
print(vector1)
vector2<-c(TRUE, FALSE, TRUE, TRUE)
print(vector1[-2])
print(vector1[-c(2,4)])
print(vector1[c(TRUE, TRUE, FALSE)])
numbers<-1:10
print(numbers)

my_list <- list(1, "hello", TRUE)
vecto3<-c(1, 2, 3, 4, 5)
new_list<-list(my_list, vecto3)
print(new_list)
#(09-08-2023)


subject1 <- list(fullname = "John",
                 temperature = 98.6,
                 flu_status = "TRUE",
                 gender = "Male")

#to add item
# Create an existing vector
original_vector <- c(1, 2, 3, 4)

# Values to append
values_to_append <- c(5, 6)

# Append values after position 2
new_vector <- append(original_vector, values_to_append, after = 2)

# Print the new vector
print(new_vector)
#factor
gender <- c("Male", "Female", "Male", "Female", "Male")
gender_factor <- factor(gender)


blood <- factor(c("h", "AB", "A"),levels=c("A", "B", "AB", "O"))
blood[1:2]
levels(blood)
length(blood)

# Creating a dataframe
dataframe <- data.frame(
  name = c("John", "Mary", "Hyka"),
  age = c(20, 21, 22),
  course = c("cse", "ece", "eee"),
  stringsAsFactors = "FALSE")

# Display the dataframe
print(dataframe[1])
dataframe[2, -1]
dataframe[c(2,3),-1]
dataframe[-1,-1]

# Creating a matrix
m <- matrix(c(3:14), nrow = 4, byrow = FALSE) #it will not fill by row

# Printing the matrix
print(m)

#define the column and row names
rownames = c("row1", "row2", "row3", "row4")
colnames = c("col1", "col2", "col3")

p <- matrix(c(3:14), nrow = 4, byrow = TRUE, dimnames = list(rownames, colnames))
print(p)

#vector
vector1 <- c(5,9,3)
vector2 <- c(10,11,12,13,14,15)
column.names <- c("col1", "col2", "col3")
row.names <- c("ROW1", "ROW2", "ROW3")
matrix.names<- c("Matrix1", "Matrix2")

#Take these vectors as input to the array
result <- array(c(vector1,vector2),dim = c(3,3,2),dimnames = list(row.names,column.names,matrix.names))
print(result)


view(usedcars)

usedcars <- read.csv("usedcars.csv", stringsAsFactors = FALSE)
write.csv(dataframe, "dataframe.RData")


x=2
y=3
z=1

mean(x, y, z)
median(x, y, z)

a=c(1:9)
mean(a)
median(a)


dataframe <- data.frame(
  name = c("John", "Mary", "Hyka"),
  age = c(20, 21, 22),
  course = c("cse", "ece", "eee"),
  stringsAsFactors = "FALSE")

write.csv(dataframe, "dataframe.RData", row.names = FALSE)

mean(c(36000, 44000, 56000))
median(c(36000, 44000, 56000))


usedcars <- read.csv("usedcars.csv", stringsAsFactors = FALSE)
str(usedcars)
summary(usedcars[c("price", "mileage")])


range(usedcars$price)
diff(range(usedcars$price))
#quantile(usedcars$price, seq(from = 0, to = 1, by = 0.20))
table(usedcars$year)
table(usedcars$model)
table(usedcars$color)
model_table <- table(usedcars$model)
prop.table(model_table)
color_pct <- table(usedcars$color)
color_pct <- prop.table(color_pct) * 100
round(color_pct, digits = 1)
getwd()


#if-else in R programming to check whether the element is present in vector or not.
vector3<-c(1,2,3,4,5)

# Element to check for

a <- 5L

#solution
if (a %in% vector3) {
  print("yes")
} else {
  print("no")
}

#to check if element is an integer
if(is.integer(a)){
  print("Yes")
} else {
  print("No")
}

x<-switch(
  3,
  "first",
  "second",
  "third",
  "fourth"
)
print(x)
print(1:10)

e <- c(2,5,3,9,8,11,6)
count <- 0
for(x in e){
  if(x%%2==0){
    count = count+1
  }
}
print(count)

#R operators - R logical operators example for boolean vectors

a<-c(TRUE, TRUE, FALSE, FALSE)
b<-c(TRUE, FALSE, TRUE, FALSE)

print(a & b)
print(a | b)
print(!a)
print(a&&b)
print(a||b)

#importing the dataset
dataset = read.csv('Data.csv')
#dataset=read.csv(file.choose())
View(dataset)
str(dataset)

#labelling the variables

names(dataset)

names(dataset)<-c("Country","Age","Salary")


str(dataset)
View(dataset)

#Checking missing values
is.na(dataset)

sum(is.na(dataset))
View(dataset)

colSums(is.na(dataset))

dataset[1,]

missingdata<-dataset[!complete.cases(dataset), ]

missingdata
#sum(is.na(missingdata))
View(dataset)

#taking care of missing values

dataset$Age=ifelse(is.na(dataset$Age),
                   ave(dataset$Age, FUN = function(x) mean(x,na.rm=TRUE)),
                   dataset$Age)

dataset$Salary=ifelse(is.na(dataset$Salary),
                   ave(dataset$Salary, FUN = function(x) mean(x,na.rm=TRUE)),
                   dataset$Salary)

sum(is.na(dataset))
View(dataset)

str(dataset)

#handling the categorical values
dataset$Country=factor(dataset$Country,
                       levels=c('France','Spain','Germany'),
                       labels=c(1,2,3))

dataset$Purchased=factor(dataset$Purchased,
                       levels=c("No","Yes"),
                       labels=c(1,2,3))

#cleaning the data

#*replace it with mean
#*remove the rows containing NA
#*replace it with random number



#importing the dataset
s0 = read.csv('salesdata.csv')
#s0<- read.csv(file.choose(),header = TRUE,stringsAsFactors = TRUE)

View(s0)
str(s0)


#Checking missing values
is.na(s0)

sum(is.na(s0))


colSums(is.na(s0))


#deleting all rows with missing data and name the new dataset as 's1'

s1 <- na.omit(s0)
View(s1)

summary(s1)

#replace the missing values with the mean value of each variable
s0$Sales[is.na(s0$Sales)] <- mean(s0$Sales, na.rm = TRUE)
s0$Profit[is.na(s0$Profit)] <- mean(s0$Profit, na.rm = TRUE)
s0$Unit.Price[is.na(s0$Unit.Price)] <- mean(s0$Unit.Price, na.rm = TRUE)

summary(s0)


#**********************#
#load dataset again after cleaning environment
s0 <- read.csv(file.choose(),stringsAsFactors = TRUE)
#Replacing the missing values for numerical variables:

s0$Sales[is.na(s0$Sales)] <- runif(n=sum(is.na(s0$Sales)),
                                   min = min(s0$Sales, na.rm = TRUE),
                                   max = max(s0$Sales, na.rm = TRUE))

s0$Profit[is.na(s0$Profit)] <- runif(n=sum(is.na(s0$Profit)),
                                     min = min(s0$Profit, na.rm = TRUE),
                                     max = max(s0$Profit, na.rm = TRUE))

s0$Unit.Price[is.na(s0$Unit.Price)] <- runif(n=sum(is.na(s0$Unit.Price)),
                                             min = min(s0$Unit.Price, na.rm = TRUE),
                                             max = max(s0$Unit.Price, na.rm = TRUE))


summary(s0)


#**********************#

#*missing values for variables Order.Priority,Ship.Mode & Customer.Name
#*cannot be replaced by the mean value,because these variables are categorical
#*since categorical variables do not have min & max values,we can replace the 
#*missing values for categorical variables by random values from each variable

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
#Data Exploration
summary(s1)
sd(s1$Order.Quantity)
sd(s1$Sales)

#KNN
getwd()
wbcd <- read.csv("wisc_bc_data.csv", stringsAsFactors = FALSE)
str(wbcd)
wbcd <- wbcd[-1]
View(wbcd)
table(wbcd$diagnosis)
wbcd$diagnosis<- factor(wbcd$diagnosis, levels = c("B", "M"),
                        labels = c("Benign", "Malignant"))
round(prop.table(table(wbcd$diagnosis)) * 100, digits = 1)

summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
View(wbcd_n)
summary(wbcd_n$area_mean)
wbcd_train<-wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]
wbcd_train_labels <- wbcd[1:460, 1]
wbcd_test_labels <- wbcd[470:569, 1]

install.packages("class")
library(class)
wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test,
                      cl = wbcd_train_labels, k = 21)

install.packages("gmodels")
library(gmodels)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred,
           prop.chisq = FALSE)
aa<-table(wbcd_test_labels,wbcd_test_pred)
library(caret)
confusionMatrix(aa)
typeof(wbcd_train_labels)

#KNN for iris dataset
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
iris.training <- iris[ind==1, 1:4]
iris.training

iris.test <- iris[ind==2, 1:4]
iris.test
iris.trainLables <- iris[ind==1, 5]
iris.trainLables 
iris.testLables <- iris[ind==2, 5]
iris.testLables
iris.pred = knn(train = iris.training, test = iris.test,
                cl=iris.trainLables, k = 5)
library(gmodels)
CrossTable(x = iris.testLables, y = iris.pred,
           prop.chisq = FALSE)
tab<-table(iris.testLables,iris.pred)
tab


#this function divides the correct predictions by total number of predictions that 
tab
accuracy <-
  function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)
library(caret)
confusionMatrix(tab)


#*August 31,2023(Thursday)
#*Probabilistic Learning: Using Naive Bayes
#*
#*
#*

sms_raw <- read.csv("sms_spam.csv",stringsAsFactors = FALSE)
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
lapply(sms_corpus[1:2],as.character)
sms_corpus_clean <<- tm_map(sms_corpus,
                            content_transformer(tolower))
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

sms_corpus_clean <- tm_map(sms_corpus_clean,removeNumbers)
sms_corpus_clean <- tm_map(sms_corpus_clean,
                           removeWords, stopwords())
sms_corpus_clean <- tm_map(sms_corpus_clean,removePunctuation)


install.packages("SnowballC")
library(SnowballC)
wordStem(c("learns","learned","learning","learns"))
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

sms_corpus_clean <- tm_map(sms_corpus_clean,stripWhitespace)
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
sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]
inspect(sms_dtm_freq_train)
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2,
                   convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2,
                  convert_counts)
View(sms_train)
install.packages("e1071")

library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

sms_test_pred <- predict(sms_classifier, sms_test)
head(sms_test_pred)
a=table(sms_test_pred,sms_test_labels)
a
library(gmodels)
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))

#########end-----------------------
vector= -10:10
range(vector)

#Naive Bayes on iris dataset
data("iris")
View(iris)
summary(iris)
str(iris)
sum(is.na(iris))
table(iris$Species)
round(prop.table(table(iris$Species))*100,digits = 1)
normalize=function(x){
  num=x-min(x)
  denom=max(x)-min(x)
  return(num/denom)
}
iris_norm=as.data.frame(lapply(iris[1:4], normalize))
summary(iris_norm)
set.seed(1234)
ind=sample(2,nrow(iris),replace=T,prob=c(0.70,0.30))
##SAMPLING TECHNIQUE
##INDEXES=SAMPLE(150,100) =>IT WILL PICK RANDOMLY PICK 100 ROWS FROM INTEGER 1 TO 150row indexes WITHOUT REPLACEMENT 
ind
table(ind)
prop.table(table(ind))
round(prop.table(table(ind))*100,digits=1)
iris.training=iris[ind==1, 1:4]
iris.training
iris.test=iris[ind==2, 1:4]
iris.test
iris.trainlabels=iris[ind==1,5]
iris.trainlabels
iris.testlabels=iris[ind==2,5]
iris.testlabels

#Another sampling technique
library(e1071)
install.packages("caTools")
library(caTools)
split=sample.split(iris,SplitRatio=0.7)
View(split)

train_cl=subset(iris,split="TRUE")
train_cl
test_cl=subset(iris,split="FALSE")
test_cl
##Feature scaling
##Scaling covert the values in the range of -1 to 1.
##But Normalization ranges the values between  0  to 1
train_scale=scale(train_cl[,1:4])
train_scale
test_scale=scale(test_cl[,1:4])

##Fitting naive bayes model
set.seed(120)
classifier_cl=naiveBayes(train_scale,train_cl[5])
##prediction on the testv data
Y_pred=predict(classifier_cl, newdata=test_scale)

cm=table(test_cl$Species,Y_pred)
cm




library(e1071)
classifier=naiveBayes(iris.training,iris.trainlabels)
test_pred=predict(classifier,iris.test)
head(test.pred)
a=table(test_pred,iris.testlabels)
a

require(gmodels)
CrossTable(x = iris.testlabels, y = test.pred,
           prop.chisq = FALSE)
##Accuracy mattrix
accuracy=function(x){
  sum(diag(x)/sum(rowSums(x)))*100
}
accuracy(p.matrix)

library(caret)
confusionMatrix(a)

#decision tree
install.packages("rpart")
library(rpart)
data(iris)
str(iris)


##Sampling technique 
## from 1 to 150 without replacement that is by default replace =FALSE
indexes=sample(150,110)
indexes
##fetch all the rowa in the indexES AND PUT ALL THE DATYA IN TRAIN-LABELS

iris_train=iris[indexes,]  ##[Rows, ALL COL]
iris_train
iris_test=iris[-indexes,]
iris_test

target = Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
#species~. it will select all column
target
tree = rpart(target, data = iris.train, method = "class")

install.packages("rpart.plot")
library("rpart.plot")
rpart.plot(tree)
predictions = predict(tree, iris_test)
predictions

#decision tree on cardio dataset
data=read.csv("Cardiotocographic.csv", stringsAsFactors = F)
data
View(data)

str(data)
sum(is.na(data))

##sampling 
indexes=sample(2126,1500)
indexes
data_train=data[indexes,]
data_train
data_test=data[-indexes,]

target=NSP~LB+AC+FM
target
tree=rpart(target,data=data_train,method='class')


library(rpart.plot)
rpart.plot(tree)
predictions=predict(tree,data_test)
predictions

install.packages("party")
library(party)


#decision tree on cardio dataset
data=read.csv("Cardiotocographic.csv", stringsAsFactors = F)
data
View(data)

str(data)
sum(is.na(data))

##sampling 
indexes=sample(2126,1500)
indexes
data_train=data[indexes,]
data_train
data_test=data[-indexes,]

target=NSP~LB+AC+FM
target
tree=rpart(target,data=data_train,method='class')


library(rpart.plot)
rpart.plot(tree)
predictions=predict(tree,data_test)
predictions
#decision tree using party package
#Classes = Normal, Suspect, Pathologic. NSP Variable is representing classes
data <- read.csv("Cardiotocographic.csv", stringsAsFactors = FALSE)
View(data)
str(data)
data$NSPF<-factor(data$NSP)#integer variable will be converted into factor
str(data)
#Training and Testing Data
set.seed(1234)
pd<-sample(2,nrow(data), replace = TRUE, prob = c(0.8,0.2))#sample of size
#replacement as TRUE and probability of sample as 80% training and 20% testing
train<-data[pd==1,]
test<-data[pd==2,]
#Decision Tree
install.packages("party")
library(party)
tree<-ctree(NSPF~LB+AC+FM,data = train, controls = 
              ctree_control(mincriterion = 0.90, minsplit = 200))
#using LB, AC, FM to classify data. controls is a parameter to control the 
#is the confidence level.it means that 90% confidence is there
#that a variable is significant.minsplit is 
#200 means that a tree will split into 2 when the sample size is atleast 200
tree
plot(tree)

#prediction
predict(tree,test,type = "prob")
predict(tree,test)
#dim(predict(tree1, test))
#dim(train)
#Decision Tree with rpart package
library(rpart)
tree1 <- rpart(NSPF ~ LB+AC+FM, train)
library(rpart.plot)
rpart.plot(tree1)
#rpart.plot(tree1, type = 2)
#predict(tree,test)

#misclassification error for training data
table <- table(predict(tree), train$NSPF)

table
1-sum(diag(table)/sum(table)) #amount of misclassification error

##height of a predictor vector
x=c(5.1,2.3,5.4,1.6,5.8,1.6,5.9,6.9,5.9,10)
#wight respionmse of the inde
y=c(63,66,69,65,68,62,56,75,56,65)


#lm()
relation=lm(y~x)
summary(relation)


a=data.frame(x=6.3)
result=predict(relation,a)
result



# Simple Linear Regression

#importing the dataset
dataset = read.csv("Salary_Data.CSV")
View(dataset)
#Splitting the dataset into the Training set and Test set
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#feature Scaling
#training_set = scale(training_set)
#test_set = scale(test_set)

#Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
#predicting the test set results
y_pred = predict(regressor, newdata = test_set)

#to visualize the training subset using scatter plot
plot(x=training_set$YearsExperience, y = training_set$Salary,
     main = "Scatterplot of Years of Experience vs. Salary",
     xlab = "Years pf Experience",
     ylab = "Salary")

#visualizing the Training set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience,y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')


# Multiple Linear Regression
install.packages('datarium')
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
train = subset(marketing,sample == TRUE)
test = subset(marketing,sample==FALSE)
#train_size = dim(train)
model <- lm(sales ~ youtube+facebook+newspaper, data = marketing)

pred <- predict(model, test)
pred



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

#5th oct 2023
dataset=read.csv("Position_Salaries.csv")
View(dataset)
dataset = dataset[2:3] #selecting only 2 colummns from dataset
View(dataset)
#Splitting the dataset into the training set and Test set
lin_reg = lm(formula = Salary ~ .,
             data = dataset)
lin_reg
#fitting polynomial regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)
View(dataset)

# Visualising the Linear Regression results
# install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Linear Regression)') +
  xlab('Level') +
  ylab('Salary')

#Visualizing the Polynomial Regression results
# install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

#predicting a new result with linear regression
predict(lin_reg, data.frame(Level = 6.5))

#predicting a new result with polynomial regression
predict(poly_reg, data.frame(Level = 6.5,
                             Level2 = 6.5^2,
                             Level3 = 6.5^3,
                             Level4 = 6.5^4))


#...................end......................................
#Visualizing the Regression Model results (for higher resolution and smoother)
# install.packages('ggplot2')
# Load the cars dataset
data(cars)

# Perform linear regression
lin_reg <- lm(formula = dist ~ speed, data = cars)


lin_reg
#Visualizing the linear Regression
library(ggplot2)
ggplot() + 
  geom_point(aes(x = cars$speed, y = cars$dist),
             colour = 'red') +
  geom_line(aes(x = cars$speed, y = predict(lin_reg, newdata = cars)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Linear Regression)') +
  xlab('Speed') +
  ylab('Distance')

#---------------------------------ANN------------------------------------
concrete = read.csv("Concrete_Data.csv")
View(concrete)
str(concrete)
#total 9 variables are there. one is 'strength' dependent on all the other 
#Neural network works best when the input data are scaled to a narrow
normalize <- function(x) { return((x - min(x)) / (max(x) - min(x)))}
concrete_norm <- as.data.frame(lapply(concrete, normalize))

summary(concrete_norm$strength)
#now all the values are in the range of zero and one

summary(concrete$strength)
#actual values are larger than the normalized values

concrete_train <- concrete_norm[1:773, ]
#75% training data

concrete_test <- concrete_norm[774:1030, ]
#25% testing data

#install.packages("neuralnet")
library(neuralnet)
#install the packages neuralnet for neural network implementation and load library

concrete_model <- neuralnet(strength ~ cement + slag + ash + water + superplasticizer + 
                              coarseagg + fineagg + age,
                            data = concrete_train)
#training the simplest multilayer feedforward network with only a single

plot(concrete_model)
#In this simple model, there is one input node for each of the eight features, follow
#and a single output node that predicts the concrete strength. Lower errors mean better accuracy

model_results = compute(concrete_model, concrete_test[1:8])
#It returns a list with two components: $neurons, which stores the neurons for each layers
#$net.results, which stores the predicted values.

predicted_strength <- model_results$net.result

cor(predicted_strength, concrete_test$strength)
#correlations close to 1 indicate strong linear relationships between two variables.

concrete_model2 <- neuralnet(strength ~ cement + slag + ash + water +
                               superplasticizer + coarseagg + fineagg + age,
                             data = concrete_train, hidden = 5)

plot(concrete_model2)
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$strength)

# ---------------------------12 October 2023------------------------------
#SVM on iris dataset
data = iris
str(data)
summary(data)
library(caTools)
set.seed(123)
split = sample.split(data$Species, SplitRatio = 0.75)
training_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

library(e1071)
?svm
classifier = svm(formula = Species ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')
#linear Kernel is used when the data is Linearly separable, that is,
#it can be separated using a single line. It is mostly used when there are 
y_pred = predict(classifier, newdata = test_set[-5])
y_pred

cm = table(test_set[, 5], y_pred)
cm
1 - sum(diag(cm)) / sum(cm)

plot(classifier, training_set, Petal.Width ~ Petal.Length,
     Slice = list(Sepal.Width = 3, Sepal.length = 4))
plot(classifier, test_set, Petal.Width ~ Petal.Length,
     slice = list(Sepal.Width = 3, Sepal.Length = 4))
#SVM plot visualising the iris data.Support vectors ae shown as 'X'

#------------------------END----------------------------------


#Neural Network on Boston 
library(MASS) 
#Boston dataset is present in MASS package
library(neuralnet)
#set seed so that we can get same values everytime
set.seed(123)
#storing boston dataset 
DataFrame<-Boston
#to get the details of Boston
help("Boston")
str(DataFrame)
hist(DataFrame$medv)
dim(DataFrame)
#range means min and max values of columns
apply(DataFrame,2,range)

#it will find the max value of every column
maxValue=apply(DataFrame,2,max)
maxValue
#It will find the min value for every column
minValue=apply(DataFrame, 2, min)
minValue
#sacle function give mean = 0 and standard deviation =1 for each variable
DataFrame = as.data.frame(scale(DataFrame,center = minValue, 
                                scale = maxValue-minValue))
#DataFrame = as.data.frame(scale(DataFrame,center = TRUE, 
#                               scale = TRUE))

#?scale
#used to create sample of 400 rows from 506 rows
x=sample(1:nrow(DataFrame),400) 
x
train = DataFrame[x,] #400 rows
test = DataFrame[-x,] #106 rows
neuralmodel = neuralnet(medv ~.,hidden = c(4,2),
                        data = train)
#in the model 13 are the input nodes. 2 hidden layers are there. first 
#consist of 4 nodes and second consist of 2
plot(neuralmodel)
model_results = compute(neuralmodel,test[1:13])
#str(model_results)
predicted <- model_results$net.result
cor(predicted, test$medv)

#--------------------------end---------------------
dataset = read.csv("social.csv")
str(dataset)
dataset = dataset[3:5]
View(dataset)
#Taking Columns 3-5
#dataset = dataset[3:5]
# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))
#splitting the dataset 
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification')

----------------------------------------------------
#Market basket on Grocerie
    
install.packages("arules")
library(arules)
getwd()
groceries <- read.transactions("groceries.csv", sep = ",")
#it results in a sparse matrix suitable for transactional data.
#summary(groceries) 
#details about no of transactions, no of items, density of non zero,
inspect(groceries[1:5])
#provide the detail of first 5 transactions
itemFrequency(groceries[, 1:3])
#allows us to see the proportion of transactions that contain the 
#item and to view the support level for
#the first three items in the grocery data
itemFrequencyPlot(groceries, support = 0.1)
#plot the bar chart using atleast 10% of support
itemFrequencyPlot(groceries, topN = 20)
#plot with 20 items
image(groceries[1:5])

#visualize the sparse matrix including 5 transactions and 169 items. 
#cell will be black where transaction
#is done 
image(sample(groceries, 100))
#combining it with the sample() function, you can view the sparse matrix 
#for a randomly sampled set of 
#transactions.
apriori(groceries) 
#by default support is 0.1 and confidence is 0.8
groceryrules <- apriori(groceries, parameter = 
                          list(support =
                                 0.006, confidence = 0.25, minlen = 2))
#for support, assumimg one item is getting purchased 2 times a day, 
#means 60 times a month. it means 
#60/9835 equals 0.006. for confidence consider a rule moving the
#smoke detectors closer to the
#batteries increase sale
groceryrules
#463 rules created
#summary(groceryrules)
#if lift more than 1, it means that the two items are found 
#together more often than one would expect
#by chance. Lift of greater than 1 means products A and B are more likely to be bought together.
inspect(groceryrules[1:3])
inspect(sort(groceryrules, by = "lift")[1:5])
#sort the best 5 rules from all the rules
berryrules <- subset(groceryrules, items %in% "berries")
#filter out all the rules having berries
inspect(berryrules)

write(groceryrules, file = "groceryrules22.csv",
      sep = ",",  row.names = FALSE)
#write rules to the csv file
groceryrules_df <- as(groceryrules, "data.frame")
str(groceryrules_df)

---------------------------------------------------------------------
                           #naive Bayes
        
#Slide number 12
head(subset(sms_test_prob, sms_test_prob[,1] > 0.40 & sms_test_prob[,1] < 0.60))
library(caret)
confusionMatrix(sms_test_pred, sms_test_labels, positive = "spam")

# ROCR
install.packages("ROCR")
library(ROCR)
pred <- prediction(prediction = sms_test_prob[,2], labels = sms_test_labels)

perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, main = "ROC curve for SMS spam filter", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)

#random Forest