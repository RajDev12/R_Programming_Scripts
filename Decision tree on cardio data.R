

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
