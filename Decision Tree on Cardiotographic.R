#Decision Tree using Party Package
#Classes =Normal, Suspect, Pathologic. NSP variable is representing classes. It is a categorical variable
getwd()
data <- read.csv("Cardiotocographic.csv",stringsAsFactors = FALSE )
View(data)
str(data)
data$NSPF<-factor(data$NSP)#integer variable will be converted into factor
str(data) #a new variable now added into existing dataset

#Training and Testing Data
set.seed(1234)
pd<-sample(2,nrow(data), replace = TRUE, prob = c(0.8,0.2)) #sample of size 2, no of rows as equal to data
#replacement as TRUE and probobability of sample as 80% training and 20% thttp://127.0.0.1:41481/graphics/3df968e5-25a4-49a7-9e2e-915e1aee0012.pngesting
train<-data[pd==1,]
test<-data[pd==2,]

#Decision Tree
install.packages("party")
library(party)
tree<-ctree(NSPF~LB+AC+FM,data = train, controls = 
              ctree_control(mincriterion = 0.90, minsplit = 200)) 
#using LB, AC, FM to classify data. controls is a parameter to control the length of the tree. mincriterian
#is the confidence level. It means that 90% confidence is there 
#that a variable is significant.minsplit is
#200 means that a tree will split into 2 when the sample size is atleast 200
tree
plot(tree)

#prediction
predict(tree,test,type="prob")
predict(tree,test)
#dim(predict(tree1,test))
#dim(train)
#Decision Tree with rpart package
library(rpart)
tree1 <-rpart(NSPF ~ LB+AC+FM, train)
library(rpart.plot)
rpart.plot(tree1)
#rpart.plot(tree1,
    #       type = 2)
#predict(tree,test)

#misclassification error for training data
table <- table(predict(tree), train$NSPF)

table
1-sum(diag(table)/sum(table)) #amount of misclassification error

#misclassification error for testing data
predtest <-predict(tree,test)
table <- table(predtest, test$NSPF)
table
1-sum(diag(table)/sum(table)) #amount of misclassification error
