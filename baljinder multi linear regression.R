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
sample=sample.split(marketing, SplitRatio = splitratio)
train = subset(marketing,sample==TRUE)
test=subset(marketing,sample==FALSE)
#train_size = dim(train)
#test_size = dim(test)
model <- lm(sales ~ youtube+facebook+newspaper, data = marketing)


pred <- predict(model, test)
pred
