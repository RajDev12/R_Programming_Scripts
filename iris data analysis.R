
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