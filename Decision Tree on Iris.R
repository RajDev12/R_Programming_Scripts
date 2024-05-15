install.packages("rpart")
library("rpart")
data("iris")
str(iris)


indexes = sample(150, 110)


indexes
iris_train = iris[indexes,]
iris_train

iris_test = iris[-indexes,]
iris_test

target = Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
target
tree = rpart(target, data = iris_train, method = "class")

install.packages("rpart.plot")
library("rpart.plot")
rpart.plot(tree)
predictions = predict(tree, iris_test)
predictions

