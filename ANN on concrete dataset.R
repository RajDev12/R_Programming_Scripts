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