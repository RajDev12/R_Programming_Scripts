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
