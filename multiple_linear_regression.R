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
  xlab("R.D.Spend") +
  ylab("Profit") +
  ggtitle("Administration vs Profit")


