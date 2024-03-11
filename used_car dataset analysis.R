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