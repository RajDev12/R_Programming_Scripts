#preprocessing for sales data
S0 <- read.csv(file.choose(),header = TRUE, stringsAsFactors = TRUE)
str(S0)
summary(S0)

#Variables `Order.Priority`, `Sales`, `Ship.Mode`, `Profit`, `Unit.Price` and 
#`Customer.Name` have missing data.






#Delete all rows with missing data and name the new dataset as `S1`
S1 <- na.omit(S0)
View(S1)








summary(S1)
#Replace the missing values with the mean value of each variable.
S0$Sales[is.na(S0$Sales)] <- mean(S0$Sales, na.rm = TRUE)
S0$Profit[is.na(S0$Profit)] <- mean(S0$Profit, na.rm = TRUE)
S0$Unit.Price[is.na(S0$Unit.Price)] <- mean(S0$Unit.Price, na.rm = TRUE)
summary(S0)





#Replace the missing values with random value between min and max of each variable.

#Since missing values in S0 have been replaced with the mean value of each variable 
#in part, we need to import the dataset again to restore the missing values.use any method to import 

View(S0)
summary(S0)
S0 <- read.csv(file.choose(),stringsAsFactors = TRUE)
#Replacing the missing values for numerical variables:

S0$Sales[is.na(S0$Sales)] <- runif(n = sum(is.na(S0$Sales)),
                                   min = min(S0$Sales, na.rm = TRUE),
                                   max = max(S0$Sales, na.rm = TRUE))
S0$Profit[is.na(S0$Profit)] <- runif(n = sum(is.na(S0$Profit)),
                                     min = min(S0$Profit, na.rm = TRUE),
                                     max = max(S0$Profit, na.rm = TRUE))
S0$Unit.Price[is.na(S0$Unit.Price)] <- runif(n = sum(is.na(S0$Unit.Price)),
                                             min = min(S0$Unit.Price, na.rm = TRUE),
                                             max = max(S0$Unit.Price, na.rm = TRUE))
summary(S0)

#Missing values for variables `Order.Priority`, `Ship.Mode` and `Customer.Name` 
#cannot be replaced by the mean value, because these variables are categorical.
#Since categorical variables do not have min and max values, we can replace the 
#missing values for categorical variables by random value from each variable.

S0$Order.Priority[is.na(S0$Order.Priority)] <- sample(levels(S0$Order.Priority),
                                                      size = sum(is.na(S0$Order.Priority)),
                                                      replace = TRUE)
S0$Ship.Mode[is.na(S0$Ship.Mode)] <- sample(levels(S0$Ship.Mode),
                                            size = sum(is.na(S0$Ship.Mode)),
                                            replace = TRUE)
S0$Customer.Name[is.na(S0$Customer.Name)] <- sample(levels(S0$Customer.Name),
                                                    size = sum(is.na(S0$Customer.Name)),replace = TRUE)
summary(S0)
#Data Exploration
summary(S1)
sd(S1$Order.Quantity)
sd(S1$Sales)
sd(S1$Profit)
sd(S1$Unit.Price)
sd(S1$Shipping.Cost)
