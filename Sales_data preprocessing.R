#preprocessing for sales data
s0 <- read.csv(file.choose(),header = TRUE, stringsAsFactors = TRUE)
str(s0)
summary(s0)

#Delete all the rows with missing data and name the new fataset as s1
s1 <- na.omit(s0)
View(s1)

summary(s1)
#Replace the missing values with mean vaue for each variable.
s0$Sales[is.na(s0$Sales)] <- mean(s0$Sales, na.rm = TRUE)
s0$Profit[is.na(s0$Profit)] <- mean(s0$Profit, na.rm = TRUE)
s0$Unit.Price[is.na(s0$Unit.Price)] <- mean(s0$Unit.Price, na.rm = TRUE)
summary(s0)



#########or other method###########

#Replacing the missing values for numerical variables

s0$Sales[is.na(s0$Sales)] <- runif(n = sum(is.na(s0$Sales)),
                                   min = min(s0$Sales, na.rm = TRUE),
                                   max = max(s0$Sales, na.rm = TRUE))

s0$Profit[is.na(s0$Profit)] <- runif(n = sum(is.na(s0$Profit)),
                                     min = min(s0$Profit, na.rm = TRUE),
                                     max = max(s0$Profit, na.rm = TRUE))
s0$Unit.Price[is.na(s0$Unit.Price)] <- runif(n = sum(is.na(s0$Unit.Price)),
                                             min = min(s0$Unit.Price, na.rm = TRUE),
                                             max = max(s0$Unit.Price, na.rm = TRUE))
summary(s0)

######------end-------#########

s0$Order.Priority[is.na(s0$Order.Priority)] <- sample(levels(s0$Order.Priority),
                                                      size = sum(is.na(s0$Order.Priority)),
                                                      replace = TRUE)

s0$Ship.Mode[is.na(s0$Ship.Mode)] <- sample(levels(s0$Ship.Mode),
                                            size = sum(is.na(s0$Ship.Mode)),
                                            replace = TRUE)

s0$Customer.Name[is.na(s0$Customer.Name)] <- sample(levels(s0$Customer.Name),
                                                    size = sum(is.na(s0$Customer.Name)),
                                                    replace = TRUE)

summary(s0)