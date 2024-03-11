a <- 2
b<-3
print(a+b)
print("Hello")
print(a*b)
print(a/b)

vector1<-c(1,23,4,5,5)
print(vector1)
vector2<-c(TRUE, FALSE, TRUE, TRUE)
print(vector1[-2])
print(vector1[-c(2,4)])
print(vector1[c(TRUE, TRUE, FALSE)])
numbers<-1:10
print(numbers)

my_list <- list(1, "hello", TRUE)
vecto3<-c(1, 2, 3, 4, 5)
new_list<-list(my_list, vecto3)
print(new_list)
#(09-08-2023)


subject1 <- list(fullname = "John",
                 temperature = 98.6,
                 flu_status = "TRUE",
                 gender = "Male")

#to add item
# Create an existing vector
original_vector <- c(1, 2, 3, 4)

# Values to append
values_to_append <- c(5, 6)

# Append values after position 2
new_vector <- append(original_vector, values_to_append, after = 2)

# Print the new vector
print(new_vector)
#factor
gender <- c("Male", "Female", "Male", "Female", "Male")
gender_factor <- factor(gender)


blood <- factor(c("h", "AB", "A"),levels=c("A", "B", "AB", "O"))
blood[1:2]
levels(blood)
length(blood)

# Creating a dataframe
dataframe <- data.frame(
  name = c("John", "Mary", "Hyka"),
  age = c(20, 21, 22),
  course = c("cse", "ece", "eee"),
  stringsAsFactors = "FALSE")

# Display the dataframe
print(dataframe[1])
dataframe[2, -1]
dataframe[c(2,3),-1]
dataframe[-1,-1]

# Creating a matrix
m <- matrix(c(3:14), nrow = 4, byrow = FALSE) #it will not fill by row

# Printing the matrix
print(m)

#define the column and row names
rownames = c("row1", "row2", "row3", "row4")
colnames = c("col1", "col2", "col3")

p <- matrix(c(3:14), nrow = 4, byrow = TRUE, dimnames = list(rownames, colnames))
print(p)

#vector
vector1 <- c(5,9,3)
vector2 <- c(10,11,12,13,14,15)
column.names <- c("col1", "col2", "col3")
row.names <- c("ROW1", "ROW2", "ROW3")
matrix.names<- c("Matrix1", "Matrix2")

#Take these vectors as input to the array
result <- array(c(vector1,vector2),dim = c(3,3,2),dimnames = list(row.names,column.names,matrix.names))
print(result)


view(usedcars)

usedcars <- read.csv("usedcars.csv", stringsAsFactors = FALSE)
write.csv(dataframe, "dataframe.RData")


x=2
y=3
z=1

mean(x, y, z)
median(x, y, z)

a=c(1:9)
mean(a)
median(a)


dataframe <- data.frame(
  name = c("John", "Mary", "Hyka"),
  age = c(20, 21, 22),
  course = c("cse", "ece", "eee"),
  stringsAsFactors = "FALSE")

write.csv(dataframe, "dataframe.RData", row.names = FALSE)

mean(c(36000, 44000, 56000))
median(c(36000, 44000, 56000))
