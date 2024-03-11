
#______________________________________________________________#
  # Load the necessary libraries
  library(rpart)
library(caret)
library(e1071)

# Load the dataset (assuming you have 'mushrooms.csv' in your working directory)
mushrooms <- read.csv("mushrooms.csv")

# Check the structure of the dataset
str(mushrooms)

# Preprocess the data (if needed)
# For this example, we assume the dataset is already preprocessed.

# Split the data into training (80%) and testing (20%)
set.seed(123)  # Set seed for reproducibility
index <- createDataPartition(mushrooms$class, p = 0.8, list = FALSE)
train_data <- mushrooms[index, ]
test_data <- mushrooms[-index, ]

# Train a decision tree model
tree_model <- rpart(class ~ ., data = train_data, method = "class")

# Make predictions on the test data
predictions <- predict(tree_model, test_data, type = "class")

# Create a confusion matrix
confusion_matrix <- table(Actual = test_data$class, Predicted = predictions)

# Calculate the accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Print the confusion matrix and accuracy
print(confusion_matrix)
cat("Accuracy: ", accuracy, "\n")

# Plot the decision tree
library(rpart.plot)
rpart.plot(tree_model)

# Conclusion
# Analyze the decision tree and its splits to draw conclusions about the features that are important for classifying mushrooms as poisonous or edible.