# Data Analysis Tasks in R

## September 1, 2023

1. **Loading SMS Spam Data:**
   - Loading and preprocessing SMS spam data:
     ```R
     sms_raw <- read.csv("sms_spam.csv", stringsAsFactors = FALSE)
     ```

2. **Text Processing:**
   - Preprocessing SMS text for analysis:
     ```R
     sms_corpus <- Corpus(VectorSource(sms_raw$text))
     sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
     sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)
     sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)
     sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords("english"))
     sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)
     ```

3. **Creating Document-Term Matrix:**
   - Generating a document-term matrix for analysis:
     ```R
     sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
     ```

4. **Model Training and Prediction:**
   - Training a Naive Bayes classifier and making predictions:
     ```R
     sms_train <- sms_dtm[1:4500, ]
     sms_test <- sms_dtm[4501:5559, ]
     sms_train_labels <- sms_raw$type[1:4500]
     sms_test_pred <- naiveBayes(sms_train, sms_train_labels, sms_test)
     ```

## September 3, 2023

1. **Reading Twitter Data:**
   - Loading and preprocessing Twitter data:
     ```R
     tweets <- read.csv("twitter_data.csv", stringsAsFactors = FALSE)
     ```

2. **Text Processing:**
   - Preprocessing tweet text for analysis:
     ```R
     tweets$text <- gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", tweets$text)
     tweets$text <- gsub("@\\w+", "", tweets$text)
     tweets$text <- gsub("[[:punct:]]", "", tweets$text)
     tweets$text <- gsub("[[:digit:]]", "", tweets$text)
     tweets$text <- gsub("http\\w+", "", tweets$text)
     tweets$text <- gsub("[ |\t]{2,}", "", tweets$text)
     tweets$text <- gsub("^\\s+|\\s+$", "", tweets$text)
     ```

3. **Sentiment Analysis:**
   - Analyzing sentiment of tweets using AFINN lexicon:
     ```R
     tweets$score <- sapply(tweets$text, function(x) sum(get_sentiments("afinn")$score[str_extract_all(tolower(x), "[a-z]+")]))
     ```

4. **Plotting Sentiment Analysis Results:**
   - Visualizing sentiment analysis results:
     ```R
     ggplot(data = tweets, aes(x = score)) +
       geom_histogram(binwidth = 1, fill = "blue", color = "black") +
       labs(title = "Sentiment Analysis of Tweets", x = "Sentiment Score", y = "Frequency") +
       theme_minimal()
     ```

## September 5, 2023

1. **Reading Amazon Reviews Data:**
   - Loading and preprocessing Amazon reviews dataset:
     ```R
     amazon_reviews <- read.csv("amazon_reviews.csv", stringsAsFactors = FALSE)
     ```

2. **Text Processing:**
   - Preprocessing review text for analysis:
     ```R
     amazon_reviews$text <- gsub("<.*?>", "", amazon_reviews$text)
     amazon_reviews$text <- gsub("[[:punct:]]", "", amazon_reviews$text)
     amazon_reviews$text <- gsub("[[:digit:]]", "", amazon_reviews$text)
     ```

3. **Topic Modeling:**
   - Conducting topic modeling on Amazon reviews:
     ```R
     amazon_corpus <- Corpus(VectorSource(amazon_reviews$text))
     amazon_dtm <- DocumentTermMatrix(amazon_corpus)
     amazon_lda <- LDA(amazon_dtm, k = 5, control = list(seed = 1234))
     ```

4. **Visualizing Topics:**
   - Visualizing topics generated by LDA model:
     ```R
     terms <- terms(amazon_lda, 10)
     ```

## September 6, 2023

1. **Loading Airbnb Listings Data:**
   - Loading and exploring Airbnb listings dataset:
     ```R
     airbnb_data <- read.csv("airbnb_listings.csv", stringsAsFactors = FALSE)
     ```

2. **Data Cleaning:**
   - Cleaning and preprocessing Airbnb data:
     ```R
     airbnb_data <- airbnb_data[complete.cases(airbnb_data), ]
     ```

3. **Descriptive Statistics:**
   - Calculating descriptive statistics for Airbnb listings:
     ```R
     summary(airbnb_data$price)
     ```

4. **Data Visualization:**
   - Visualizing distribution of Airbnb listing prices:
     ```R
     ggplot(data = airbnb_data, aes(x = price)) +
       geom_histogram(binwidth = 50, fill = "orange", color = "black") +
       labs(title = "Distribution of Airbnb Listing Prices", x = "Price", y = "Frequency") +
       theme_minimal()
     ```

## September 7, 2023

1. **Loading Stock Market Data:**
   - Loading and exploring stock market data:
     ```R
     stock_data <- read.csv("stock_data.csv", stringsAsFactors = FALSE)
     ```

2. **Data Exploration:**
   - Exploring structure and summary statistics of stock data:
     ```R
     str(stock_data)
     summary(stock_data)
     ```

3. **Data Preprocessing:**
   - Preprocessing stock market data for analysis:
     ```R
     stock_data$Date <- as.Date(stock_data$Date, format = "%Y-%m-%d")
     ```

4. **Time Series Visualization:**
   - Visualizing stock price trends over time:
     ```R
     ggplot(data = stock_data, aes(x = Date, y = Price)) +
       geom_line(color = "blue") +
       labs(title = "Stock Price Trends", x = "Date", y = "Price") +
       theme_minimal()
     ```

## September 8, 2023

1. **Loading and Exploring Telecom Churn Dataset:**
   - Loading and summarizing telecom churn dataset:
     ```R
     telecom_data <- read.csv("telecom_churn.csv", stringsAsFactors = FALSE)
     ```

2. **Data Preprocessing:**
   - Preprocessing telecom churn data for analysis:
     ```R
     telecom_data$Churn <- ifelse(telecom_data$Churn == "Yes", 1, 0)
     ```

3. **Feature Engineering:**
   - Creating new features from existing ones:
     ```R
     telecom_data$TotalCharges <- telecom_data$MonthlyCharges * telecom_data$tenure
     ```

4. **Correlation Analysis:**
   - Conducting correlation analysis of features:
     ```R
     cor(telecom_data[, c("tenure", "MonthlyCharges", "TotalCharges")])
     ```

## September 9, 2023

1. **Reading Diabetes Dataset:**
   - Loading and exploring diabetes dataset:
     ```R
     diabetes_data <- read.csv("diabetes_data.csv", stringsAsFactors = FALSE)
     ```

2. **Data Preprocessing:**
   - Preprocessing diabetes data for analysis:
     ```R
     diabetes_data <- diabetes_data[complete.cases(diabetes_data), ]
     ```

3. **Feature Selection:**
   - Selecting relevant features for modeling:
     ```R
     selected_features <- c("BMI", "Age", "Glucose", "Insulin", "SkinThickness", "DiabetesPedigreeFunction", "Outcome")
     diabetes_data <- diabetes_data[selected_features]
     ```

4. **Model Training:**
   - Training a logistic regression model:
     ```R
     diabetes_model <- glm(Outcome ~ ., data = diabetes_data, family = binomial)
     ```

5. **Model Evaluation:**
   - Evaluating logistic regression model performance:
     ```R
     summary(diabetes_model)
     ```

Now you have a broad overview of various data analysis tasks performed using R. Let me know if you need further assistance with any specific part or if you have any questions!
