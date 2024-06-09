# Load Required Packages & Libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(tm)) install.packages("tm", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(SnowballC)) install.packages("SnowballC", repos = "http://cran.us.r-project.org") 

library(tidyverse)
library(tm)
library(caret)
library(e1071)
library(randomForest)
library(knitr)
library(SnowballC)


# Data Extraction
data = read.csv("fake_or_real_news.csv")

# Data Preparation

# Sneak peak into the data
head(data)
summary(data)

# How many rows and columns are there in the dataset?
dimensions <- dim(data)
names(dimensions) <- c("Number of Rows", "Number of Columns")
dimensions

# Renaming the column names
colnames(data) <- c("serial_number", "title", "text", "label")

# Checking for null values in title, text and label columns
count_of_empty_title <- sum(nchar(data$title) == 0)
print(paste("Count of null values in the title -",count_of_empty_title))

count_of_empty_text <- sum(nchar(data$Text) == 0)
print(paste("Count of null values in the text -",count_of_empty_text))

count_of_empty_label <- sum(nchar(data$label) == 0)
print(paste("Count of null values in the label -",count_of_empty_label))


# Adding a new columns to give the length, word count, number of punctuation and to see if they have numbers in both text and title
data <- data %>%
  mutate(
    label_digit = ifelse(label == "REAL", "1", "0"),
    title_length = nchar(title),
    text_length = nchar(text),
    title_word_count = sapply(strsplit(title, " "), length),
    text_word_count = sapply(strsplit(text, " "), length),
    title_has_number = grepl("\\d", title),
    text_has_number = grepl("\\d", text),
    title_punctuation_count = sapply(strsplit(title, "[[:punct:]]"), length),
    text_punctuation_count = sapply(strsplit(text, "[[:punct:]]"), length)
  )


# Cleaning the text in the data
preprocess_corpus = function(text) {
  
  # Convert the text to lower case
  text = tolower(text) 
  # Remove numbers
  text = removeNumbers(text)
  # Remove punctuation
  text = removePunctuation(text)
  # Remove common English stopwords
  text = removeWords(text, stopwords("en")) 
  # Stem words to root words
  text = stemDocument(text)   
  # Eliminate extra white spaces
  text = stripWhitespace(text) 
  
  return(text)
}

data$title = sapply(data$title, preprocess_corpus)
data$text = sapply(data$text, preprocess_corpus)


# Data Exploration

# Deep dive into news titles
# Representation of length of real and fake news titles
ggplot(data, aes(x = label, y = title_length, fill = label)) +
  geom_violin() +
  labs(title = "Title Length Distribution Comparison",
  subtitle = "Fake news articles tend to have longer titles than real news articles",
  x = "Real or Fake",
  y = "Title Length")+
  guides(none)+
  theme_classic()


# Representation of presence of numbers in the title of real and fake news
data_summary <- data %>%
  count(label, title_has_number)

ggplot(data_summary, aes(x = label, y = n, fill = title_has_number)) +
  geom_bar(stat = "identity") + 
  labs(title = "Presense of Numbers in Title Comparison",
  subtitle = "Most of the titles in both real and fake news, do not contain numbers",
  x = "Real or Fake",
  y = "Title Has Number Count",
  fill = "Title has Number") +
  theme_classic()


# Representation of number of punctuation in title of real and fake news
ggplot(data, aes(x = label, y = title_punctuation_count, fill = label)) +
  geom_violin() +
  labs(title = "Presense of Punctuation in Title Distribution Comparison",
  subtitle = "Fake news articles tend to use more punctuation marks than real news articles",
  x = "Real or Fake",
  y = "Title Punctuation Count")+
  guides(none)+
  theme_classic()


# Deep dive into news text
# Representation of word count of real and fake news texts
ggplot(data, aes(x = label, y = text_word_count, fill = label)) +
  geom_violin() +
  labs(title = "Text Word Count Distribution Comparison",
  subtitle = "Fake news articles tend to have less text content than real news articles",
  x = "Real or Fake",
  y = "Text Word Count")+
  guides(none)+
  theme_classic()


# Representation of numbers in the text of real and fake news
data_summary_2 <- data %>%
  count(label, text_has_number)

ggplot(data_summary_2, aes(x = label, y = n, fill = text_has_number)) +
  geom_bar(stat = "identity") +  
  labs(title = "Presense of Numbers in Text Comparison",
  subtitle = "Most of the texts in both real and fake news, do contain numbers",
  x = "Real or Fake",
  y = "Count of Text Which Has Number",
  fill = "Text has number") +
  theme_classic()


# Representation of number of punctuation in text of real and fake news
ggplot(data, aes(x = label, y = text_punctuation_count, fill = label)) +
  geom_violin() +
  labs(title = "Presense of Punctuation in Text Distribution Comparison",
  subtitle = "Fake news articles tend to use more punctuation marks than real news articles",
  x = "Real or Fake",
  y = "Text Punctuation Count")+
  guides(none)+
  theme_classic()


# Modeling Analysis

# Convert variable label into a factor data type
data$label = as.factor(data$label)

# train and test data split (80% - training, 20 - testing )
set.seed(1) 
splitIndex = createDataPartition(data$label, p = 0.8, list = FALSE)
train_data = data[splitIndex, ]
test_data = data[-splitIndex, ]


# Naive Bayes

# Naive bayes using the title features
# Model
title_nb_model = naiveBayes(label ~ title_length 
                            + title_word_count 
                            + title_punctuation_count
                            , data = train_data)

# Prediction
title_nb_prediction = predict(title_nb_model, newdata = test_data)

# Result
title_nb_accuracy = confusionMatrix(title_nb_prediction, test_data$label)$overall["Accuracy"]

results <- tibble(Model = "Naive Bayes - Using Title Features"
                      , Accuracy = round(title_nb_accuracy*100,2))
print(paste("Naive Bayes - Using Title Features: ",title_nb_accuracy*100))


# Naive bayes using the text features
# Model
text_nb_model = naiveBayes(label ~ text_length 
                           + text_word_count 
                           + text_has_number 
                           + text_punctuation_count
                           , data = train_data)

# Prediction
text_nb_prediction = predict(text_nb_model, newdata = test_data)

# Result
text_nb_accuracy = confusionMatrix(text_nb_prediction, test_data$label)$overall["Accuracy"]

results <- bind_rows(results
                     , tibble(Model ="Naive Bayes - Using Text Features"
                     , Accuracy = round(text_nb_accuracy*100,2)))
print(paste("Naive Bayes - Using Text Features: ",text_nb_accuracy*100))


# Naive bayes using the title and text features 
# Model
combined_nb_model = naiveBayes(label ~ title_length 
                               + title_word_count 
                               + title_has_number 
                               + title_punctuation_count 
                               + text_length 
                               + text_word_count 
                               + text_has_number 
                               + text_punctuation_count
                               , data = train_data)

# Prediction
combined_nb_prediction = predict(combined_nb_model, newdata = test_data)

# Result
combined_nb_accuracy = confusionMatrix(combined_nb_prediction, test_data$label)$overall["Accuracy"]

results <- bind_rows(results
                     , tibble(Model ="Naive Bayes - Using Title and Text Features"
                     , Accuracy = round(combined_nb_accuracy*100,2)))
print(paste("Naive Bayes - Using Title and Text Features: ",combined_nb_accuracy*100))

 
# Support Vector Machine

# Support Vector Machine using the title features 
# Model
title_svm_model = svm(label ~ title_length 
                      + title_word_count 
                      + title_has_number 
                      + title_punctuation_count
                      , data = train_data)

# Prediction
title_svm_prediction = predict(title_svm_model, newdata = test_data)

# Result
title_svm_accuracy = confusionMatrix(title_svm_prediction, test_data$label)$overall["Accuracy"]

results <- bind_rows(results
                     , tibble(Model = "Support Vector Machine - Using Title Features"
                     , Accuracy = round(title_svm_accuracy*100,2)))
print(paste("Support Vector Machine - Using Title Features: ",title_svm_accuracy*100))


# Support Vector Machine using the text features 
# Model
text_svm_model = svm(label ~ text_length 
                     + text_word_count 
                     + text_has_number 
                     + text_punctuation_count
                     , data = train_data)

# Prediction
text_svm_prediction = predict(text_svm_model, newdata = test_data)

# Result
text_svm_accuracy = confusionMatrix(text_svm_prediction, test_data$label)$overall["Accuracy"]

results <- bind_rows(results
                     , tibble(Model = "Support Vector Machine - Using Text Features"
                     , Accuracy = round(text_svm_accuracy*100,2)))
print(paste("Support Vector Machine - Using Text Features: ",text_svm_accuracy*100))


# Support Vector Machine using the title and text features 
# Model
combined_svm_model = svm(label ~ title_length 
                         + title_word_count 
                         + title_has_number 
                         + title_punctuation_count 
                         + text_length 
                         + text_word_count 
                         + text_has_number 
                         + text_punctuation_count
                         , data = train_data)

# Prediction
combined_svm_prediction = predict(combined_svm_model, newdata = test_data)

# Result
combined_svm_accuracy = confusionMatrix(combined_svm_prediction, test_data$label)$overall["Accuracy"]

results <- bind_rows(results
                     , tibble(Model = "Support Vector Machine - Using Title and Text Features"
                     , Accuracy = round(combined_svm_accuracy*100,2)))
print(paste("Support Vector Machine - Using Title and Text Features: ",combined_svm_accuracy*100))

 
# Random Forest

# Random Forest using the title features 
# Model
title_rf_model = randomForest(label ~ title_length 
                              + title_word_count 
                              + title_has_number 
                              + title_punctuation_count
                              , data = train_data)

# Prediction
title_rf_prediction = predict(title_rf_model, newdata = test_data)

# Result
title_rf_accuracy = confusionMatrix(title_rf_prediction, test_data$label)$overall["Accuracy"]

results <- bind_rows(results
                     , tibble(Model = "Random Forest - Using Title Features"
                     , Accuracy = round(title_rf_accuracy*100,2)))
print(paste("Random Forest - Using Title Features: ",title_rf_accuracy*100))


# Random Forest using the text features 
# Model
text_rf_model = randomForest(label ~ text_length 
                             + text_word_count 
                             + text_has_number 
                             + text_punctuation_count
                             , data = train_data)

# Prediction
text_rf_prediction = predict(text_rf_model, newdata = test_data)

# Result
text_rf_accuracy = confusionMatrix(text_rf_prediction, test_data$label)$overall["Accuracy"]

results <- bind_rows(results
                     , tibble(Model = "Random Forest - Using Text Features"
                     , Accuracy = round(text_rf_accuracy*100,2)))
print(paste("Random Forest - Using Text Features: ",text_rf_accuracy*100))


# Random Forest using the title and text features    
# Model
combined_rf_model = randomForest(label ~ title 
                                 + text 
                                 + title_length 
                                 + title_word_count 
                                 + title_has_number 
                                 + title_punctuation_count 
                                 + text_length 
                                 + text_word_count 
                                 + text_has_number 
                                 + text_punctuation_count
                                 , data = train_data)

# Prediction
combined_rf_prediction = predict(combined_rf_model, newdata = test_data)

# Result
combined_rf_accuracy = confusionMatrix(combined_rf_prediction, test_data$label)$overall["Accuracy"]

results <- bind_rows(results
                     , tibble(Model = "Random Forest - Using Title and Text Features"
                     , Accuracy = round(combined_rf_accuracy*100,2)))
print(paste("Random Forest - Using Title and Text Features: ",combined_rf_accuracy*100))


# Final Result  

results_final <- results %>% kable()
results_final

ggplot(results, aes(y = Model, x = Accuracy, fill = Accuracy)) +
  geom_col() +  # Adjust fill color as desired
  labs(title = "Model Performance on Fake News Detection",
  subtitle = "The highest is For the Model Random Forest - Using Title and Text Features",
  y = "Model Configuration", x = "Accuracy") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90),
  plot.title = element_text(hjust = 1.5),  
  plot.subtitle = element_text(hjust = 1))
