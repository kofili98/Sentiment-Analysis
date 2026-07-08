# Load libraries to use
install.packages(c(
  "tidyverse",
  "tidytext",
  "solitude",
  "stringr",
  "randomForest"
))

library(tidyverse)
library(tidytext)
data(stop_words)
library(solitude)
library(stringr)
library(randomForest)

# Load dataset
IMDB_data <- read.csv("Processed_IMDB_Dataset.csv", stringsAsFactors = FALSE) #Set working directory with session in Rstudio


# Text preprocessing and feature extraction
# Calculating unique word ratio
get_unique_ratio <- function(text_vector) {
  map_dbl(text_vector, function(txt) {
    words <- str_split(str_to_lower(txt), "\\s+")[[1]]
    words <- words[words != ""]
    if(length(words) == 0) return(0)
    return(length(unique(words)) / length(words))
  })
}

df_features <- IMDB_data %>% 
  mutate(
    # Character Count
    char_count = nchar(as.character(Processed_Text)),
    # Lowercase and clean punctuation
    clean_text = Processed_Text %>% 
      str_to_lower() %>% 
      str_replace_all("[^a-z\\s]", ""),
    # Compute Unique Word Ratio using helper function
    unique_word_ratio = get_unique_ratio(clean_text)
  )

# Tokenize and compute wor Count using tidytext
word_counts <- df_features %>%
  mutate(row_id = row_number()) %>%
  select(row_id, clean_text) %>%
  unnest_tokens(output = word, input = clean_text, token = "words") %>%
  # stopword removal
  filter(!word %in% stop_words$word) %>%
  count(row_id, name = "word_count")

# Combine all extracted features back into a single dataframe
df_final <- df_features %>%
  mutate(row_id = row_number()) %>%
  left_join(word_counts, by = "row_id") %>%
  # Handle case where a row has 0 words after stopword removal
  mutate(word_count = replace_na(word_count, 0))


# Random forest classification as supervised learning
# Extracted text features
ml_dataset <- df_final %>%
  select(word_count, char_count, unique_word_ratio, sentiment) %>%
  mutate(Target_Label = as.factor(sentiment)) %>%
  select(-sentiment) # Remove original character sentiment column

# Data splitting
set.seed(123)
train_idx <- sample(1:nrow(ml_dataset), 0.7 * nrow(ml_dataset))
train_set <- ml_dataset[train_idx, ]
test_set  <- ml_dataset[-train_idx, ]

# Training model
cat("\nTraining Random Forest model... \n")
rf_model <- randomForest(Target_Label ~ ., data = train_set, ntree = 200)

# Predicting on test dataset
rf_predictions <- predict(rf_model, test_set)

# Performance Evaluation
cat("\n--- Supervised Learning (Random Forest) Evaluation ---\n")
rf_confusion <- table(Predicted = rf_predictions, Actual = test_set$Target_Label)
print(rf_confusion)

# Classification accuracy
rf_accuracy <- sum(diag(rf_confusion)) / sum(rf_confusion)
cat("\nRandom Forest Model Accuracy:", round(rf_accuracy, 4), "\n")

# Binding test set features with predictions for a data frame
plot_data <- as.data.frame(test_set)
plot_data$Predicted_Label <- rf_predictions

# Visualizing results
ggplot(plot_data, aes(x = char_count, y = word_count, color = Predicted_Label)) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_manual(values = c("positive" = "blue", "negative" = "red")) +
  labs(
    title = "Random Forest Classification Results on Test Data",
    x = "Character Count",
    y = "Word Count",
    color = "Predicted Sentiment"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Displaying detected anomalies
mismatches <- test_set %>%
  mutate(Predicted_Label = rf_predictions) %>%
  filter(Target_Label != Predicted_Label)

View(mismatches)
