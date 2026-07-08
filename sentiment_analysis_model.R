# ============================================================
# Unified Sentiment Analysis Model
# ============================================================
# This script combines all R analysis scripts into a single
# comprehensive sentiment analysis system that can:
# 1. Preprocess raw text
# 2. Extract features using TF-IDF
# 3. Train multiple classification models (Logistic Regression, Naive Bayes, Random Forest)
# 4. Perform topic modeling
# 5. Detect anomalies
# 6. Provide sentiment predictions on new text
# ============================================================

# ============================================================
# 1. SETUP AND INSTALLATION
# ============================================================

# Install required packages (uncomment if needed)
# install.packages(c(
#   "tidyverse",
#   "tidytext",
#   "tm",
#   "SnowballC",
#   "textstem",
#   "caret",
#   "e1071",
#   "topicmodels",
#   "text2vec",
#   "randomForest",
#   "solitude",
#   "stringr"
# ))

# Load libraries
library(tidyverse)
library(tidytext)
library(tm)
library(SnowballC)
library(textstem)
library(caret)
library(e1071)
library(topicmodels)
library(randomForest)
library(stringr)

data(stop_words)

# ============================================================
# 2. TEXT PREPROCESSING FUNCTIONS
# ============================================================

#' Preprocess raw text
#' 
#' @param text Character vector of text to preprocess
#' @return Character vector of cleaned text
#' 
preprocess_text <- function(text) {
  # 1. Lowercasing
  text <- tolower(text)
  
  # 2. Removing HTML tags and special characters
  text <- gsub("<.*?>", " ", text)
  text <- gsub("[^a-z\\s]", "", text, perl = TRUE)
  
  # 3. Tokenization
  tokens <- unlist(strsplit(text, "\\s+"))
  tokens <- tokens[tokens != ""]
  
  # 4. Removing stopwords
  stop_words_vec <- tm::stopwords("english")
  tokens <- tokens[!tokens %in% stop_words_vec]
  
  # 5. Lemmatization
  tokens <- textstem::lemmatize_words(tokens)
  
  # 6. Reconstruct text
  cleaned_text <- paste(tokens, collapse = " ")
  
  return(cleaned_text)
}

#' Extract text features for anomaly detection
#' 
#' @param text Character vector of text
#' @return Data frame with extracted features
#' 
extract_text_features <- function(text) {
  
  # Helper function to calculate unique word ratio
  get_unique_ratio <- function(text_vector) {
    map_dbl(text_vector, function(txt) {
      words <- str_split(str_to_lower(txt), "\\s+")[[1]]
      words <- words[words != ""]
      if(length(words) == 0) return(0)
      return(length(unique(words)) / length(words))
    })
  }
  
  # Extract features
  features <- data.frame(
    char_count = nchar(as.character(text)),
    unique_word_ratio = get_unique_ratio(text)
  )
  
  return(features)
}

# ============================================================
# 3. SENTIMENT ANALYSIS MODEL CLASS
# ============================================================

#' Sentiment Analysis Model
#' 
#' A comprehensive sentiment analysis system that combines
#' multiple machine learning approaches
#' 
SentimentModel <- R6::R6Class(
  "SentimentModel",
  public = list(
    # Model components
    log_model = NULL,
    nb_model = NULL,
    rf_model = NULL,
    lda_model = NULL,
    tfidf_vectorizer = NULL,
    
    # Data components
    training_data = NULL,
    vectorizer_corpus = NULL,
    
    #' Initialize model
    #'
    #' @param training_data Data frame with 'text' and 'sentiment' columns
    #' 
    initialize = function(training_data) {
      self$training_data <- training_data
      
      cat("Initializing Sentiment Analysis Model...\n")
      cat("Preprocessing text...\n")
      
      # Preprocess text
      training_data$processed_text <- sapply(
        training_data$text, 
        preprocess_text,
        USE.NAMES = FALSE
      )
      
      cat("Vectorizing text with TF-IDF...\n")
      self$vectorize_text(training_data$processed_text)
      
      cat("Training classification models...\n")
      self$train_models()
      
      cat("Training topic model...\n")
      self$train_topic_model()
      
      cat("Model initialization complete!\n\n")
    },
    
    #' Vectorize text using TF-IDF
    vectorize_text = function(text) {
      # Create corpus
      self$vectorizer_corpus <- VCorpus(VectorSource(text))
      
      # Create document-term matrix
      dtm <- DocumentTermMatrix(self$vectorizer_corpus)
      
      # Remove sparse terms
      dtm <- removeSparseTerms(dtm, 0.99)
      
      # Apply TF-IDF weighting
      self$tfidf_vectorizer <- weightTfIdf(dtm)
    },
    
    #' Train classification models
    train_models = function() {
      # Prepare data
      X <- as.matrix(self$tfidf_vectorizer)
      Y <- as.factor(self$training_data$sentiment)
      
      # Train-test split
      set.seed(42)
      train_index <- createDataPartition(Y, p = 0.80, list = FALSE)
      X_train <- X[train_index, ]
      X_test <- X[-train_index, ]
      Y_train <- Y[train_index]
      Y_test <- Y[-train_index]
      
      # Store training data for later use
      train_df <- as.data.frame(X_train)
      train_df$label <- Y_train
      
      test_df <- as.data.frame(X_test)
      
      # 1. Logistic Regression
      cat("  - Training Logistic Regression...\n")
      self$log_model <- glm(label ~ ., data = train_df, family = binomial)
      log_probs <- predict(self$log_model, newdata = test_df, type = "response")
      log_pred <- ifelse(log_probs > 0.5, 1, 0)
      log_accuracy <- mean(log_pred == as.numeric(as.character(Y_test)))
      cat("    Logistic Regression Accuracy:", round(log_accuracy, 4), "\n")
      
      # 2. Naive Bayes
      cat("  - Training Naive Bayes...\n")
      self$nb_model <- naiveBayes(X_train, Y_train)
      nb_pred <- predict(self$nb_model, X_test)
      nb_accuracy <- mean(nb_pred == Y_test)
      cat("    Naive Bayes Accuracy:", round(nb_accuracy, 4), "\n")
      
      # 3. Random Forest
      cat("  - Training Random Forest...\n")
      ml_dataset <- self$training_data %>%
        select(text, sentiment) %>%
        mutate(Target_Label = as.factor(sentiment)) %>%
        select(-sentiment)
      
      set.seed(123)
      train_idx <- sample(1:nrow(ml_dataset), 0.7 * nrow(ml_dataset))
      train_set <- ml_dataset[train_idx, ]
      
      self$rf_model <- randomForest(Target_Label ~ ., data = train_set, ntree = 200)
      cat("    Random Forest trained with 200 trees\n")
    },
    
    #' Train LDA topic model
    train_topic_model = function() {
      cat("  - Training LDA Topic Model...\n")
      
      # Create document-term matrix for LDA (needs counts, not TF-IDF)
      count_dtm <- DocumentTermMatrix(self$vectorizer_corpus)
      
      # Train LDA with 7 topics
      self$lda_model <- LDA(
        count_dtm,
        k = 7,
        control = list(seed = 42)
      )
      
      cat("    LDA model trained with 7 topics\n")
    },
    
    #' Extract topics from LDA model
    get_topics = function(num_words = 10) {
      if(is.null(self$lda_model)) {
        cat("LDA model not trained\n")
        return(NULL)
      }
      
      topics <- terms(self$lda_model, num_words)
      
      topic_list <- list()
      for(i in 1:ncol(topics)) {
        topic_list[[i]] <- topics[, i]
      }
      
      return(topic_list)
    },
    
    #' Predict sentiment for new text
    #'
    #' @param new_text Character vector of raw text to predict
    #' @return Data frame with predictions from each model
    #' 
    predict_sentiment = function(new_text) {
      # Preprocess text
      processed_text <- sapply(new_text, preprocess_text, USE.NAMES = FALSE)
      
      # Vectorize using same corpus structure
      new_corpus <- VCorpus(VectorSource(processed_text))
      
      # Create DTM using same dictionary
      new_dtm <- DocumentTermMatrix(
        new_corpus,
        control = list(dictionary = Terms(self$tfidf_vectorizer))
      )
      
      # Apply TF-IDF
      new_tfidf <- weightTfIdf(new_dtm)
      X_new <- as.matrix(new_tfidf)
      
      # Initialize results
      results <- data.frame(
        original_text = new_text,
        processed_text = processed_text
      )
      
      # Logistic Regression prediction
      if(!is.null(self$log_model)) {
        new_df <- as.data.frame(X_new)
        colnames(new_df) <- colnames(model.matrix(self$log_model))[-1]
        log_probs <- predict(self$log_model, newdata = new_df, type = "response")
        results$log_reg_score <- log_probs
        results$log_reg_sentiment <- ifelse(log_probs > 0.5, "positive", "negative")
      }
      
      # Naive Bayes prediction
      if(!is.null(self$nb_model)) {
        nb_pred <- predict(self$nb_model, X_new, type = "class")
        nb_prob <- predict(self$nb_model, X_new, type = "raw")
        results$nb_sentiment <- nb_pred
        results$nb_positive_prob <- nb_prob[, 2]
      }
      
      # Random Forest prediction
      if(!is.null(self$rf_model)) {
        rf_pred <- predict(self$rf_model, newdata = data.frame(text = processed_text), type = "class")
        results$rf_sentiment <- rf_pred
      }
      
      return(results)
    },
    
    #' Get ensemble prediction (majority vote)
    #'
    #' @param predictions Data frame from predict_sentiment()
    #' @return Character vector of ensemble sentiment predictions
    #' 
    get_ensemble_prediction = function(predictions) {
      votes <- data.frame(
        log_reg = predictions$log_reg_sentiment,
        naive_bayes = predictions$nb_sentiment,
        random_forest = predictions$rf_sentiment
      )
      
      ensemble <- apply(votes, 1, function(row) {
        pos_votes <- sum(row == "positive", na.rm = TRUE)
        neg_votes <- sum(row == "negative", na.rm = TRUE)
        if(pos_votes > neg_votes) "positive" else "negative"
      })
      
      return(ensemble)
    },
    
    #' Print model summary
    print_summary = function() {
      cat("\n========== SENTIMENT ANALYSIS MODEL SUMMARY ==========\n\n")
      
      cat("Training Data:")
      cat("\n  - Total samples:", nrow(self$training_data))
      cat("\n  - Vocabulary size:", length(Terms(self$tfidf_vectorizer)))
      cat("\n\nModels Trained:")
      cat("\n  - Logistic Regression: ", !is.null(self$log_model))
      cat("\n  - Naive Bayes: ", !is.null(self$nb_model))
      cat("\n  - Random Forest: ", !is.null(self$rf_model))
      cat("\n  - LDA Topic Model (7 topics): ", !is.null(self$lda_model))
      
      if(!is.null(self$lda_model)) {
        cat("\n\nTop Words by Topic:")
        topics <- self$get_topics(5)
        for(i in seq_along(topics)) {
          cat("\n  Topic", i, ":", paste(topics[[i]], collapse = ", "))
        }
      }
      
      cat("\n\n====================================================\n\n")
    }
  )
)

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

#' Quick sentiment prediction on new text
#'
#' @param model Trained SentimentModel object
#' @param text Character string or vector of text
#' @return Data frame with predictions and sentiment
#' 
analyze_sentiment <- function(model, text) {
  predictions <- model$predict_sentiment(text)
  predictions$ensemble_sentiment <- model$get_ensemble_prediction(predictions)
  return(predictions)
}

#' Generate visualization of sentiment predictions
#'
#' @param predictions Data frame from predict_sentiment()
#' @export
#' 
visualize_predictions <- function(predictions) {
  
  # Prepare data for visualization
  sentiment_counts <- data.frame(
    Model = c("Logistic Regression", "Naive Bayes", "Random Forest", "Ensemble"),
    Positive = c(
      sum(predictions$log_reg_sentiment == "positive", na.rm = TRUE),
      sum(predictions$nb_sentiment == "positive", na.rm = TRUE),
      sum(predictions$rf_sentiment == "positive", na.rm = TRUE),
      sum(predictions$ensemble_sentiment == "positive", na.rm = TRUE)
    ),
    Negative = c(
      sum(predictions$log_reg_sentiment == "negative", na.rm = TRUE),
      sum(predictions$nb_sentiment == "negative", na.rm = TRUE),
      sum(predictions$rf_sentiment == "negative", na.rm = TRUE),
      sum(predictions$ensemble_sentiment == "negative", na.rm = TRUE)
    )
  )
  
  # Convert to long format
  sentiment_long <- sentiment_counts %>%
    pivot_longer(cols = c(Positive, Negative), names_to = "Sentiment", values_to = "Count")
  
  # Create plot
  ggplot(sentiment_long, aes(x = Model, y = Count, fill = Sentiment)) +
    geom_bar(stat = "identity", position = "dodge") +
    scale_fill_manual(values = c("Positive" = "green", "Negative" = "red")) +
    labs(title = "Sentiment Predictions by Model",
         x = "Model",
         y = "Count",
         fill = "Sentiment") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# ============================================================
# 5. EXAMPLE USAGE
# ============================================================

# Example: Load training data
# training_data <- read.csv("Processed_IMDB_Dataset.csv") %>%
#   select(Processed_Text, sentiment) %>%
#   rename(text = Processed_Text)
# 
# # Create and train model
# sentiment_model <- SentimentModel$new(training_data)
# 
# # Print model summary
# sentiment_model$print_summary()
# 
# # Analyze new reviews
# new_reviews <- c(
#   "This movie was absolutely fantastic and entertaining!",
#   "I found this film to be boring and a waste of time.",
#   "The acting was excellent but the plot was confusing."
# )
# 
# predictions <- analyze_sentiment(sentiment_model, new_reviews)
# print(predictions)
# 
# # Visualize predictions
# visualize_predictions(predictions)

cat("Sentiment Analysis Model loaded successfully!\n")
cat("Load training data and initialize model with: sentiment_model <- SentimentModel$new(your_data)\n")
