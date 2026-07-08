# Installing necessary packages 
install.packages(c(
  "tm",
  "tidyverse",
  "tidytext",
  "SnowballC",
  "caret",
  "e1071",
  "topicmodels",
  "text2vec"
))

# Load libraries
library(tm)
library(tidyverse)
library(tidytext)
library(SnowballC)
library(caret)
library(e1071)
library(topicmodels)

#Import data
IMDB_data <- read.csv("Processed_IMDB_Dataset.csv", stringsAsFactors = FALSE) #Set working directory with session in Rstudio

# Keep only needed columns 
IMDB_df <- IMDB_data[, c("Processed_Text", "sentiment")] 

# Naming the columns
colnames(IMDB_df) <- c("text", "sentiment")

# Convert the sentiment to numbers
IMDB_df$sentiment <- ifelse(IMDB_df$sentiment == "positive", 
                            1, 
                            0)

# Tokenization
IMDB_corpus <- VCorpus(VectorSource(IMDB_df$text))

# TF-IDF Vectorization 
IMDB_dtm <- DocumentTermMatrix(IMDB_corpus) 

# Remove sparse terms 
IMDB_dtm <- removeSparseTerms(IMDB_dtm, 
                          0.99) 
tfidf <- weightTfIdf(IMDB_dtm) 
#IMDB_dtm_mat <- as.matrix(IMDB_dtm) 
X <- as.matrix(tfidf)
Y <- as.factor(IMDB_df$sentiment)

#TRAIN/TEST data split
set.seed(42) 

train_index <- createDataPartition(Y, 
                                    p = 0.80, 
                                    list = FALSE) 
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
Y_train <- Y[train_index] 
Y_test <- Y[-train_index]

######################################################### 
# 1. SENTIMENT ANALYSIS 
# LOGISTIC REGRESSION 
######################################################### 
train_df <- as.data.frame(X_train) 
train_df$label <- Y_train 

test_df <- as.data.frame(X_test) 

log_model <- glm(label ~ ., data = train_df, family = binomial) 

log_probs <- predict(log_model, newdata = test_df, type = "response") 

log_pred <- ifelse(log_probs > 0.5, 1, 0) 

log_accuracy <- mean(log_pred == as.numeric(as.character(Y_test))) 

cat( "\nLogistic Regression Accuracy:", round(log_accuracy, 4), "\n")

######################################################### 
# 2. SPAM DETECTION
# NAIVE BAYES
######################################################### 

nb_model <- naiveBayes(X_train, Y_train)
nb_pred <- predict(nb_model, X_test)
nb_accuracy <- mean(nb_pred == Y_test)
cat("Spam/Classification Accuracy:", round(nb_accuracy, 2), "\n")

######################################################### 
# 3. TOPIC MODELING
# LDA (Clustering)
######################################################### 

count_dtm <- DocumentTermMatrix(IMDB_corpus)

num_topics <- 7

lda_model <- LDA(
  #IMDB_dtm,
  count_dtm, 
  k = num_topics,
  control = list(seed = 42)
)

topics <- terms(
  lda_model,
  10
)

cat("\nTOPICS IDENTIFIED:\n")

for(i in 1:num_topics){
  cat(
    "\nTopic",
    i,
    ":",
    paste(topics[,i], collapse = ", "),
    "\n"
  )
}

######################################################### 
# 4. K-MEANS CLUSTERING
######################################################### 

kmeans_model <- kmeans(X,
                       centers = 2,
                       nstart = 20)

IMDB_df$cluster <- kmeans_model$cluster

cat("\nCLUSTER COUNTS\n")

print( table(IMDB_df$cluster))

######################################################### 
# VISUALIZATION 1 
# SENTIMENT DISTRIBUTION 
######################################################### 
ggplot(IMDB_data, aes(sentiment, fill = sentiment)) + 
  geom_bar() + 
  labs(title = "Distribution of Sentiment Classes", x = "Sentiment", y = "Count" ) 

######################################################### 
# VISUALIZATION 2 
# CLUSTER DISTRIBUTION 
######################################################### 

cluster_df <- data.frame(Cluster = factor(IMDB_df$cluster)) 
ggplot(cluster_df, aes(Cluster, fill = Cluster)) + 
  geom_bar() + 
  labs(title = "K-Means Cluster Distribution", y = "Number of Reviews") 

######################################################### 
# VISUALIZATION 3 
# MODEL COMPARISON 
######################################################### 

results <- data.frame(Model = c("Logistic Regression", "Naive Bayes"), Accuracy = c(log_accuracy, nb_accuracy)) 

ggplot(results, aes(Model, Accuracy, fill = Model)) + 
  geom_bar(stat = "identity") + 
  ylim(0,1) + 
  labs(title = "Model Accuracy Comparison")
