# Group 6, Assignment 3.3
# IMDB

library(tidyverse)
library(tidytext)

#Load data
IMDB_data <- read.csv("Processed_IMDB_Dataset.csv") #Set working directory wiht session in Rstudio

# Step 1: Tokenize text and calculate term frequency per document

imdb_words <- IMDB_data %>%
  mutate(Row.ID = row_number()) %>%
  select(Row.ID, Processed_Text, sentiment) %>%
  unnest_tokens(output = word,
                input = Processed_Text,
                token = "words") %>%
  count(Row.ID, word, sentiment, sort = TRUE) %>%
  ungroup()

View(imdb_words)

# Step 2: Calculate total number of words per document

total_words <- imdb_words %>%
  group_by(Row.ID) %>%
  summarize(total = sum(n))

View(total_words)

# Step 3: Calculate TF-IDF weights

imdb_words <- left_join(imdb_words,
                        total_words,
                        by = "Row.ID")

imdb_words <- imdb_words %>%
  bind_tf_idf(term = word,
              document = Row.ID,
              n = n)

View(imdb_words)

# Step 4: Sort TF-IDF scores in descending order

imdb_words %>%
  select(-total) %>%
  arrange(desc(tf_idf)) %>%
  print()

# --------------------------------------------
# Unsupervised Learning: K-Means Clustering
# --------------------------------------------

# Step 5: Convert TF-IDF data into a document-term matrix

imdb_words_filtered <- imdb_words %>%
  group_by(word) %>%
  filter(n() >= 10) %>%   # keep only words appearing at least 10 times
  ungroup()

imdb_dtm <- imdb_words_filtered %>%
  cast_dtm(document = Row.ID,
           term = word,
           value = tf_idf)

# Convert sparse matrix to standard matrix
tfidf_matrix <- as.matrix(imdb_dtm)

# Set seed for reproducibility
set.seed(123)

# Step 6: Apply K-Means clustering

kmeans_model <- kmeans(tfidf_matrix,
                       centers = 3,
                       nstart = 20)

# Step 7: Display cluster distribution

cat("\n--- Unsupervised Learning (K-Means) Cluster Sizes ---\n")

print(table(kmeans_model$cluster))
