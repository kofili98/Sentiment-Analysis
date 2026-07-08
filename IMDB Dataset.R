# reading the first dataset, narrowing down to 10000 rows due to large amount of data
IMDB_data <- read.csv("IMDB Dataset.csv", header = TRUE, stringsAsFactors = FALSE, nrows = 10000)

head(IMDB_data)

# installing packages needed for cleaning dataset
install.packages("tm")
install.packages("textstem")

# loading the packages
library(tm)
library(textstem)

# Preprocessing text
preprocess_text <- function(text) {
  
  # 1. Lowercasing
  text <- tolower(text)
  
  # 2.Removing special characters and punctuation
  text <- gsub("<.*?>", " ", text)
  text <- gsub("[^a-z\\s]", "", text, perl = TRUE)
  
  
  # 3. Tokenization
  tokens <- unlist(strsplit(text, "\\s+"))
  tokens <- tokens[tokens != ""]
  
  # 4. Removing stopword
  stop_words <- tm::stopwords("english")
  tokens <- tokens[!tokens %in% stop_words]
  
  # 5. Lemmatization
  tokens <- textstem::lemmatize_words(tokens)
  cleaned_text <- paste(tokens, collapse = " ")
  
  return(cleaned_text)
}

# Applying to the dataframe
IMDB_data$Processed_Text <- sapply(IMDB_data$review, preprocess_text)

# Result
print(IMDB_data[, c("review", "Processed_Text")])

View(head(IMDB_data))

#Saving processed data in new file
write.csv(
  IMDB_data,
  "Processed_IMDB_Dataset.csv",
  row.names = FALSE
)
