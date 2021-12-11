
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#rm
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#predict movie ratings in the validation set (the final hold-out test set) as if 
#they were unknown

RMSE <- function (validation, edx){
  sqrt(mean((validation - edx)^2))
}

#For this project the data set is large but it does not have many dimensions, only
#6 variables, in which case distance and dimension reduction is not needed.

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
index <- createDataPartition(y = edx$rating, times = 1, p = 0.5, list = FALSE)
edx_test <- edx[index,]
edx_train <- edx[-index,]

edx_test <- edx_test %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

mu_hat <- mean(edx_train$rating)
RMSE(edx_train$rating, mu_hat)

#Remove movie bias
mu <- mean(edx_train$rating) 
movie_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
RMSE(predicted_ratings, edx_test$rating)

#Remove user effects
user_avgs <- edx_train %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
RMSE(predicted_ratings, edx_test$rating)

#There seems to be large variations across users when it comes to some genres.
#In romance it seems that many users are rating the same (3 - 4) and rarely below 2.5
#The results are larger varied with "Thriller" genre and it may be because user expectations
#on the movie is difficult to match, unlike "Romance" movies which are generally
#easily predictable
edx_test %>% group_by(movieId) %>% 
  filter(str_detect(genres,"Thriller"), n() >= 100) %>% 
  summarize(b_g = mean(rating)) 
#  ggplot(aes(b_u)) + 
#  geom_histogram(bins = 30, color = "black")

gen_avgs <- edx_train %>%
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>% 
#  filter(str_detect(genres,"Thriller")) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

predicted_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(gen_avgs, by = 'genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
RMSE(predicted_ratings, edx_test$rating)

#Penalize the movies with very low number of ratings to further help with the accuracy
#of the rating predictions
lambdas <- seq(0, 10, 0.5)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx_train$rating)
  
  b_i <- edx_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    edx_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, edx_test$rating))
})

lambdas[which.min(rmses)]

lambda <- 5
mu <- mean(edx_train$rating)
movie_reg_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

predicted_ratings <- edx_test %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
RMSE(predicted_ratings, edx_test$rating)

















