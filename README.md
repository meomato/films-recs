## Movie Recommendation System (MovieLens 1M)

### Overview

This project builds a movie recommendation system using the MovieLens-1M dataset and compares several machine learning approaches for predicting user ratings. The goal is to estimate how a user would rate a movie and use these predictions to generate personalized recommendations.

The project includes the full data science pipeline: exploratory data analysis, feature engineering, model training, hyperparameter tuning, and model comparison.

### Dataset

The project uses the MovieLens-1M dataset with additional metadata and movie posters available on Kaggle:

[https://www.kaggle.com/datasets/mohamedelmallah1/movielens-1m-with-posters-and-metadata](https://www.kaggle.com/datasets/mohamedelmallah1/movielens-1m-with-posters-and-metadata)

The dataset contains three main tables:

1. users: demographic information about users
2. movies: movie titles, genres, metadata and posters
3. ratings: user ratings for movies

The target variable is the user rating assigned to a movie.

### Data Analysis and Preprocessing

EDA was performed to understand the structure of the dataset and the distribution of user activity and movie popularity.

Key observations include:

1. the user-movie interaction matrix is highly sparse
2. most users rate only a small fraction of available movies
3. ratings are concentrated around values between 3 and 5
4. some users and movies have significantly higher activity than others

Categorical features such as movie genres, user occupation and age groups were encoded using one-hot encoding. The release year of each movie was extracted from the movie title.

### Feature Engineering

Several additional features were created to improve model performance:

1. movie popularity (number of ratings per movie)
2. mean movie rating
3. Bayesian weighted rating to stabilize ratings for movies with few votes
4. average rating given by each user
5. number of ratings provided by each user
6. number of ratings received by each movie
7. deviation between a user’s average rating and the movie’s average rating

These features capture both user preferences and movie popularity patterns.

### Models

Several models were implemented and compared:

1. Matrix Factorization using user and movie embeddings
2. LightGBM gradient boosting model
3. CatBoost gradient boosting model
4. MLP combining embeddings with engineered features

Hyperparameters for the boosting models were tuned using RandomizedSearchCV with 5-fold GroupKFold cv based on user IDs.

The neural network uses user and movie embeddings together with additional features and applies dropout and optional batch normalization for regularization.

### Evaluation

Models were evaluated on a held-out test set using the following metrics:

1. RMSE
2. MAE
3. R^2 score

RMSE is the primary metric for rating prediction tasks.

### Results

| Model                | RMSE  | MAE   | R^2   |
| -------------------- | ----- | ----- | ----- |
| LightGBM             | 0.914 | 0.723 | 0.328 |
| CatBoost             | 0.915 | 0.720 | 0.328 |
| Matrix Factorization | 0.957 | 0.735 | 0.264 |
| MLP                  | 0.982 | 0.772 | 0.226 |

Gradient boosting models achieved the best performance. LightGBM produced the lowest RMSE, closely followed by CatBoost.

Matrix Factorization provides a strong collaborative filtering baseline but does not utilize additional engineered features. The neural network showed lower performance, likely due to the relatively small size of the dataset and the tabular nature of the features.

### Conclusion

The results demonstrate that feature engineering combined with gradient boosting models performs very well for recommendation problems based on tabular interaction data. LightGBM achieved the best predictive performance and can be used as the final model for generating recommendations.
