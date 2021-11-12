# Jester-Joke-Recommender
## About Dataset
The dataset for this problem can be found here https://www.kaggle.com/vikashrajluhaniwal/jester-17m-jokes-ratings-dataset?select=jester_ratings.csv 
Data is distributed across two CSV files:- jester_ratings.csv containing three columns [userID],[jokeID],[Ratngs] and other jester_items.csv mapping ratings to the actual jokes.
Ratings are real values ranging between -10 to +10.
There are few jokes rated by almost every user . These jokes are 7, 8, 13, 15, 16, 17, 18, 19

## About Problem
We will try to recommend top 10 jokes to a user given ratings that the user has already rated.
Performance Metric to be used is MAP@K(Mean average precision) more can be found about it here http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html

## Approach
I have tried Different approaches for this problem. Started with a simple Content filtering based system by representing our jokes using BOW,TF-IDF and Weighted
W2v. Then used Collaborative Filtering based model where similarity between each item is calulated as each item is represented by its user ratings. Also tried 
Eigen state algorithm which was presented by the team which collected this dataset.More about eigen state can be found here https://goldberg.berkeley.edu/pubs/eigentaste.pdf.
Then Tried Regression based approach where i was predicting the actual ratings by featurizing the data using MAtrix Factorization techniques.
Out of all the techniques Regression based approach was outperforming others on my metric. 
So my final model is an Ensemble model where we are having 10 distinct base learners and an Meta model (GBDT) which are predicting the ratings for us, and based on
the ratings the model predicts top 10 jokes to a given user.

Here are my RMSE and MAP@K scores for the model that i have tried:

MAP@k for Item-Item Simialrity Model using BOW representation is 7e-05

MAP@k for Item-Item Simialrity Model using TFIDF representation is 7e-05

MAP@k for Item-Item Simialrity Model using W2V representation is 0.00065

MAP@k for Collaborative Filtering Model using user Ratings as Vectors is 0.0025

Test RMSE and MAP@K for XGboost model: 3.42, 0.0234

Test RMSE and MAP@K for LAsso model: 4.64, 0.00339

Test RMSE and MAP@K for Ridge model: 4.522, 0.0053

Test RMSE and MAP@K for Linear SVR model: 4.57, 0.00929

Test RMSE and MAP@K for Decision Tree model: 5.0, 0.0046

Test RMSE and MAP@K for LGBM model: 3.84, 0.0179

Test RMSE and MAP@K for Ensemble model: 3.471, 0.0265

## How to run
Once we have downloaded the data from here https://www.kaggle.com/vikashrajluhaniwal/jester-17m-jokes-ratings-dataset?select=jester_ratings.csv 
run the following commands

python train.py

python test.py

train.py cotains the code to train our Ensemble model. Once the model is trained we can run test.py to get predictions for any user.

## Deployment
Above solution is also deplyed on heroku and can ne accessed using https://joke-recommendation-system.herokuapp.com/
