'''
Model Validation and Overfitting
'''

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold

my_df = pd.read_csv("Projects/Data_Science/Data Science Infinity/Data_Cleaning/feature_selection_sample_data.csv")

#Test/Train Split
X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
'''
If this was a classification model, there is one extra parameter: stratify = y
preserve the percentage of samples for each class
ex: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify = y)
'''

regressor = LinearRegression()
regressor.fit(X_train, y_train) #fit/train the model
y_pred = regressor.predict(X_test) #predict using the test set
r2_score(y_test, y_pred) #evalute the prediction using r2 score

'''
At this point, we should also consider that perhaps that the training and/or test sets may not have a representative spread of data and by chance, maybe 
the r2 value might be higher than it should be.  
To help confirm or deny this, we use cross validation
'''
#This variable will contain the accuracy scores of each model split, cv specifies the number of chunks of data(defulat = 5)
cv_scores = cross_val_score(regressor, X, y, cv = 4, scoring = "r2")
#The result is 0.78,0.58,0.45,0.74

cv_scores.mean() 
'''
we find the average cross validation score is around 0.64 which is quite a bit lower than what we previously saw of 0.83.  
This indicates that perhaps our first result/score was a little over inflated

But also note that cross_val_score functionality on its own does not contain a random state parameter for reproducibility of results
Also doesn't shuffle the data so if there is some unknown order in data, it may skew the result
So here, we use KFold
'''

cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X, y, cv = cv, scoring = "r2")
# And now we see that the mean value is slightly higher

# Classification
cv = StratifiedKFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(clf, X, y, cv = cv, scoring = "accuracy")
cv_scores.mean()