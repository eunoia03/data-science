'''
Applying Grid szearch to find optimal hyperparameter
Applying to random forest regression
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

#Import sample data
my_df = pd.read_csv("Projects/Data_Science/Data Science Infinity/Machine_learning/data/sample_data_regression.csv")

#Split data into input and output objects
X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

#Instantiate Gridsearch object
gscv = GridSearchCV(
    estimator = RandomForestRegressor(random_state = 42),
    param_grid = {"n_estimators": [10, 50, 100, 500],
                  "max_depth": [1,2,3,4,5,6,7,8,9,10,None]},
    cv = 5,
    scoring = "r2",
    n_jobs = -1
)

'''
n_estimators is the number of decision trees built - default is 100
The more parameters you test the more combinations of models it will need to build train and test so make sure you consider the resources and time you have
cv is the number of partitions that the grid search process will use for cross validation
n_jobs = -1 mean it will use all of the computer's processes to run the task to speed things up
'''

# Fit to data
gscv.fit(X,y)

# Get the best CV score(mean)
gscv.best_score_ #The best mean was 0.647

# Optimal parameters
gscv.best_params_ # Best params show the best max depth was 3, n estimators 500

# Creating the optimal model object
regressor = gscv.best_estimator_ # We create the regressor object but this time, it has all of the parameter values of the optimal model