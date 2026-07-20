'''
Random Forest for Regression basic template
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
import pickle

my_df = pd.read_csv("Projects/Data_Science/Data Science Infinity/Machine_learning/data/sample_data_regression.csv")

#Split data into input and output objects

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

#Instantiate model object
regressor = RandomForestRegressor(random_state=42, n_estimators=1000) #n_estimators - number of decision trees built(100 by default)
regressor.fit(X_train, y_train)

#Assess model accuracy
y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)

# Feature Importance

regressor.feature_importances_ #this gives us an array of importance score of each input variables

feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis=1)
feature_importance_summary.columns = ["input_variable", "feature_importance"]
feature_importance_summary.sort_values(by="feature_importance", inplace=True)


plt.barh(feature_importance_summary["input_variable"], feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt. tight_layout()
plt.show()

''' =====================================================================================================================================
Random Forest more in depth
========================================================================================================================================'''

data_for_model = pickle.load(open("Projects/Data_Science/Data Science Infinity/Machine_learning/data/abc_regression_modelling.p", "rb"))
data_for_model.drop("customer_id", axis=1, inplace=True)

#Shuffle data just in case there is an unknown order
data_for_model = shuffle(data_for_model, random_state=42)

#Deal with missing values
data_for_model.isna().sum() #Since we see that there aren't that many rows with missing values, it would be safe to drop instead of impute
data_for_model.dropna(how="any", inplace=True)

# In random forest, there is no need to deal with outliers

# Split input/output
X = data_for_model.drop(["customer_loyalty_score"], axis=1)
y = data_for_model["customer_loyalty_score"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

# ===============================================================================================================================
# Deal with categorical variables 

categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first") #sparse=False will return an array rather than an object
#drop="first" ensures that one of the columns are always dropped

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

#To see which column names are which
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)


X_train_encoded= pd.DataFrame(X_train_encoded, columns = encoder_feature_names) #Now we have the one hot encoder with the column names
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1) 
# Pandas aligns rows by index labels, not by position. That could create missing values where labels don't match.  So we reset the index
X_train.drop(categorical_vars, axis=1, inplace = True)

X_test_encoded= pd.DataFrame(X_test_encoded, columns = encoder_feature_names) 
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1) 
X_test.drop(categorical_vars, axis=1, inplace = True)

# ==============================================================================================================================
# Feature Selection
# One thing to note is that although it isn't quite necessary, it isn't necessarily a bad thing to do feature selection and remove features
# since reducing the number of input variables will help reduce the computation cost
# That being said, it won't make any changes in terms of model accuracy, but could help in terms of computation

#===============================================================================================================================

# Model Training
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

# ==============================================================================================================================
# Model Assessment
y_pred = regressor.predict(X_test)

r_squared = r2_score(y_test, y_pred)
print(r_squared) #0.96

# Cross Validation
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()
# Our mean cross validation score was 0.92

#adjusted R2
# remember that adjusted R2 gives a more fair representation as each input variable only contributes(increase) to the R2 value
num_data_points, num_inputs_vars = X_test.shape
adjusted_r_squared = 1 - (1-r_squared) * (num_data_points - 1) / (num_data_points - num_inputs_vars -1)
print(adjusted_r_squared) #0.955

# ===============================================================================================================================
# Feature importance is looking at the impact/influence of each input variable on the predictive power

feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis=1)
feature_importance_summary.columns = ["input_variable", "feature_importance"]
feature_importance_summary.sort_values(by="feature_importance", inplace=True)

plt.barh(feature_importance_summary["input_variable"], feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt. tight_layout()
plt.show()
'''
The generate plot shows that distance from store seems to be the most important metric for predicting customer loyalty scores while the
other variables had relatively really low impact.

But another way we can see the feature importance is through permutation importance
As a reminder, permutation importance is the decrease in model performance when the features are randomly shuffled.  And this shuffled data
would destroy any relationship it had with the output variable and gives us an idea on how important it is
'''
result = permutation_importance(regressor, X_test, y_test, n_repeats=10, random_state = 42) 
#n_repeats: how many times we want to apply this random shuffling to each variable

#Now to visualize, we put everything into a dataframe and plot it
permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis=1)
permutation_importance_summary.columns = ["input_variable", "permutation_importance"]
permutation_importance_summary.sort_values(by="permutation_importance", inplace=True)

plt.barh(permutation_importance_summary["input_variable"], permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt. tight_layout()
plt.show()
#After this approach we see that distance from store was still the dominant feature but even proportionally larger 

#Predictions under the hood
y_pred[0]
new_data = [X_test.iloc[0]]
regressor.estimators_ # This gives us all the decision tree objects that were created

predictions = []
tree_count = 0
for tree in regressor.estimators_:
    prediction = tree.predict(new_data)[0]
    predictions.append(prediction)
    tree_count += 1
    
sum(predictions)/tree_count

pickle.dump(regressor, open("Projects/Data_Science/Data Science Infinity/Machine_learning/data/random_forest_regression_model.p", "wb"))
pickle.dump(one_hot_encoder, open("Projects/Data_Science/Data Science Infinity/Machine_learning/data/random_forest_regression_ohe.p", "wb"))
# Save objects through pickle













