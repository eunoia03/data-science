"""
Decision Trees for Regression
"""

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
import pickle

my_df = pd.read_csv("Projects/Data_Science/Data Science Infinity/Machine_learning/data/sample_data_regression.csv")

#Split data into input and output objects

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

#Instantiate model object
regressor = DecisionTreeRegressor(min_samples_leaf = 7) #this parameter limits the minimum data points to 7 per node
regressor.fit(X_train, y_train)

#Assess model accuracy
y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)

# Overfitting Demo
y_pred_training = regressor.predict(X_train)
r2_score(y_train, y_pred_training)

#plot decision tree
plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X.columns, 
                 filled = True,
                 rounded = True,
                 fontsize = 24)


'''
Regression Tree for ABC Grocery
'''


data_for_model = pickle.load(open("Projects/Data_Science/Data Science Infinity/Machine_learning/data/abc_regression_modelling.p", "rb"))
data_for_model.drop("customer_id", axis=1, inplace=True)

#Shuffle data just in case there is an unknown order
data_for_model = shuffle(data_for_model, random_state=42)

#Deal with missing values
data_for_model.isna().sum() #Since we see that there aren't that many rows with missing values, it would be safe to drop instead of impute
data_for_model.dropna(how="any", inplace=True)

# In decision trees, there is no need to deal with outliers

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
regressor = DecisionTreeRegressor(random_state=42, max_depth=4)
regressor.fit(X_train, y_train)

# ==============================================================================================================================
# Model Assessment
y_pred = regressor.predict(X_test)

r_squared = r2_score(y_test, y_pred)
print(r_squared) #0.90

# Cross Validation
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()
# Our mean cross validation score was 0.87 

#adjusted R2
# remember that adjusted R2 gives a more fair representation as each input variable only contributes(increase) to the R2 value
num_data_points, num_inputs_vars = X_test.shape
adjusted_r_squared = 1 - (1-r_squared) * (num_data_points - 1) / (num_data_points - num_inputs_vars -1)
print(adjusted_r_squared) #0.88

# ===============================================================================================================================
# Overfitting 
y_pred_training = regressor.predict(X_train)
r2_score(y_train, y_pred_training)
# This score shows a value of 1 which means it is definitely overfitting - add parameters - max depth


# Finding the best max depth
max_depth_list = list(range(1,9))
accuracy_scores = []

for depth in max_depth_list:
    regressor = DecisionTreeRegressor(max_depth = depth, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

# Plot of max depths
plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy by Max Depth \n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy, 4)}")
plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()
#We see that the max depth is 7
    
#Plot the Model
plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X.columns, 
                 filled = True,
                 rounded = True,
                 fontsize = 16)
