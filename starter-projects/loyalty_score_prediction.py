'''
Predicting Customer Loyalty Scores using random forest model
'''

import pandas as pd
import pickle

# Import data for customers to be scored
to_be_scored = pickle.load(open("Projects/Data_Science/Data Science Infinity/Machine_learning/data/abc_regression_scoring.p", "rb"))

#Import the saved model and model objects that we saved in the fundamentals files
# The model was created in "random_forest.py"
regressor = pickle.load(open("Projects/Data_Science/Data Science Infinity/Machine_learning/data/random_forest_regression_model.p", "rb"))
one_hot_encoder = pickle.load(open("Projects/Data_Science/Data Science Infinity/Machine_learning/data/random_forest_regression_ohe.p", "rb"))

# Since we don't use customer id column, we can drop it
to_be_scored.drop(["customer_id"], axis=1, inplace = True)

# Drop missing values
to_be_scored.dropna(how = "any", inplace = True)

# Apply one hot encoding
categorical_vars = ["gender"]

encoded_vars_array = one_hot_encoder.transform(to_be_scored[categorical_vars])

#To see which column names are which
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

encoder_vars_df = pd.DataFrame(encoded_vars_array, columns = encoder_feature_names)
#Now we have the one hot encoder with the column names

to_be_scored = pd.concat([to_be_scored.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis=1) #axis so that pd knows to concat columns not rows
# Pandas aligns rows by index labels, not by position. That could create missing values where labels don't match.  So we reset the index
to_be_scored.drop(categorical_vars, axis=1, inplace = True)

# Make the prediction
loyalty_predictions = regressor.predict(to_be_scored)
