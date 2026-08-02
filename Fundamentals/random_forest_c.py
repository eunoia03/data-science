'''
Random Forest for Classification
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
import pickle

my_df = pd.read_csv("Projects/Data_Science/Data Science Infinity/Machine_learning/data/sample_data_classification.csv")

#Split data into input and output objects
X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42, stratify=y)
#note again that stratify makes both training and test sets the same number of 0s and 1s in the overall data

#Instantiate model object
clf = RandomForestClassifier(random_state=42) 
clf.fit(X_train, y_train)

#Assess model accuracy
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred) #accuracy score is simply the number of predictions we got correct vs number of predictions we made 

# Overfitting 
# We may actually see some overfitting because scikit learn doesn't randomly limit the number of variables offered up at each specific split point



'''
Random Forest Classification for ABC Grocery
'''


data_for_model = pd.read_pickle("Projects/Data_Science/Data Science Infinity/Machine_learning/data/abc_classification_modelling.p")
data_for_model.drop("customer_id", axis=1, inplace=True)

#Shuffle data just in case there is an unknown order
data_for_model = shuffle(data_for_model, random_state=42)

# Take a look at class balance - the proportion of each class
data_for_model["signup_flag"].value_counts() #Here we see that the data isn't really that balanced but not too unbalanced

#Deal with missing values
data_for_model.isna().sum() 
data_for_model.dropna(how="any", inplace=True)

#No need to deal with outliers


# Split input/output

X = data_for_model.drop(["signup_flag"], axis=1)
y = data_for_model["signup_flag"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42, stratify=y) #stratify allows training and tests sets to have the same number of 0/1

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

# Feature Selection
# Again, there may be computational benefits to feature selection, but no real difference in terms of the model

# Model Training
clf = RandomForestClassifier(random_state=42, n_estimators=500, max_features=5) # number of decision trees
clf.fit(X_train, y_train)

# ========================================= Model Assessment ===========================================================

#Predict
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test) #This returns the probability that the data points will fall into
#The issue with the current prob variable is that it may be redundant to have probability of both classes whereas we really only need one
# So we update it to just have one
y_pred_prob = clf.predict_proba(X_test)[:,1]
 
# Remember that class predictions will be in the form of a 1 or 0 for each customer based on the default 50% threshold.  And this 50% threshold is based on the number of 
# decision trees within the random forest that came to the conclusion that the data point was either in the positive class or the negative class

#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)

plt.style.use("seaborn-v0_8-poster")
plt.matshow(conf_matrix, cmap="coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()

# Accuracy: The number of correct classifications out of all attempted classifications
accuracy_score(y_test, y_pred_class) # 0.935

# Precision: Of all obervations that were predicted as positive, how many were actually positive?
precision_score(y_test, y_pred_class) #0.887 - each time we predicted a positive class, we were correct 88% of the time

#Recall: Of all positive observations, how many did we predict as positive?
recall_score(y_test, y_pred_class) #0.90

#F1 Score: harmonic mean of precision and recall
f1_score(y_test, y_pred_class) #0.895
# The reason the scores are the same is because there are equal number of erros on either side of the matrix


# ===============================================================================================================================
# Feature importance is looking at the impact/influence of each input variable on the predictive power

feature_importance = pd.DataFrame(clf.feature_importances_)
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
The generate plot shows that distance from store still seems to be the most important metric for predicting customer loyalty scores while the
other variables had relatively really low impact.

But another way we can see the feature importance is through permutation importance
As a reminder, permutation importance is the decrease in model performance when the features are randomly shuffled.  And this shuffled data
would destroy any relationship it had with the output variable and gives us an idea on how important it is
'''
result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state = 42) 
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

'''
One interesting thing we see from this result is that there are some negative permutation scores for total sales and gender.
In these cases, the predictions on the shuffled data happened to be more accurate than on the real data.  So we can infer that the input variable doesn't really matter 

'''


