'''
Logistic Regression Basic Template
'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

my_df = pd.read_csv("Projects/Data_Science/Data Science Infinity/Machine_learning/data/sample_data_classification.csv")

#Split data into input and output objects

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42, stratify=y)

#Instantiate model object
clf = LogisticRegression(random_state=42)

clf.fit(X_train, y_train)

#Assess model accuracy
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred) #0.8 at this point meaning we were able to successfully classify 80%

y_pred_prob = clf.predict_proba(X_test) #This returns the probability that the data points will fall into

#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

plt.style.use("seaborn-v0_8-poster")
plt.matshow(conf_matrix, cmap="coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()

'''
Advanced Logistic Regression Template
'''

from sklearn.utils import shuffle

data_for_model = pd.read_pickle("Projects/Data_Science/Data Science Infinity/Machine_learning/data/abc_classification_modelling.p")
data_for_model.drop("customer_id", axis=1, inplace=True)

#Shuffle data just in case there is an unknown order
data_for_model = shuffle(data_for_model, random_state=42)

# Take a look at class balance - the proportion of each class
data_for_model["signup_flag"].value_counts() #Here we see that the data isn't really that balanced but not too unbalanced

#Deal with missing values
data_for_model.isna().sum() #Since we see that there aren't that many rows with missing values, it would be safe to drop instead of impute
data_for_model.dropna(how="any", inplace=True)

#Deal with outliers
outlier_investigation = data_for_model.describe() #

outlier_columns = ["distance_from_store", "total_sales", "total_items"]

#Boxplot
for column in outlier_columns:
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    #Note that .index is important for storing just the index of the outliers.  Without it, it would save as a dataframe thus resulting
    #in a key error when trying to access it
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)
    

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
clf = LogisticRegression(random_state=42, max_iter = 1000) #number of iterations the model takes to find the optimal line
feature_selector = RFECV(clf)

fit = feature_selector.fit(X_train,y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}") #Results show optimal number of features is 7

X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()] #With this, it seems total sales is dropped

plt.plot(range(1, len(fit.cv_results_["mean_test_score"]) + 1), fit.cv_results_['mean_test_score'], marker = "o")
#range because we want the x axis values to go from 1 feature to 4 instead of 0 to 3 by default
# cv_results_ is the array that contains the accuracy scores at each number of variables that the algorithm found
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()
#Note that once we see the plot, it actually doesn't seem like there is a big difference between 7 or 8 - very minimal

# Model Training
clf = LogisticRegression(random_state=42, max_iter = 1000)
clf.fit(X_train, y_train)

# ========================================= Model Assessment ===========================================================

#Predict
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test) #This returns the probability that the data points will fall into
#The issue with the current prob variable is that it may be redundant to have probability of both classes whereas we really only need one
# So we update it to just have one
y_pred_prob = clf.predict_proba(X_test)[:,1]

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
accuracy_score(y_test, y_pred_class) #0.866

# Precision: Of all obervations that were predicted as positive, how many were actually positive?
precision_score(y_test, y_pred_class) #0.78 - each time we predicted a positive class, we were correct 78% of the time

#Recall: Of all positive observations, how many did we predict as positive?
recall_score(y_test, y_pred_class) #0.69

#F1 Score: harmonic mean of precision and recall
f1_score(y_test, y_pred_class) #0.774

# ============================================ Finding the optimal threshold ======================================
thresholds = np.arange(0,1, 0.01)

precision_scores = []
recall_scores = []
f1_scores = []

for threshold in thresholds:
    pred_class = (y_pred_prob >= threshold) * 1
    
    precision = precision_score(y_test, pred_class, zero_division=0)
    precision_scores.append(precision)

    recall = recall_score(y_test, pred_class)
    recall_scores.append(recall)
    
    f1 = f1_score(y_test, pred_class)
    f1_scores.append(f1)
    
max_f1 = max(f1_scores)
max_f1_idx = f1_scores.index(max_f1)

plt.style.use("seaborn-v0_8-poster")
plt.plot(thresholds, precision_scores, label = "Precision", linestyle = "--")
plt.plot(thresholds, recall_scores, label = "Recall", linestyle = "--")
plt.plot(thresholds, f1_scores, label = "F1", linewidth = 5)
plt.title(f"Finding the Optimal Threshold for Classification Model \n Max F1: {round(max_f1, 2)} (Threshold = {round(thresholds[max_f1_idx], 2)})")
plt.xlabel("Threshold")
plt.ylabel("Assessment Score")
plt.legend(loc = "lower left")
plt.tight_layout()
plt.show()
'''
From this result, we see that from the plot is that the best probability threshold is 0.44
At this threshold, we would get this optimal f1 score of 78%
'''

optimal_threshold = 0.44
y_pred_class_opt_thresh = (y_pred_prob >= optimal_threshold) * 1
# Now we have predictions based on our optimal threshold








