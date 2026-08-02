'''
KNN for Classification
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.utils import shuffle
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
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
clf = KNeighborsClassifier() # n_neighbours sets k(default=5), weight
clf.fit(X_train, y_train)

#Assess model accuracy
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred) #accuracy score is simply the number of predictions we got correct vs number of predictions we made 

'''
KNN for ABC Grocery
'''

data_for_model = pd.read_pickle("Projects/Data_Science/Data Science Infinity/Machine_learning/data/abc_classification_modelling.p")
data_for_model.drop("customer_id", axis=1, inplace=True)

#Shuffle data just in case there is an unknown order
data_for_model = shuffle(data_for_model, random_state=42)

# Take a look at class balance - the proportion of each class
data_for_model["signup_flag"].value_counts() #Here we see that the data isn't really that balanced but not too unbalanced

#Deal with missing values
data_for_model.isna().sum() #Since we see that there aren't that many rows with missing values, it would be safe to drop instead of impute
data_for_model.dropna(how="any", inplace=True)

#Deal with outliers - best to just remove them since outliers may influence
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

# Feature Scaling
#Normalization
scale_norm = MinMaxScaler()
X_train = pd.DataFrame(scale_norm.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scale_norm.transform(X_test), columns = X_test.columns)
                       



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
clf = RandomForestClassifier(random_state=42) # note that too much may also hinder the model's accuracy
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
#Note that once we see the plot, it actually doesn't seem like there is a big difference between 3 or 8 - very minimal
# But we do see that 6 is the most optimal

# Model Training
clf = KNeighborsClassifier() #no random state because there is nothing random since it is distance based
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
accuracy_score(y_test, y_pred_class) #0.936

# Precision: Of all obervations that were predicted as positive, how many were actually positive?
precision_score(y_test, y_pred_class) #1 - we made no errors 
#Recall: Of all positive observations, how many did we predict as positive?
recall_score(y_test, y_pred_class) #0.76

#F1 Score: harmonic mean of precision and recall
f1_score(y_test, y_pred_class) #0.865

# ============================================ Finding the optimal max depth ======================================
# Finding the optimal of k
klist = list(range(2,25))
accuracy_scores = []

for depth in klist:
    clf = KNeighborsClassifier(n_neighbours = k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_k_value = klist[max_accuracy_idx]

# Plot of max depths
plt.plot(klist, accuracy_scores)
plt.scatter(optimal_k_value, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy (F1 Score) by k \n Optimal Value for k: {optimal_k_value} (Accuracy: {round(max_accuracy, 4)}")
plt.xlabel("k")
plt.ylabel("Accuracy (F1 Score)")
plt.tight_layout()
plt.show()
#We see that the max depth is 9
    












