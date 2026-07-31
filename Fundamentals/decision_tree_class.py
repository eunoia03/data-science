'''
Decision Tree for Classification
'''

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
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
clf = DecisionTreeClassifier(min_samples_leaf = 7, random_state=42) #this parameter limits the minimum data points to 7 per node
clf.fit(X_train, y_train)

#Assess model accuracy
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# Overfitting Demo
y_pred_training = clf.predict(X_train)
accuracy_score(y_train, y_pred_training)
#notice how even with overfitting the score is not 100.  That is because we put min_samples_leaf condition

#plot decision tree
plt.figure(figsize=(25,15))
tree = plot_tree(clf,
                 feature_names = X.columns, 
                 filled = True,
                 rounded = True,
                 fontsize = 24)

'''
Classification Tree for ABC Grocery
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
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
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
accuracy_score(y_test, y_pred_class) # 0.93

# Precision: Of all obervations that were predicted as positive, how many were actually positive?
precision_score(y_test, y_pred_class) #0.88 - each time we predicted a positive class, we were correct 88% of the time

#Recall: Of all positive observations, how many did we predict as positive?
recall_score(y_test, y_pred_class) #0.88

#F1 Score: harmonic mean of precision and recall
f1_score(y_test, y_pred_class) #0.88
# The reason the scores are the same is because there are equal number of erros on either side of the matrix


# Finding the best max depth
max_depth_list = list(range(1,15))
accuracy_scores = []

for depth in max_depth_list:
    clf = DecisionTreeClassifier(max_depth = depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

# Plot of max depths
plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy (F1 Scire) by Max Depth \n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy, 4)}")
plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy (F1 Score)")
plt.tight_layout()
plt.show()
#We see that the max depth is 9
    
#Plot the Model
plt.figure(figsize=(25,15))
tree = plot_tree(clf,
                 feature_names = X.columns, 
                 filled = True,
                 rounded = True,
                 fontsize = 16)



