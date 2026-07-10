'''
This file explores common preprocessing techniques used before training machine
learning models. The focus is on preparing input data so that models can learn
more effectively, improve prediction accuracy, and reduce unnecessary complexity.

Topics Covered
--------------

1. Feature Scaling
   - Standardization (StandardScaler)
   - Normalization (MinMaxScaler)
   - When scaling is required and which algorithms benefit from it
   - Converting scaled NumPy arrays back into Pandas DataFrames

2. Feature Selection
   - Correlation matrices for exploring relationships between variables
   - Univariate feature selection using SelectKBest
   - F-tests for regression and Chi-Squared tests for classification
   - Understanding F-scores and p-values
   - Selecting important input variables based on statistical significance
   - Using transform() and get_support() to create reduced feature sets

3. Recursive Feature Elimination (RFECV)
   - Automatically selecting the optimal subset of features
   - Cross-validation for evaluating different feature combinations
   - Determining the best number of input variables
   - Visualizing model performance as features are added or removed
'''

# =============================================== Feature Scaling ==========================================================================
'''
Feature Scaling
Force the values from different columns to exist on the same scale
- Standardization: rescale data to mean = 0 and std dev = 1
- Normalization: rescale data to a range between 0 and 1
Choosing between these two methods may not matter in most situations, but there are some considerations:
    - if you need your values to remain positive, normalization - something like image data with pixel intensities
    - for algorithms like regression where you want to preserve the intensity, standardization
    
Feature scaling isn't a requirement for something like decision trees or random forest since they process the variables all independently, but for 
something like algorithms like k-means or knn where it relies on distance
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, chi2, RFECV
from sklearn.linear_model import LinearRegression

my_df = pd.DataFrame({"Height": [1.98,1.77,1.76,1.80,1.64],
                      "Weight": [99,81,70,86,82]})

#Standardization
scale_standard = StandardScaler()
scale_standard.fit_transform(my_df)

scale_standard.fit_transform(my_df["Height"]) #Scaler only applies to the height column
my_df_standardized = pd.DataFrame(scale_standard.fit_transform(my_df), columns = my_df.columns) #output as a dataframe

#Normalization
scale_norm = MinMaxScaler()
scale_norm.fit_transform(my_df)
my_df_normalized = pd.DataFrame(scale_norm.fit_transform(my_df), columns = my_df.columns)

# =========================================================== Feature Selection =================================================================
'''
Select the input variables that are most important to the machine learning task
There are a couple reasons for feature selection
- improved model accuracy(noisy/unecessary data)
- lower computational cost
- easier to understand and explain

But how do we know which variables are relevant?
- Correlation Matrix
- univariate feature selection (statistical tests to find relationships between input and output)
- recursive feature elimination (fit a model that starts with all input variables then iteratively removes those with weakest relationship)

'''
my_df = pd.read_csv("feature_selection_sample_data.csv")
correlation_matrix = my_df.corr()
# But just from the correlation matrix, there are still some unambiguous aspects.  In this example, input 1&2 seems to have high correlation 
# with the output, but input1 & 2 also seem to have high correlation with each other.  So do we include both input 1 & 2?
# So correlation matrix is limited in some ways, but it can give you a general idea about the relationship between variables
# This is where we can perhaps utilize univariate testing


#Regression
X = my_df.drop(["output"], axis = 1) #Just the input variables
y = my_df["output"] #Just the output

feature_selector = SelectKBest(f_regression, k = "all")
'''
electKBest selects the k best features, with 10 being the default and k=2 will mean 2 highest scoring features will be kept
f_regression is the scoring function.  SelectKBest doesn't know how to judge a feature by itself so you tell it which test to use which was
f test for regression

Also notice how fit takes in 2 arguments now.  For imputers, we only needed the input data, and needs to answer what is the mean of each column?  It doesn't need
to care about what the output is.  But for feature selection, we are figuring out which is useful for predicting output and to answer that, it must know the output
The program then takes the output

So in general, we initialize the object, then feature selector loops through every feature and runs an f-test then compares those results with other inputs
to come up with a score
'''
fit = feature_selector.fit(X,y)


p_values = pd.DataFrame(fit.pvalues_)
scores = pd.DataFrame(fit.scores_)
input_variable_names = pd.DataFrame(X.columns)

summary_stats = pd.concat([input_variable_names, p_values, scores], axis=1)
summary_stats.columns = ["input_variable", "p_value", "f_score"]
summary_stats.sort_values(by="p_value", inplace = True)

'''
Interpreting the results, the f score tells the signal strength and the p-value is the percentage of observing that event assuming the null hypothesis/
that there is no relationship.  So in this example, input 3 has a p value of 0.3286 which means that there is about a 32.9% probability of observing a 
relationship at least this strong if the feature actually has no relationship with the target.  
Remember, the F score here is a measure of the ratio of explained variance/unexplained variance where explained variance is The portion of the output's 
variation that can be explained by a feature and unexplained variance is The portion of the output's variation that the feature cannot explain like random noise
'''

p_value_threshold = 0.05
score_threshold = 5

selected_variables = summary_stats.loc[(summary_stats["f_score"] >= score_threshold) & (summary_stats["p_value"] <= p_value_threshold)]
selected_variables = selected_variables["input_variable"].tolist()
X_new = X[selected_variables]
#Now we have a dataset with only the variables that we believe to be important

# If we were to just select 2 - the transform method would automatically select the 2 best variables
feature_selector = SelectKBest(chi2, k = 2)
fit = feature_selector.fit(X,y)
X_new1 = feature_selector.transform(X) #But at this point, it only returns an array but not clear on what the columns/values actually mean

feature_selector.get_support() #This tells us which variables have been selected
X_new1 = X.loc[:, feature_selector.get_support()]

'''
But note that generally, we won't really know which value of k will be best straight off the bat
'''

#Classification
X = my_df.drop(["output"], axis = 1) #Just the input variables
y = my_df["output"] #Just the output

feature_selector = SelectKBest(f_regression, k = "all")
fit = feature_selector.fit(X,y)


p_values = pd.DataFrame(fit.pvalues_)
scores = pd.DataFrame(fit.scores_)
input_variable_names = pd.DataFrame(X.columns)

summary_stats = pd.concat([input_variable_names, p_values, scores], axis=1)
summary_stats.columns = ["input_variable", "p_value", "chi2_score"]
summary_stats.sort_values(by="p_value", inplace = True)

p_value_threshold = 0.05
score_threshold = 5

selected_variables = summary_stats.loc[(summary_stats["chi2_score"] >= score_threshold) & (summary_stats["p_value"] <= p_value_threshold)]
selected_variables = selected_variables["input_variable"].tolist()
X_new = X[selected_variables]


# ================================================================ Recursive Feature Elimination ==========================================================
'''
Recursive Feature Elimination with Cross Validation
This method splits all the data into different chunks and iteratively trains and validates the models of each chunk 
The algorithm can then determine which model provided the best accuracy thus infer the best set of input variables to use
'''

X = my_df.drop(["output"], axis=1)
y = my_df["output"]

regressor = LinearRegression()
feature_selector = RFECV(regressor)

fit = feature_selector.fit(X,y)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

X_new = X.loc[:, feature_selector.get_support()]

plt.plot(range(1, len(fit.cv_results_["mean_test_score"]) + 1), fit.cv_results_['mean_test_score'], marker = "o")
#range because we want the x axis values to go from 1 feature to 4 instead of 0 to 3 by default
# cv_results_ is the array that contains the accuracy scores at each number of variables that the algorithm found
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()

# From the output plot we see that 2 inputs give the highest score/performance

