'''
Learning about preparing and cleaning data before ML processes
'''

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

#======================================================== Missing Values =======================================================================
'''
Imputers help replace missing values (NaN) in your dataset with reasonable substitute values so that machine learning models can use the data.
There is SimpleInputer that defaults to using the mean as substitute values and KNNImputer that uses the Kth nearest neighbours to compute the value
'''

my_df = pd.DataFrame({"A": [1,2,4,np.nan,5,np.nan,7],
                      "B": [4,np.nan,7,np.nan, 1,np.nan,2]})

#Finding Missing Values
my_df.isna()
my_df.isna().sum()

#Dropping Missing Values with Pandas
my_df.dropna()
my_df.dropna(how = "any") #drop any rows that have a missing value
my_df.dropna(how = "all") #drop only the rows that have missing values for all columns

my_df.dropna(how = "any", subset = ["A"]) #drop logic only looks at the subset parameter

my_df.dropna(how = "any", inplace = True) #In place actually allows us to commit the changes onto the dataframe

#Filling missing values with pandas
my_df = pd.DataFrame({"A": [1,2,4,np.nan,5,np.nan,7],
                      "B": [4,np.nan,7,np.nan, 1,np.nan,2]})

my_df.fillna(value = 100) #replace missing values with 100

mean_value = my_df["A"].mean()
my_df["A"].fillna(value = mean_value)
my_df.fillna(value = my_df.mean(), inplace = True)

# SimpleImputer
my_df = pd.DataFrame({"A" : [1,4,7,10,13],
                      "B" : [3,6,9,np.nan,15],
                      "C" : [2,5,np.nan,11,np.nan]})

imputer = SimpleImputer() #Create the imputer object
# By default, it replaces the missing value with the mean.  To change this, use strategy: SimpleImputer(strategy="median")

imputer.fit(my_df) #Learns the value needed from the input data
imputer.transform(my_df) # This uses what was learned using fit and replaces the missing values - returns an array

my_df1 = imputer.transform(my_df) #Again, note that this is just an array

imputer.fit_transform(my_df) #Only use fit_transform on training data

my_df2 = pd.DataFrame(imputer.fit_transform(my_df), columns = my_df.columns)  #Convert it back into a dataframe
#Note that my_df.columns - column names are stored in dataframe objects allowing us to specify the columns

imputer.fit_transform(my_df[["B"]]) #Only applicable to column B

#KNNImputer
my_df = pd.DataFrame({"A": [1,2,3,4,5],
                      "B": [1,1,3,3,4],
                      "C": [1,2,9,np.nan,20]})
knn_imputer = KNNImputer() #default k = 5, weight is uniform
knn_imputer = KNNImputer(n_neighbors=1)
knn_imputer.fit_transform(my_df)

#Give more weight to the points closer
knn_imputer = KNNImputer(n_neighbors=2, weights = "distance")
knn_imputer.fit_transform(my_df)

#Once again, the imputer returns an array so to turn this into a dataframe:
my_df_knn = pd.DataFrame(knn_imputer.fit_transform(my_df), columns= my_df.columns)


# ==================================================== Dealing with Categorical Variables ====================================================
'''
As an example, the location type when analyzing house prices may give valuable insight so it might be something to consider when building out model.
Our first instincts may be to assign a number to each category like -> Rural: 1, Urban: 2, etc.  And while this may work when we put it into a model,
the issue with this might be that we have imposed ssigned an order on these categorical variables where we don't really know if an order does exist.
This has the potential for the model to identify a false relationship or no relationship
Instead, we create dummy variables - one hot encoding(categorical variables as binary vectors)

Rural    Urban     Suburban
 0         1          0
 0         0          1
 1         0          0
 
 But we also have to waych out for the dummy variable trap where input variables perfectly predict each other creating multicollinearity.
 The solution would be to drop 1 column to prevent complete information
 At first it may seem like you're losing information, but in reality, the information of the dropped column can be inferred using other columns
  - if it isn't 1 column or the other, it must be the column that was dropped

'''

from sklearn.preprocessing import OneHotEncoder

X = pd.DataFrame({"Input1": [1,2,3,4,5],
                  "input2": ["A","A","B","B","C"],
                  "input3": ["X","X","X","Y","Y"]})

categorical_vars = ["input2", "input3"]

one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first") #sparse=False will return an array rather than an object
#drop="first" ensures that one of the columns are always dropped

encoded_vars_array = one_hot_encoder.fit_transform(X[categorical_vars])

#To see which column names are which
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

encoder_vars_df = pd.DataFrame(encoded_vars_array, columns = encoder_feature_names)
#Now we have the one hot encoder with the column names

X_new = pd.concat([X.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis=1) #axis so that pd knows to concat columns not rows
# Pandas aligns rows by index labels, not by position. That could create missing values where labels don't match.  So we reset the index
#Now that we have the one hot encoding done, we just need to drop the input 2&3

X_new.drop(categorical_vars, axis=1, inplace = True)
#Question: what happens if we specify axis=0 for rows?
# Pandas interprets input2,3 as row labels and so it will try to remove rows labeled input2,3 but there are none so it raises a KeyError

#===================================================================== Outliers ==================================================================
'''
Outliers can be any value that differs significantly from other values.  There are various ways to detect outliers
- Boxplots
- Standard deviation +-3
'''

my_df = pd.DataFrame({"input1": [15,41,44,47,50,53,56,59,99],
                      "input2": [29,41,44,47,50,53,56,59,66]})

my_df.plot(kind = "box", vert = False) #plot displayed horizontally

outlier_columns = ["input1", "input2"]

#Boxplot approach
for column in outlier_columns:
    lower_quartile = my_df[column].quantile(0.25)
    upper_quartile = my_df[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 1.5
    min_border = lower_quartile - iqr_extended
    max_border = lower_quartile + iqr_extended
    
    outliers = my_df[(my_df[column] < min_border) | (my_df[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    my_df.drop(outliers, inplace = True)
    

#Standard Deviation approach
my_df = pd.DataFrame({"input1": [15,41,44,47,50,53,56,59,99],
                      "input2": [29,41,44,47,50,53,56,59,66]})


for column in outlier_columns:
    mean = my_df[column].mean()
    std_dev = my_df[column].std()
    
    min_border = mean - std_dev * 3
    max_border = mean + std_dev * 3
    
    outliers = my_df[(my_df[column] < min_border) | (my_df[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    my_df.drop(outliers, inplace = True)




























