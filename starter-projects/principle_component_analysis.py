'''
Principle Component Analysis
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

data_for_model = pd.read_csv("data/sample_data_pca.csv")
# Current dataset contains customer ids and whether they have purchased the most recent album along with 100 columns of different artists that 
# the user had allocated their time to in percentage.  However, we don't know who those artists are

data_for_model.drop("user_id", axis=1, inplace=True)

#Shuffle data just in case there is an unknown order
data_for_model = shuffle(data_for_model, random_state=42)

# Take a look at class balance - the proportion of each class
data_for_model["purchased_album"].value_counts() #Here we see that the data isn't really that balanced but not too unbalanced

#Deal with missing values
data_for_model.isna().sum().sum() # This gives the total missing values in which there are non in this dataset
data_for_model.dropna(how="any", inplace=True)

#No need to deal with outliers


# Split input/output

X = data_for_model.drop(["purchased_album"], axis=1)
y = data_for_model["purchased_album"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42, stratify=y) #stratify allows training and tests sets to have the same number of 0/1

# Feature Scaling
scale_standard = StandardScaler()
X_train = scale_standard.fit_transform(X_train)
X_test = scale_standard.transform(X_test)

# Apply PCA
pca = PCA(n_components = None, random_state = 42) # None will create as many components as there are columns
pca.fit(X_train)

# Extract the explained variance across components
explained_variance = pca.explained_variance_ratio_
explained_variance_cumulative = pca.explained_variance_ratio_.cumsum()

# Create list for number of components
num_vars_list = list(range(1,101))
plt.figure(figsize = (15,10))

plt.subplot(2,1,1)
plt.bar(num_vars_list, explained_variance)
plt.title("Variance across Principal Components")
plt.xlabel("Number of components")
plt.ylabel("% Variance")
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(num_vars_list, explained_variance_cumulative)
plt.title("Cumulative Variance across Principal Components")
plt.xlabel("Number of components")
plt.ylabel("Cumulative % Variance")
plt.tight_layout()

plt.show()
# The plot doesn't show an exact clear point to decide how many components.  However, around 75% is explained by about 20-25 components

# Reinstantiate
pca = PCA(n_components = 0.75, random_state = 42) 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# Notice that in the new dataset we have 24 columns now

# We could also use pca.n_components_

# Apply PCA with selected number of components
clf = RandomForestClassifier(random_state = 42)
clf.fit(X_train, y_train)

# Assess Model Accuracy
y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)
# We see that it is about 0.92 accuracy with 24 components.  We can now apply this model to predict which customers are highly likely to pruchase the new album

'''
Remember it is not always necessary and you will also lose some information
PCA can be a great addition - principle components by definition are not correlated with each other so it can help to ensure your input variables
are not correlated which can be important for things like linear or logistic regression
But again, it is a tradeoff with interpretability


PRINCIPAL COMPONENT ANALYSIS OVERVIEW


DATA FORMAT - just as a note

Scikit-learn expects input data (X) in the following format:

Rows    = samples / observations
Columns = features / dimensions

X.shape = (number_of_samples, number_of_features)


Example:

             Dairy   Fruit   Meat   Vegetables
Customer A     20      30      40       10
Customer B     30      20      35       15
Customer C     10      50      20       20

If the dataset is reversed:

                 Customer A   Customer B   Customer C
Dairy                20           30           10
Fruit                30           20           50
Meat                 40           35           20
Vegetables           10           15           20

Scikit-learn will NOT automatically recognize that the rows are supposed
to be features, and it would interpret this as 4 samples 3 features so you would need to transpose it yourself

------------------------------------------------------------------------------------------------------------------------

PCA CONCEPT

PCA transforms a dataset with many features/dimensions into a new set of features called principal components.
This allows us to keep the components containing most of the information while removing components that contain relatively little information.


In the customer music dataset:
Original X:

              Artist 1   Artist 2   ...   Artist 100
Customer 1       value      value   ...      value
Customer 2       value      value   ...      value
Customer 3       value      value   ...      value

If there are 1000 customers:

X.shape = (1000, 100)

1000 samples
100 dimensions

PCA then creates new features:


If we choose to retain 75% of the variance, we may only need around
24 principal components:
~75% of the original variance retained


The data changes from something like:

Customer = [Artist1, Artist2, Artist3, ..., Artist100]

to:

Customer = [PC1, PC2, PC3, ..., PC24]


The number of customers/rows does NOT change.
Before PCA: X.shape = (1000, 100)
After PCA:X.shape = (1000, 24)
Only the representation of each customer changes.

-----------------------------------------------------------------------

Overall process:

100 artist features
        ↓
StandardScaler
        ↓
PCA finds the most important patterns/directions
        ↓
Keep enough components to retain 75% of the variance
        ↓
100 dimensions → approximately 24 dimensions
        ↓
Random Forest receives the 24 components as features
        ↓
Predict purchased_album
        ↓
0 or 1


IMPORTANT:

PCA does not change what each ROW represents. A row that represented Customer A before PCA still represents Customer A after PCA.
PCA changes what the COLUMNS represent:
Before PCA:
Columns = original features (artists)
After PCA:
Columns = principal components
'''

