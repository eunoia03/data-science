'''
This project uses KMeans Clustering to find whether there are distinct groups that have similar tendencies within all the customers that
shop at ABC grocery.
This file is the code base whereas explanations on the project itself is on danhokim.ca - you are what you eat
'''
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name = "transactions")
product_areas = pd.read_excel("data/grocery_database.xlsx", sheet_name = "product_areas")

# Merge on product area name
transactions = pd.merge(transactions, product_areas, how = "inner", on = "product_area_id")

# Drop the non-food category
transactions.drop(transactions[transactions["product_area_name"] == "Non-Food"].index, inplace = True)

# Aggregate sales at customer level (by product area)
transaction_summary = transactions.groupby(["customer_id", "product_area_name"])["sales_cost"].sum().reset_index()

# Pivot data to place product areas as columns
transaction_summary_pivot = transactions.pivot_table(index = "customer_id",
                                                    columns = "product_area_name",
                                                    values = "sales_cost",
                                                    aggfunc = "sum",
                                                    fill_value = 0,
                                                    margins = True,
                                                    margins_name = "Total").rename_axis(None, axis = 1)

'''
The code above
index
columns: what columns we want
values: what values we want in the cells in the column
aggfunc: aggregation functionality
fill_value: value to replace missing values when doing aggregation
margins: we want a total column since we want to tuirn it into percentages
rename_axis: so that we don't end up with both product area name and customer id in the index of the transaction summary pivot dataframe
'''

# Turn sales into % sales
transaction_summary_pivot = transaction_summary_pivot.div(transaction_summary_pivot["Total"], axis = 0)
# div allows us to divide all the values in the dataframe by a certain value

# drop the total column
data_for_clustering = transaction_summary_pivot.drop(["Total"], axis = 1)

# ================================================== Data Prep and Cleaning ==============================================================

# Check for missing values
data_for_clustering.isna().sum() # we see that we don't have any missing values

# Normalize data
# even though it's already spread between 0 to 1, one product area may commonly make up a large portion of customer sales and this might
# dominate the clustering

scale_norm = MinMaxScaler()
data_for_clustering_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustering), columns = data_for_clustering.columns)

# ================================== Use WCSS to find a good value for k ==================================================================
k_values = list(range(1,10))
wcss_list = []

for k in k_values:
    kmeans = KMeans(n_clusters = k, random_state=42, n_init = 10)
    kmeans.fit(data_for_clustering_scaled)
    wcss_list.append(kmeans.inertia_)
    
plt.plot(k_values, wcss_list)
plt.title("Within Cluster Sum of Squares by k")
plt.xlabel("k")
plt.ylabel("WCSS Score")
plt.tight_layout()
plt.show()

# The result shows that k = 3 is a sufficient k

# =================================== Instantiate and fit model =======================================================
kmeans = KMeans(n_clusters = 3, random_state=42, n_init = 10)
kmeans.fit(data_for_clustering_scaled)

'''
At this point, for each of these clusters, the algorithm calculates the mean values for the data points across each dimensions - 4
Then it repositions the centroids along each dimensions based on these mean values
Then reassign the points
Each data point could represent a customer's weekly spending:
Customer A = [10, 20, 30, 40]
Customer B = [20, 30, 40, 50]
Customer C = [30, 40, 50, 60]

 Suppose all 3 customers are currently assigned to the same cluster.

 Calculate the mean for EACH dimension:
 Dairy:
 (10 + 20 + 30) / 3 = 20

 Fruit:
 (20 + 30 + 40) / 3 = 30

 Meat:
 (30 + 40 + 50) / 3 = 40

 Vegetables:
 (40 + 50 + 60) / 3 = 50

 New centroid:
 [20, 30, 40, 50]

 K-Means then:
 1. Moves the centroid to [20, 30, 40, 50]
 2. Calculates the distance between every customer and each centroid
 3. Reassigns each customer to its closest centroid
 4. Calculates new mean values for each cluster
 5. Moves the centroids again
 6. Repeats until the centroids stop changing significantly
'''

# Add cluster labels
data_for_clustering["cluster"] = kmeans.labels_

# Check cluster sizes
data_for_clustering["cluster"].value_counts()

# Profile the clusters to make meaning of the results
cluster_summary = data_for_clustering.groupby("cluster")[["Dairy", "Fruit", "Meat", "Vegetables"]].mean().reset_index()