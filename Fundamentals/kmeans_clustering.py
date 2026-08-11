'''
K Means Clustering
'''

from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# import data
my_df = pd.read_csv("data/sample_data_clustering.csv")

# Initial plot of the data
plt.scatter(my_df["var1"], my_df["var2"])
plt.xlabel("var1")
plt.ylabel("var2")
plt.show()

# Instantiate and fit
kmeans = KMeans(n_clusters = 3, random_state = 42)
kmeans.fit(my_df)

# Add the cluster labels to our df
my_df["cluster"] = kmeans.labels_
my_df["cluster"].value_counts() # See how many points are in each cluster

centroids = kmeans.cluster_centers_
print(centroids)

# Plot clusters
clusters = my_df.groupby("cluster")

for cluster, data in clusters:
    plt.scatter(data["var1"], data["var2"], marker = "o", label = cluster)
    plt.scatter(centroids[cluster,0], centroids[cluster, 1], marker = "x", color = "black", s = 300)
plt.legend()
plt.tight_layout()
plt.show()
