
# coding: utf-8


import pandas as pd
data = pd.read_csv('Banknote-authentication-dataset.csv')
data

import numpy as np
variance_mean = np.mean(data['V1'])
skewness_mean = np.mean(data['V2'])
print(f"V1 mean = {variance_mean}")
print(f"V2 mean = {skewness_mean}")


v1_dev = np.std(data["V1"])
v2_dev = np.std(data["V2"])

print(f"standard deviation for v1 and v2 is {v1_dev} and {v2_dev}")


import matplotlib.pyplot as plt
plt.xlabel('V1')
plt.ylabel('V2')
plt.scatter(data['V1'],data['V2'])


#using kmeans to obtain data clusters
from sklearn.cluster import KMeans
v1 = data["V1"]
v2 = data['V2']
v1_v2 = np.column_stack((v1,v2))
print(v1_v2)

km_result = KMeans(n_clusters = 3).fit(v1_v2)
km_result.cluster_centers_
clusters = km_result.cluster_centers_

plt.scatter(v1,v2)
plt.scatter(clusters[:,0], clusters[:,1], s= 1000)

