# Author: Joshua Krasnogorov
# Date: 3/14/2025

# K-Means Clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

NUM_CENTROIDS=8

data = pd.read_csv('FinlandWhole.txt', delimiter=',', header=None, names=['X','Y'])

# random seed for reproducibility
np.random.seed(47)

X = data['X']
Y = data['Y']


# init random centroids
centroids_x = np.random.randint(np.min(X), np.max(X), NUM_CENTROIDS)
centroids_y = np.random.randint(np.min(Y), np.max(Y), NUM_CENTROIDS)

Distance = np.zeros((np.shape(X)[0], NUM_CENTROIDS))

epochs = 1000

for i in range(epochs):
    print(f"Epoch {i+1}/{epochs}")
    
    # calculate distance from each point to each centroid
    for j in range(NUM_CENTROIDS):
        Distance[:,j] = np.sqrt((X - centroids_x[j])**2 + (Y - centroids_y[j])**2)
    
    # find the minimum distance for each point
    min_distance = np.min(Distance, axis=1)
    
    # find the index of the minimum distance for each point
    min_distance_index = np.argmin(Distance, axis=1)
    
    # update the centroids to the data point closest to the mean of each cluster
    for j in range(NUM_CENTROIDS):
        points_in_cluster = min_distance_index == j
        if np.sum(points_in_cluster) > 0:
            # calculate the mean of this cluster
            mean_x = np.mean(X[points_in_cluster])
            mean_y = np.mean(Y[points_in_cluster])
            
            # find the point closest to the mean
            cluster_points_x = X[points_in_cluster]
            cluster_points_y = Y[points_in_cluster]
            
            # calculate distances from each point in cluster to the mean
            distances_to_mean = np.sqrt((cluster_points_x - mean_x)**2 + (cluster_points_y - mean_y)**2)
            
            # find the index of the closest point within this cluster
            closest_point_idx = np.argmin(distances_to_mean)
            
            # set the centroid to be this closest point
            centroids_x[j] = cluster_points_x.iloc[closest_point_idx]
            centroids_y[j] = cluster_points_y.iloc[closest_point_idx]
        else:
            # if no points in cluster, pick a random point
            random_idx = np.random.randint(0, len(X))
            centroids_x[j] = X.iloc[random_idx]
            centroids_y[j] = Y.iloc[random_idx]

plt.xlabel('x')
plt.ylabel('y')
plt.title('Clustering')

#plot data (blue) and centroids (red)
plt.scatter(X,Y,c='b')
plt.scatter(centroids_x,centroids_y,c='r')
plt.show()