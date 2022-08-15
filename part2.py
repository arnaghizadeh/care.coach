import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

# number of samples per cluster
pts = 50

# random number generator
rng = np.random.default_rng()

a = rng.multivariate_normal(
    mean=[0,0],
    cov=[[0.08,0],
         [0,4]],
    size=pts,
)

b = rng.multivariate_normal(
    mean=[3,3],
    cov=[[4,0],
         [0,0.08]],
    size=pts,
)

x = np.concatenate((a, b))

def kmeans(x, k, no_of_iterations):

    """ Function to implement k-means clustering
    # x is a (num_samples, 2) numpy array
    # k is number of clusters
    # no_of_iterations is the number of iterations to run k means
    """

    # Randomly choose initial Centroids
    idx = np.random.choice(len(x), k, replace=False)

    centroids = x[idx, :]
    distances = cdist(x, centroids ,'euclidean')
    points = np.array([np.argmin(i) for i in distances])

    # main loop
    for _ in range(no_of_iterations):

        centroids = []

        for idx in range(k):
            temp_cent = x[points==idx].mean(axis=0)
            centroids.append(temp_cent)
        centroids = np.vstack(centroids)
        distances = cdist(x, centroids ,'euclidean')
        points = np.array([np.argmin(i) for i in distances])

    return points

# use function
points = kmeans(x,k=2,no_of_iterations=10)

# assign clusters
cluster1 = x[points==0]
cluster2 = x[points==1]

# Plot the data in labeled clusters
plt.scatter(cluster1[:, 0], cluster1[:, 1])
plt.scatter(cluster2[:, 0], cluster2[:, 1])
plt.xlabel("x0")
plt.ylabel("x1")
plt.show()