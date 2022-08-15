import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

# Part 1:

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

# Plot data
plt.scatter(x[:, 0], x[:, 1])
plt.xlabel("x0")
plt.ylabel("x1")
plt.show()
