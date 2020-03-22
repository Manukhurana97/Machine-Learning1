# Machine automatically detects the numbers of clusters
# It Take all the feature -center as a cluster center
# Its a hierarchical clustering algo


import numpy as np
from sklearn.cluster import MeanShift
import pandas as pd
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
from sklearn.datasets.samples_generator import make_blobs
style.use('ggplot')

#  initial centers
center = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]

X, _ = make_blobs(n_samples=1000, centers=center, cluster_std=1.5)

ms = MeanShift()
ms.fit(X)
lables = ms.labels_

#  final centers
cluster_center = ms.cluster_centers_

n_cluster = len(np.unique(lables))


color = 10*['r', 'g', 'c', 'k', 'y', 'm']
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=color[lables[i]], marker='o')
ax.scatter(cluster_center[:, 0], cluster_center[:, 1], cluster_center[:, 2], marker='X', color='k', zorder=10)
plt.show()

if len(center) == len(cluster_center):
    for i in range(len(center)):
        print(cluster_center[i]-center[i])
