# Clustering(UnSupervised Learning) :
# 1) flat
# 2) Hierarichal
# semi supervised supervised learning

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

style.use('ggplot')

X = np.array([[1, 0], [1, 2], [2, 2], [2, 0], [8, 10], [8, 8], [9, 8], [9, 10], [1, 10], [1,8], [2,10], [2,8]],)
# plt.scatter(X[:, 0], X[:, 1], s=150, linewidths=5)

#  KMeans alg
clf = KMeans(n_clusters=3)
clf.fit(X)

#  print centroid for cluster  using Kmeans
centroid = clf.cluster_centers_
print(centroid)

#  lables of feature X
labels = clf.labels_
print(labels)


color=['g.', 'b.', 'c.',  'r.', 'k.'] # . is period (marker)

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], color[labels[i]], markersize=25)

plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', s=100)
plt.show()

