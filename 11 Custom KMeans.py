import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np



style.use('ggplot')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [1, 3],
              [8, 9],
              [0, 3],
              [5, 4],
              [6, 4]])
colors = ['g.', 'b.', 'c.',  'r.', 'k.']


class K_Means:

    # tol : how much centroid mover, no of times program run
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        #  start with 2 centroid as k=2
        #     for this 1 and 2 points as taken as initial centroid
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []  # self.classification = keys/ centroids & features are the values


            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))

                self.classifications[classification].append(featureset)


            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
                optimized = True

                for c in self.centroids:
                    original_centroid = prev_centroids[c]
                    current_centroid = self.centroids[c]

                    if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                        # print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                        optimized = False

                if optimized:
                    break


    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", s=150, linewidths=5,color='k')

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x",  s=150, linewidths=5)


unknowns = np.array([[1, 3],
                    [8, 9],
                    [0, 3],
                    [5, 4],
                    [6, 4]])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker='*', color='r', s=150, linewidths=5)
# plt.plot