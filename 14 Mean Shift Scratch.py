import matplotlib.pylab as plt
import numpy as np

X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [8, 2], [10, 3], [9, 3]])

color = 10*['r', 'g', 'b', 'c', 'b']


class MeanShift:
    def __init__(self, radius=20):
        self.radius = radius

    def fit(self, data):
        centroids = {}

        # initial centroid
        for i in range(len(data)):
            centroids[i] = data[i]  # key = value

#   create the initial centroid
        while True:

            new_centroids = []

            for i in centroids:

                in_bandwidth = []
                centroid = centroids[i]
                # print("centroid", centroid)

                for featureset in data:
                    # print('featureset', featureset)
                    # if euclidean distance < bandwidth
                    if np.linalg.norm(featureset-centroid) < self.radius:
                        in_bandwidth.append(featureset)

                new_centroid = np.average(in_bandwidth, axis=0)
                # print(new_centroid)

                new_centroids.append(tuple(new_centroid))  # converting array to tuple

            unique = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)  # copying the centroid dictionary or initial centroid (X)

            centroids = {}

            for i in range(len(unique)):
                centroids[i] = np.array(unique[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False

                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids

    def predict(selfself,data):
        pass


clf = MeanShift()
clf.fit(X)

centroid = clf.centroids

plt.scatter(X[:, 0], X[:, 1], s=150)

for c in centroid:
    plt.scatter(centroid[c][0], centroid[c][1], marker='*', s=150, color='k')
# plt.show()
