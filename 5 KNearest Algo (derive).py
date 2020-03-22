import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import random
import pandas as pd

# plot1 =[1, 2]
# plot2 =[5, 5]
# #  harded coded euclidean formula
# euclidean_distance = sqrt((plot2[0]-plot1[0])**2 +(plot2[1]-plot1[1])**2)
# print(euclidean_distance)


style.use('fivethirtyeight')

dataset = {'K': [[1, 2], [2, 3], [3, 1], [2, 2]], 'r': [[6, 5], [7, 6], [8, 6], [7, 7]]}
new_feature = [4, 5]

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1], s=100)
plt.show()


#  calculate Knn
def K_nearest_neighnors(data, predict, k=3):

    if len(data) >= k:
        warnings.warn('less data to predict')

    distance = []

    for group in data:
        for feature in data[group]:

            # calculate euclidean distance
            euclidean_distance = np.linalg.norm(np.array(feature)-np.array(predict))  # formula

            # Add distance in list
            distance.append([euclidean_distance, group])

    #  get k smallest  euclidean distance
    votes = [i[1] for i in sorted(distance)[:k]]

    vote_result = Counter(votes).most_common(1)[0][0]

    confidence = Counter(votes).most_common(1)[0][1]
    return vote_result, confidence


result = K_nearest_neighnors(dataset, new_feature, k=3)
print(result)


# Cancer dataset
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)

#  convert data to float
full_data = df.astype(float).values.tolist()

test_size = 0.2

train_set = {2: [], 4: []}
test_set = {2: [], 4: []}

train_data = full_data[:-int(test_size*len(full_data))]  # (.2 * length of full data) 80% of train data
test_data = full_data[-int(test_size*len(full_data)):]  # last 20% of data


for i in train_data:  # populating the dictionaies
    train_set[i[-1]].append([i[:-1]])  # -ve 1 first element(last value) in the list
#   train_set[i[-1]] is to find type ,append data into list

for i in test_data:
    test_set[i[-1]].append([i[:-1]])  # -1 for last element of list, appending list into test_set  list till last element
#   test_set[i[-1]] is to find type ,append data into list

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        votes = K_nearest_neighnors(train_set, data, k=3)
        if group == votes:
            correct += 1
        # else:
        #     print(confidence) # confidence votes that are incorrect
        total += 1
print(' Accuracy', correct/total)
