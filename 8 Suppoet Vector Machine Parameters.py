# 1) OVR : ONE VS REST
# 2) OVO : ONE VS ONE


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm


df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

for a in [ 'linear', 'rbf', 'sigmoid', 'poly']:
    print(a)
    clf = svm.SVC(kernel=a,gamma='scale')
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    print(score)


# example = np.array([(2, 3, 2, 8, 4, 2, 7, 3, 2)])
# example = example.reshape(len(example), -1)
# predict = clf.predict(example)
#
# print(predict)