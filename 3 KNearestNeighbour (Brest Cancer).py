import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# read data set
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

P = df.iloc[:].describe

empty = df.isnull().sum()
empty1 = df.isna().sum()


df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)

Xs = np.array(df.drop(['class'], axis=1))
ys = np.array(df['class'])

# test train data model
X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=.2)

clf = neighbors.KNeighborsClassifier(n_jobs=-1, n_neighbors=5)

clf.fit(X_train, y_train)
pickle.dump(clf, open('breastcancer2.pkl', 'wb'), protocol=2)

score = clf.score(X_test, y_test)
print(score)

predict_data = np.array([(2, 1, 2, 1, 2, 1, 3, 1, 1)])

predict_data = predict_data.reshape(len(predict_data), -1)
result = clf.predict(predict_data)

print(result)
