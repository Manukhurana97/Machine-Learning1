# linear reqresion take the continous data and produce best fit line

import pandas as pd
import quandl
import math
import numpy as np

from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pickle


# google stock dataset
df = quandl.get('wiki/googl')


# 'Adj. Open', 'Adj. High', 'Adj.Low', 'Adj.Close', 'Adj.Volume' , only 4 columns
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]


#  df [high-low] percentage -> (high-low)/close*100
df['HL-percentage'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0


# daily percentage change -> (close-open) / open
df['percentage-change'] = ((df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']) * 100.0


# New dataframe
df = df[['Adj. Close', 'HL-percentage', 'percentage-change', 'Adj. Volume']]


forcast_col = 'Adj. Close'


# print null element
# print(df.isnull().sum())
# print(df.shape)
# print(df.dtypes)


#  fill NaN element with -99999
df.fillna(-99999)


# math.ceil -> Round up to next largest number.
forcast_out = int(math.ceil(0.01*len(df)))


df['label'] = df[forcast_col].shift(-forcast_out)


#  remove NaN
df.dropna(inplace=True)


X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])


# scaling the data to one unit
# X=preprocessing.scale(X)


# training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# Classifier
#  n_jobs -> no of threads  -1 means all the thread available
clf = LinearRegression()


# fitting training features and training labels
#  find patterns in data
clf.fit(X_train, y_train)


# saving the classifier
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)


#  read the classifier
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)


#predict accuracy
predict = clf.score(X_test, y_test)
print(predict)


# for k in ['linear','poly','rbf','sigmoid']:
#     clf = svm.SVR(kernel=k)
#     clf.fit(X_train, y_train)
#     accuracy = clf.score(X_test, y_test)
#     print(accuracy)
#
