import numpy as np
import pandas as pd
from matplotlib import style
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing


df = pd.read_excel('titanic.xls')

df.drop(['name', 'body'], axis=1, inplace=True)



# pd.to_numeric(df)
df.fillna(0, inplace=True)


def _handle_non_numeric_data(df):
    columns = df.columns.values  # every columns

    for column in columns:
        text_digit_value = {}

        def convert_to_int(val):
            return text_digit_value[val]  # return value of the key

        if df[column].dtypes != np.int64 and df[column].dtypes != np.float64:

            # make list of the the non numeric data
            column_content = df[column].values.tolist()  # convert column to list

            unique_element = set(column_content)  # remove duplicate elemet from list

            x = 0
            for unique in unique_element:

                if unique not in text_digit_value:
                    text_digit_value[unique] = x  # unique id key and x is value
                    x += 1

            # resetting the value of column by mapping  the value of key of convert_to_int function
            df[column] = list(map(convert_to_int, df[column]))
    return df


df = _handle_non_numeric_data(df)


X = np.array(df.drop(['survived'], axis=1).astype(float))  # drop survived and conert rest of the data to float
X = preprocessing.scale(X)  # put the data in one scale
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct =0

for i in range(len(X)):
    predict = np.array(X[i].astype(float))
    predict = predict.reshape(-1, len(predict))
    prediction = clf.predict(predict)

#   check accuracy
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))

