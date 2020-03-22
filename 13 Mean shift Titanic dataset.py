import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn import preprocessing

df = pd.read_excel('titanic.xls')

original_data = pd.DataFrame.copy(df)

df.drop(['name', 'body'], 1, inplace=True)
df.fillna(0, inplace=True)


def handle_Non_numeric_data(df):
    columns = df.columns.values

    for column in columns:
        text_to_value = {}

        def convert_to_int(val):
            return text_to_value[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float:
            column_content = df[column].values.tolist()
            unique_column_content = set(column_content)

            x = 0
            for unique in unique_column_content:
                if unique not in text_to_value:
                    text_to_value[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))
            # df[column] = list(text_to_value[(df[column])])

    return df

handle_Non_numeric_data(df)

df.drop(['ticket', 'boat'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

label = clf.labels_
cluster_centers_ = clf.cluster_centers_

print(len(set(label)))


# new column[cluster group] to our original dataframe:
original_data['cluster_group'] = np.nan

for i in range(len(X)):
    # ith row of original data frame under columns of cluster group  = label[i]
    original_data['cluster_group'].iloc[i] = label[i]

#  total unique label or class type
n_clusters = len(np.unique(label))

survival_rates = {}



for i in range(n_clusters):

    # X= (original_data['cluster_group'] == float(i)) -> True, False;  orignal_data[X] = Take True condition
    temp_df = original_data[(original_data['cluster_group'] == float(i))]

    # X= (temp_df['survived'] == 1) -> TRUE, FALSE; temp_df[X] , take true condition
    survival_cluster = temp_df[(temp_df['survived'] == 1)]

    survival_rate = len(survival_cluster) / len(temp_df)
    # print(survival_rate)
    survival_rates[i] = survival_rate

print(survival_rates)
