from __future__ import print_function
from sklearn.cluster import KMeans
import Utils

if __name__ == '__main__':
    df = Utils.get_dataframe("crime_data.csv")
    KMeans_model = KMeans(n_clusters=5, random_state=1)
    df.drop(['crime$cluster'], inplace=True, axis=1)
    df.rename(columns={df.columns[0]: 'State'}, inplace=True)
    print(df.head(), end="\n\n")
    numeric_columns = df._get_numeric_data()
    KMeans_model.fit(numeric_columns)
    labels = KMeans_model.labels_
    centroids = KMeans_model.cluster_centers_
    print(labels)
    print(KMeans_model.predict([[15, 236, 58, 21.2]]))
