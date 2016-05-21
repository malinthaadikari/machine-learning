from __future__ import print_function
import pandas as pand
import os.path
from sklearn.cluster import KMeans
resource_path = os.path.join(os.path.split(__file__)[0], "resources")


def load_data():
    df = pand.read_csv(resource_path+"/datasets/crime_data.csv")
    return df


if __name__ == '__main__':
    df = load_data()
    print(df.shape)
    kmeans_model = KMeans(n_clusters=2, random_state=1)
    df.drop(['crime$cluster'], inplace=True, axis=1)
    df.rename(columns={df.columns[0]: 'State'}, inplace=True)
    print(df.head(), end="\n\n")
    numeric_columns = df._get_numeric_data()
    kmeans_model.fit(numeric_columns)
    labals = kmeans_model.labels_
    centroids = kmeans_model.cluster_centers_
    print(labals)
    print(kmeans_model.predict([[15, 236, 58, 21.2]]))