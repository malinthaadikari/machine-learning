from __future__ import print_function
import pandas as pand
from sklearn.tree import DecisionTreeClassifier
import os.path
import os.path
resource_path = os.path.join(os.path.split(__file__)[0], "resources")


def load_data():
    df = pand.read_csv(resource_path+"/datasets/titanic.csv");
    return df


def map_nominal_to_integers(df):
    df_refined = df.copy()
    sex_types = df_refined['Sex'].unique()
    cabin_types = df_refined['Cabin'].unique()
    embarked_types = df_refined["Embarked"].unique()
    sex_types_to_int = {name: n for n, name in enumerate(sex_types)}
    cabin_types_to_int = {name: n for n, name in enumerate(cabin_types)}
    embarked_types_to_int = {name: n for n, name in enumerate(embarked_types)}
    df_refined["Sex"] = df_refined["Sex"].replace(sex_types_to_int)
    df_refined["Cabin"] = df_refined["Cabin"].replace(cabin_types_to_int)
    df_refined["Embarked"] = df_refined["Embarked"].replace(embarked_types_to_int)
    return df_refined


def refine_data(df):
    df["Age"].fillna(0, inplace=True)
    return df


if __name__ == '__main__':
    print("Loading data from titanic.csv file")
    df = load_data()
    df.drop(['Name', 'Ticket', 'PassengerId'], inplace=True, axis=1)
    df = map_nominal_to_integers(df)
    df = refine_data(df)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
    X = df[features]
    Y = df["Survived"]
    Z = [1, 1, 22.0, 1, 0, 7.25, 0, 0]

    dt = DecisionTreeClassifier(min_samples_split=20, random_state=9)
    dt.fit(X, Y)
    print(dt.predict([Z]))
    