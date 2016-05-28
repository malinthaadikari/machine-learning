from __future__ import print_function
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from scipy.stats import sem
import numpy as np


def mean_score(scores):
    return ("Mean score: {0:,.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))


# load iris dataset in to iris variable
iris = datasets.load_iris()
# load iris dataset features into x_iris
x_iris = iris.data
# load iris dataset classes in to y_iris
y_iris = iris.target
# Select only first two features from features and assign it to X and related targets to Y
X, Y = x_iris[:, :2], y_iris

# split dataset into train and test folds in 4:1 proportion
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=33)

# standardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# plotting each class in each iteration
colors = ['red', 'green', 'blue']
for i in xrange(len(colors)):
    xs = X_train[:, 0][Y_train == i]
    ys = X_train[:, 1][Y_train == i]
    plt.scatter(xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
# plt.show()

# crating Stochastic Gradient Descent model classifier and train it
clf = SGDClassifier()
clf.fit(X_train, Y_train)

print(clf.coef_)
print(clf.intercept_)

# predicting class for given data using the trained model
print(clf.predict(scaler.transform([[4.7, 3.1]])))

# Getting the accuracy of the model using training set
print(metrics.accuracy_score(Y_train, clf.predict(X_train)))

# Getting the accuracy of the model using test set
print(metrics.accuracy_score(Y_test, clf.predict(X_test)))

# Getting confusion matrix
print(metrics.confusion_matrix(Y_test, clf.predict(X_test)))

# Preparing pipeline providing the scaler and the classifier
clfCV = Pipeline([('scaler', StandardScaler()), ('linear_model', SGDClassifier())])

# defining K-Folds
cv = KFold(X.shape[0], 5, shuffle=True, random_state=33)

# getting cross validation score
scores = cross_val_score(clfCV, X, Y, cv=cv)

print(scores)

# calculating mean score
print(mean_score(scores))
