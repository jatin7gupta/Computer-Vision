from sklearn import metrics
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
test_size = 0.25
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=test_size)


def knn():
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    predict = neigh.predict(X_test)
    accuracy = round(neigh.score(X_test, y_test), 3)
    recall = round(metrics.recall_score(y_test, predict, average="micro"), 3)
    return f'Accuracy: {accuracy}\t Recall: {recall}', confusion_matrix(y_test, predict)


def SGD():
    sgd = SGDClassifier()
    sgd.fit(X_train, y_train)
    predict = sgd.predict(X_test)
    accuracy = round(sgd.score(X_test, y_test), 3)
    recall = round(metrics.recall_score(y_test, predict, average="micro"), 3)
    return f'Accuracy: {accuracy}\t Recall: {recall}'


def DT():
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    predict = dt.predict(X_test)
    accuracy = round(dt.score(X_test, y_test), 3)
    recall = round(metrics.recall_score(y_test, predict, average="micro"), 3)
    return f'Accuracy: {accuracy}\t Recall: {recall}'


print('COMP9517 Week 5 Lab - z5240221\n')
print(f'Test size = {test_size}')
knn_result, confusion_mat = knn()
print(f'KNN {knn_result}')
print(f'SGD {SGD()}')
print(f'DT {DT()}')
print(f'\nKNN Confusion Matrix: \n{confusion_mat}')


