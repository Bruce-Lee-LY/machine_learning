# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 23:37:41 on Mon, May 23, 2022
#
# Description: KNN

#!/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.externals import joblib
import joblib

MODEL_PATH = "model/KNN/"
KNN_IRIS_MODEL = "KNN_iris.pkl"


def KNN_iris():
    # Get dataset
    iris = load_iris()

    # Partition dataset
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=6)

    # Feature engineering: standardization
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    if not os.path.exists(MODEL_PATH + KNN_IRIS_MODEL):
        # Build classifier
        classifier = KNeighborsClassifier()

        # Model tuning
        grid_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]}
        classifier = GridSearchCV(classifier, param_grid=grid_dict, cv=10)

        # Train model
        classifier.fit(x_train, y_train)
        print("best_params:", classifier.best_params_)
        print("best_score:", classifier.best_score_)
        print("best_estimator:", classifier.best_estimator_)
        print("cv_results:", classifier.cv_results_)

        # Save model
        classifier = classifier.best_estimator_
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        joblib.dump(classifier, MODEL_PATH + KNN_IRIS_MODEL)
    else:
        # Load model
        classifier = joblib.load(MODEL_PATH + KNN_IRIS_MODEL)

    # Evaluate model
    y_predict = classifier.predict(x_test)
    print("predict:", y_predict)
    print("match:", y_predict == y_test)

    accuracy = classifier.score(x_test, y_test)
    print("accuracy:", accuracy)


def main():
    KNN_iris()


if __name__ == "__main__":
    main()
