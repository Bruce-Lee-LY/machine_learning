# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 23:37:41 on Mon, May 23, 2022
#
# Description: SVM

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
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# from sklearn.externals import joblib
import joblib

MODEL_PATH = "model/SVM/"
SVM_IRIS_MODEL = "SVM_iris.pkl"


def SVM_iris():
    # Get dataset
    iris = load_iris()

    # Partition dataset
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=6)

    # Feature engineering: standardization
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    if not os.path.exists(MODEL_PATH + SVM_IRIS_MODEL):
        # Build classifier
        classifier = SVC()

        # Model tuning
        grid_dict = {"C": [0.1, 1.0, 10],
                     "kernel": ['rbf', 'linear', 'poly', 'sigmoid']}
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
        joblib.dump(classifier, MODEL_PATH + SVM_IRIS_MODEL)
    else:
        # Load model
        classifier = joblib.load(MODEL_PATH + SVM_IRIS_MODEL)

    # Evaluate model
    y_predict = classifier.predict(x_test)
    print("predict:", y_predict)
    print("match:", y_predict == y_test)

    accuracy = classifier.score(x_test, y_test)
    print("accuracy:", accuracy)


def main():
    SVM_iris()


if __name__ == "__main__":
    main()
