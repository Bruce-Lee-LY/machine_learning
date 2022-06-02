# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 23:37:41 on Mon, May 23, 2022
#
# Description: logistic regression

#!/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# from sklearn.externals import joblib
import joblib

MODEL_PATH = "model/logistic_regression/"
LOGISTIC_REGRESSION_TITANIC_MODEL = "logistic_regression_titanic.pkl"

TITANIC_TRAIN_DATA = "dataset/titanic/train.csv"


def logistic_regression_titanic():
    # Get dataset
    titanic = pd.read_csv(TITANIC_TRAIN_DATA)
    titanic.dropna(inplace=True)
    data = titanic[["Pclass", "Age", "Sex"]]
    data["Sex"][data["Sex"] == "female"] = 0
    data["Sex"][data["Sex"] == "male"] = 1
    target = titanic["Survived"]

    # Partition dataset
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, random_state=6)

    # Feature engineering: standardization
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    if not os.path.exists(MODEL_PATH + LOGISTIC_REGRESSION_TITANIC_MODEL):
        # Build classifier
        classifier = LogisticRegression()

        # Model tuning
        grid_dict = {"max_iter": [100, 500, 1000, 5000, 10000]}
        classifier = GridSearchCV(classifier, param_grid=grid_dict, cv=10)

        # Train model
        classifier.fit(x_train, y_train)
        print("best_params:", classifier.best_params_)
        print("best_score:", classifier.best_score_)
        print("best_estimator:", classifier.best_estimator_)
        print("cv_results:", classifier.cv_results_)

        # Save model
        classifier = classifier.best_estimator_
        print("coef:", classifier.coef_)
        print("intercept:", classifier.intercept_)
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        joblib.dump(classifier, MODEL_PATH + LOGISTIC_REGRESSION_TITANIC_MODEL)
    else:
        # Load model
        classifier = joblib.load(
            MODEL_PATH + LOGISTIC_REGRESSION_TITANIC_MODEL)

    # Evaluate model
    y_predict = classifier.predict(x_test)
    print("predict:", y_predict)
    print("match:", y_predict == y_test)

    accuracy = classifier.score(x_test, y_test)
    print("accuracy:", accuracy)


def main():
    logistic_regression_titanic()


if __name__ == "__main__":
    main()
