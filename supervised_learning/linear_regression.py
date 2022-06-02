# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 23:37:41 on Mon, May 23, 2022
#
# Description: linear regression

#!/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
import sys
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
# from sklearn.externals import joblib
import joblib

MODEL_PATH = "model/linear_regression/"
LINEAR_REGRESSION_BOSTON_MODEL = "linear_regression_boston.pkl"
SGD_REGRESSION_BOSTON_MODEL = "SGD_regression_boston.pkl"
RIDGE_REGRESSION_BOSTON_MODEL = "ridge_regression_boston.pkl"


def linear_regression_boston():
    # Get dataset
    boston = load_boston()

    # Partition dataset
    x_train, x_test, y_train, y_test = train_test_split(
        boston.data, boston.target, random_state=6)

    # Feature engineering: standardization
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    if not os.path.exists(MODEL_PATH + LINEAR_REGRESSION_BOSTON_MODEL):
        # Build regressor
        regressor = LinearRegression()

        # Model tuning
        grid_dict = {"fit_intercept": [True, False]}
        regressor = GridSearchCV(regressor, param_grid=grid_dict, cv=10)

        # Train model
        regressor.fit(x_train, y_train)
        print(
            "[{}]: best_params: {}".format(
                sys._getframe().f_code.co_name,
                regressor.best_params_))
        print(
            "[{}]: best_score: {}".format(
                sys._getframe().f_code.co_name,
                regressor.best_score_))
        print(
            "[{}]: best_estimator: {}".format(
                sys._getframe().f_code.co_name,
                regressor.best_estimator_))
        print(
            "[{}]: cv_results: {}".format(
                sys._getframe().f_code.co_name,
                regressor.cv_results_))

        # Save model
        regressor = regressor.best_estimator_
        print(
            "[{}]: coef: {}".format(
                sys._getframe().f_code.co_name,
                regressor.coef_))
        print(
            "[{}]: intercept: {}".format(
                sys._getframe().f_code.co_name,
                regressor.intercept_))
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        joblib.dump(regressor, MODEL_PATH + LINEAR_REGRESSION_BOSTON_MODEL)
    else:
        # Load model
        regressor = joblib.load(MODEL_PATH + LINEAR_REGRESSION_BOSTON_MODEL)

    # Evaluate model
    y_predict = regressor.predict(x_test)
    print(
        "[{}]: predict: {}".format(
            sys._getframe().f_code.co_name,
            y_predict))

    error = mean_squared_error(y_predict, y_test)
    print("[{}]: error: {}".format(sys._getframe().f_code.co_name, error))


def SGD_regression_boston():
    # Get dataset
    boston = load_boston()

    # Partition dataset
    x_train, x_test, y_train, y_test = train_test_split(
        boston.data, boston.target, random_state=6)

    # Feature engineering: standardization
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    if not os.path.exists(MODEL_PATH + SGD_REGRESSION_BOSTON_MODEL):
        # Build regressor
        regressor = SGDRegressor()

        # Model tuning
        grid_dict = {"alpha": [0.0001, 0.001, 0.01, 0.1],
                     "max_iter": [500, 1000, 5000, 10000],
                     "eta0": [0.01, 0.05, 0.1, 0.5]}
        regressor = GridSearchCV(regressor, param_grid=grid_dict, cv=10)

        # Train model
        regressor.fit(x_train, y_train)
        print(
            "[{}]: best_params: {}".format(
                sys._getframe().f_code.co_name,
                regressor.best_params_))
        print(
            "[{}]: best_score: {}".format(
                sys._getframe().f_code.co_name,
                regressor.best_score_))
        print(
            "[{}]: best_estimator: {}".format(
                sys._getframe().f_code.co_name,
                regressor.best_estimator_))
        print(
            "[{}]: cv_results: {}".format(
                sys._getframe().f_code.co_name,
                regressor.cv_results_))

        # Save model
        regressor = regressor.best_estimator_
        print(
            "[{}]: coef: {}".format(
                sys._getframe().f_code.co_name,
                regressor.coef_))
        print(
            "[{}]: intercept: {}".format(
                sys._getframe().f_code.co_name,
                regressor.intercept_))
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        joblib.dump(regressor, MODEL_PATH + SGD_REGRESSION_BOSTON_MODEL)
    else:
        # Load model
        regressor = joblib.load(MODEL_PATH + SGD_REGRESSION_BOSTON_MODEL)

    # Evaluate model
    y_predict = regressor.predict(x_test)
    print(
        "[{}]: predict: {}".format(
            sys._getframe().f_code.co_name,
            y_predict))

    error = mean_squared_error(y_predict, y_test)
    print("[{}]: error: {}".format(sys._getframe().f_code.co_name, error))


def ridge_regression_boston():
    # Get dataset
    boston = load_boston()

    # Partition dataset
    x_train, x_test, y_train, y_test = train_test_split(
        boston.data, boston.target, random_state=6)

    # Feature engineering: standardization
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    if not os.path.exists(MODEL_PATH + RIDGE_REGRESSION_BOSTON_MODEL):
        # Build regressor
        regressor = Ridge()

        # Model tuning
        grid_dict = {"alpha": [0.5, 1.0, 5.0, 10.0, 20.0],
                     "max_iter": [500, 1000, 5000, 10000, 20000]}
        regressor = GridSearchCV(regressor, param_grid=grid_dict, cv=10)

        # Train model
        regressor.fit(x_train, y_train)
        print(
            "[{}]: best_params: {}".format(
                sys._getframe().f_code.co_name,
                regressor.best_params_))
        print(
            "[{}]: best_score: {}".format(
                sys._getframe().f_code.co_name,
                regressor.best_score_))
        print(
            "[{}]: best_estimator: {}".format(
                sys._getframe().f_code.co_name,
                regressor.best_estimator_))
        print(
            "[{}]: cv_results: {}".format(
                sys._getframe().f_code.co_name,
                regressor.cv_results_))

        # Save model
        regressor = regressor.best_estimator_
        print(
            "[{}]: coef: {}".format(
                sys._getframe().f_code.co_name,
                regressor.coef_))
        print(
            "[{}]: intercept: {}".format(
                sys._getframe().f_code.co_name,
                regressor.intercept_))
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        joblib.dump(regressor, MODEL_PATH + RIDGE_REGRESSION_BOSTON_MODEL)
    else:
        # Load model
        regressor = joblib.load(MODEL_PATH + RIDGE_REGRESSION_BOSTON_MODEL)

    # Evaluate model
    y_predict = regressor.predict(x_test)
    print(
        "[{}]: predict: {}".format(
            sys._getframe().f_code.co_name,
            y_predict))

    error = mean_squared_error(y_predict, y_test)
    print("[{}]: error: {}".format(sys._getframe().f_code.co_name, error))


def main():
    linear_regression_boston()
    SGD_regression_boston()
    ridge_regression_boston()


if __name__ == "__main__":
    main()
