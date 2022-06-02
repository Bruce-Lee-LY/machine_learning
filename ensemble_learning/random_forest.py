# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 23:37:41 on Mon, May 23, 2022
#
# Description: random forest

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
# from sklearn.externals import joblib
import joblib

MODEL_PATH = "model/random_forest/"
RANDOM_FOREST_IRIS_MODEL = "random_forest_iris.pkl"

OUTPUT_PATH = "output/random_forest/"
RANDOM_FOREST_IRIS_DOT = "random_forest_iris_{}.dot"
RANDOM_FOREST_IRIS_PNG = "random_forest_iris_{}.png"


def random_forest_iris():
    # Get dataset
    iris = load_iris()

    # Partition dataset
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=6)

    # Feature engineering: standardization
    # transfer = StandardScaler()
    # x_train = transfer.fit_transform(x_train)
    # x_test = transfer.transform(x_test)

    if not os.path.exists(MODEL_PATH + RANDOM_FOREST_IRIS_MODEL):
        # Build classifier
        classifier = RandomForestClassifier()

        # Model tuning
        grid_dict = {"n_estimators": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                     "criterion": ["gini", "entropy"]}
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
        joblib.dump(classifier, MODEL_PATH + RANDOM_FOREST_IRIS_MODEL)

        # Visualization: dot -Tpng decision_tree_iris.dot -o
        # decision_tree_iris.png
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        for i, estimator in enumerate(classifier.estimators_):
            export_graphviz(
                estimator,
                out_file=(
                    OUTPUT_PATH +
                    RANDOM_FOREST_IRIS_DOT).format(i),
                feature_names=iris.feature_names)
            os.system("dot -Tpng " +
                      (OUTPUT_PATH +
                       RANDOM_FOREST_IRIS_DOT).format(i) +
                      " -o " +
                      (OUTPUT_PATH +
                       RANDOM_FOREST_IRIS_PNG).format(i))
        os.system("sync")
    else:
        # Load model
        classifier = joblib.load(MODEL_PATH + RANDOM_FOREST_IRIS_MODEL)

    # Evaluate model
    y_predict = classifier.predict(x_test)
    print("predict:", y_predict)
    print("match:", y_predict == y_test)

    accuracy = classifier.score(x_test, y_test)
    print("accuracy:", accuracy)


def main():
    random_forest_iris()


if __name__ == "__main__":
    main()
