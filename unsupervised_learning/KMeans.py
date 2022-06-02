# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 23:37:41 on Mon, May 23, 2022
#
# Description: KMeans

#!/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
# from sklearn.externals import joblib
import joblib

MODEL_PATH = "model/KMeans/"
KMEANS_BLOBS_MODEL = "KMeans_blobs.pkl"


def KMeans_blobs():
    # Get dataset
    data, target = make_blobs(
        n_samples=1000, n_features=2, centers=4, random_state=6)

    # Partition dataset
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, random_state=6)

    # Feature engineering: standardization
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    if not os.path.exists(MODEL_PATH + KMEANS_BLOBS_MODEL):
        # Build classifier
        classifier = KMeans()

        # Model tuning
        grid_dict = {"n_clusters": [2, 3, 4, 5, 6]}
        classifier = GridSearchCV(classifier, param_grid=grid_dict, cv=10)

        # Train model
        classifier.fit(x_train)
        print("best_params:", classifier.best_params_)
        print("best_score:", classifier.best_score_)
        print("best_estimator:", classifier.best_estimator_)
        print("cv_results:", classifier.cv_results_)

        # Save model
        classifier = classifier.best_estimator_
        print("cluster_centers:", classifier.cluster_centers_)
        print("inertia:", classifier.inertia_)
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        joblib.dump(classifier, MODEL_PATH + KMEANS_BLOBS_MODEL)
    else:
        # Load model
        classifier = joblib.load(MODEL_PATH + KMEANS_BLOBS_MODEL)

    # Evaluate model
    y_predict = classifier.predict(x_test)
    print("predict:", y_predict)
    print("match:", y_predict == y_test)

    contour_factor = silhouette_score(x_test, y_predict)
    print("contour_factor:", contour_factor)


def main():
    KMeans_blobs()


if __name__ == "__main__":
    main()
