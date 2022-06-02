# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 23:37:41 on Mon, May 23, 2022
#
# Description: PCA

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
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
# from sklearn.externals import joblib
import joblib

MODEL_PATH = "model/PCA/"
PCA_BLOBS_MODEL = "PCA_blobs.pkl"


def PCA_blobs():
    # Get dataset
    data, target = make_blobs(
        n_samples=10000, n_features=6, centers=[
            [
                0, 0, 0], [
                1, 1, 1], [
                    2, 2, 2], [
                        3, 3, 3], [
                            4, 4, 4]], cluster_std=[
                                0.1, 0.2, 0.2, 0.2, 0.3], random_state=6)

    # Partition dataset
    # x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=6)

    # Feature engineering: standardization
    # transfer = StandardScaler()
    # x_train = transfer.fit_transform(x_train)
    # x_test = transfer.transform(x_test)

    if not os.path.exists(MODEL_PATH + PCA_BLOBS_MODEL):
        # Build pca
        pca = PCA()

        # Model tuning
        grid_dict = {"n_components": [0.6, 0.8, 0.9, 0.95, 0.99]}
        pca = GridSearchCV(pca, param_grid=grid_dict, cv=10)

        # Train model
        pca.fit(data)
        print("best_params:", pca.best_params_)
        print("best_score:", pca.best_score_)
        print("best_estimator:", pca.best_estimator_)
        print("cv_results:", pca.cv_results_)

        # Save model
        pca = pca.best_estimator_
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        joblib.dump(pca, MODEL_PATH + PCA_BLOBS_MODEL)
    else:
        # Load model
        pca = joblib.load(MODEL_PATH + PCA_BLOBS_MODEL)

    # Evaluate model
    print("n_components:", pca.n_components_)
    print("explained_variance:", pca.explained_variance_)
    print("explained_variance_ratio:", pca.explained_variance_ratio_)


def main():
    PCA_blobs()


if __name__ == "__main__":
    main()
