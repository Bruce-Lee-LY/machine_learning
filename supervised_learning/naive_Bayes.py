# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 23:37:41 on Mon, May 23, 2022
#
# Description: naive Bayes

#!/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
# from sklearn.externals import joblib
import joblib

MODEL_PATH = "model/naive_Bayes/"
NAIVE_BAYES_NEWS_MODEL = "naive_Bayes_news.pkl"

DATASET_PATH = "dataset/"


def naive_Bayes_news():
    # Get dataset
    news = fetch_20newsgroups(data_home=DATASET_PATH, subset="all")

    # Partition dataset
    x_train, x_test, y_train, y_test = train_test_split(
        news.data, news.target, random_state=6)

    # Feature engineering: tf-idf
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    if not os.path.exists(MODEL_PATH + NAIVE_BAYES_NEWS_MODEL):
        # Build classifier
        classifier = MultinomialNB()

        # Model tuning
        grid_dict = {
            "alpha": [
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
                1.2,
                1.4,
                1.6,
                1.8,
                2.0]}
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
        joblib.dump(classifier, MODEL_PATH + NAIVE_BAYES_NEWS_MODEL)
    else:
        # Load model
        classifier = joblib.load(MODEL_PATH + NAIVE_BAYES_NEWS_MODEL)

    # Evaluate model
    y_predict = classifier.predict(x_test)
    print("predict:", y_predict)
    print("match:", y_predict == y_test)

    accuracy = classifier.score(x_test, y_test)
    print("accuracy:", accuracy)


def main():
    naive_Bayes_news()


if __name__ == "__main__":
    main()
