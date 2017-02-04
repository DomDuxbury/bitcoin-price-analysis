from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sl
import pydotplus
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

import os, sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(utils_path)

import utils.general as utils
import utils.binary_classification as model_utils

def main():

    df = prep_data()

    feature_columns = [
                "Close Price", "output", "fee", "pos_ratio"
            ]

    features = df.as_matrix(columns = feature_columns)
    labels = df.as_matrix(columns = ["increase"]).flatten()

    returns = df.as_matrix(columns = ["hourly_return"]).flatten()

    clf = svm.SVC(kernel = "poly", degree = 3)
    
    confusion_matrices = model_utils.cross_validate(clf, features, labels, 10, True)
    model_utils.report_results(confusion_matrices)

    def prep_data():

    price = utils.getPriceData()
    labelledTweets = utils.getAllTweetsAggregated() 
    blocks = utils.getBlockData()

    labelledTweets["total"] = labelledTweets["pos"] + labelledTweets["neg"] + labelledTweets["spam"]

    df = labelledTweets.join(price).join(blocks).dropna()
    
    df['return'] = (df['Close Price'].shift(-9) - df['Close Price']) / df['Close Price']

    df["hourly_return"] = ((df['Close Price'].shift(-1) - df['Close Price']) / df['Close Price']) + 1
    df['increase'] = df['return'] > 0

    df["ratio"] = utils.compareToDailyCycle(df)
    df["pos_ratio"] = utils.compareToDailyCycle(df, "pos")
    df["neg_ratio"] = utils.compareToDailyCycle(df, "neg")
    df["spam_ratio"] = utils.compareToDailyCycle(df, "spam")

    df['recent_change'] = (df['Close Price'] - df['Close Price'].shift(1)) / df['Close Price']
    
    return df.dropna()

if __name__ == "__main__":
    main()
