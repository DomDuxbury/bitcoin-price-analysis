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
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(utils_path)

import utils.general as utils
import utils.model as model_utils

def main():

    df = prep_data()

    feature_columns = [
                "Close Price", "pos_ratio", "ratio"
            ]

    features = df.as_matrix(columns = feature_columns)
    labels = df.as_matrix(columns = ["increase"]).flatten()

    returns = df.as_matrix(columns = ["hourly_return"]).flatten()

    clf = svm.SVC()
    
    # confusion_matrices = model_utils.cross_validate(clf, features, labels, 4)
    # model_utils.report_results(confusion_matrices)

    returns = model_utils.calculate_returns(clf, features, labels, 10, returns)
    profits = (returns * 3000) - 3000

    print returns
    print profits

    print "Weighted Returns"
    print returns[:,0].prod()
    print "Benchmark Returns"
    print returns[:,1].prod()
    print "Returns Before Fees"
    print returns[:,2].prod()

def prep_data():

    price = utils.getPriceData()
    labelledTweets = utils.getAllTweetsAggregated() 
    labelledTweets["total"] = labelledTweets["pos"] + labelledTweets["neg"] + labelledTweets["spam"]

    df = labelledTweets.join(price).dropna()
    
    df['return'] = (df['Close Price'].shift(-9) - df['Close Price']) / df['Close Price']

    df["hourly_return"] = ((df['Close Price'].shift(-1) - df['Close Price']) / df['Close Price']) + 1
    df['increase'] = df['return'] > 0.01

    df["ratio"] = utils.compareToDailyCycle(df)
    df["pos_ratio"] = utils.compareToDailyCycle(df, "pos")
    df["neg_ratio"] = utils.compareToDailyCycle(df, "neg")
    df["spam_ratio"] = utils.compareToDailyCycle(df, "spam")

    df["close_price_mod_10"] = np.floor(df["Close Price"] % 10)
    # df["recent_change"] = df["Close Price"] - df["Close Price"].shift(12)
    
    return df.dropna()

if __name__ == "__main__":
    main()
