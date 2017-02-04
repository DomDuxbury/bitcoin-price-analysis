from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sl
import pydotplus
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier

import os, sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(utils_path)

import utils.general as utils
import utils.model as model_utils

def main():
    price = utils.getPriceData()
    labelledTweets = utils.getAllTweetsAggregated() 
    blocks = utils.getBlockData()

    labelledTweets["total"] = labelledTweets["pos"] + labelledTweets["neg"] + labelledTweets["spam"]

    df = labelledTweets.join(price).join(blocks).dropna()
    
    df['return'] = (df['Close Price'].shift(-1) - df['Close Price']) / df['Close Price']
    df['recent_change'] = (df['Close Price'] - df['Close Price'].shift(1)) / df['Close Price']
    df['increase'] = df['return'] > 0

    df = df.dropna()
    
    df["ratio"] = utils.compareToDailyCycle(df)
    df["pos_ratio"] = utils.compareToDailyCycle(df, "pos")
    df["neg_ratio"] = utils.compareToDailyCycle(df, "neg")
    df["spam_ratio"] = utils.compareToDailyCycle(df, "spam")
    for hour in range(1,13):
        df['pos_ratio_hour_minus_' + str(hour)] = df["pos_ratio"].shift(hour)
        # df['total_hour_minus_' + str(hour)] = df["pos_ratio"].shift(hour)
        # df['pos_ratio_hour_minus_' + str(hour)] = df["pos_ratio"].shift(hour)

    df = df.dropna() 
    # df = df.iloc[np.random.permutation(len(df))]

    feature_columns = ["Close Price", "output", "fee"]
    # feature_columns = ["Close Price", "pos_ratio"]

    features = df.as_matrix(columns = feature_columns)
    labels = df.as_matrix(columns = ["increase"]).flatten()

    # clf = MultinomialNB()
    # clf = tree.DecisionTreeClassifier()
    clf = RandomForestClassifier(n_estimators=101)
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(100, 300), random_state=1)
   
    confusion_matrices = []

    no_cycles = 1
    no_splits = 10

    for i in range(0,no_cycles):
        next_matrices = model_utils.cross_validate(clf, features, labels, no_splits)
        confusion_matrices.append(next_matrices)
    
    confusion_matrices = np.stack(np.array(confusion_matrices))
    confusion_matrices = confusion_matrices.reshape(no_splits * no_cycles,2,2)

    model_utils.report_results(confusion_matrices)

if __name__ == "__main__":
    main()
