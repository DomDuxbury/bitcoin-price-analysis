from __future__ import division
import pandas as pd
import numpy as np
import sklearn as sl
from sklearn.ensemble import ExtraTreesClassifier

import os, sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(utils_path)

import utils.general as utils
import utils.model as model_utils

def main():

    df = prep_data()

    feature_columns = [
            "hour",  "recent_change",
            "ratio", "spam_ratio", "neg_ratio", "pos_ratio",
            "total", "pos", "neg", "spam",
            "transactions", "blocks", "fee", "output"]
    # feature_columns.reverse()

    features = df.as_matrix(columns = feature_columns)
    labels = df.as_matrix(columns = ["increase"]).flatten()

    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(features, labels)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(features.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, feature_columns[indices[f]], importances[indices[f]]))

def prep_data():

    price = utils.getPriceData()
    blocks = utils.getBlockData()
    labelledTweets = utils.getAllTweetsAggregated() 
    
    labelledTweets["total"] = labelledTweets["pos"] + labelledTweets["neg"] + labelledTweets["spam"]

    df = labelledTweets.join(price).join(blocks).dropna()
    
    df['return'] = (df['Close Price'].shift(-10) - df['Close Price']) / df['Close Price']
    df['recent_change'] = (df['Close Price'] - df['Close Price'].shift(1)) / df['Close Price']
    df['increase'] = df['return'] > 0

    df = df.dropna()
    
    df["ratio"] = utils.compareToDailyCycle(df)
    df["pos_ratio"] = utils.compareToDailyCycle(df, "pos")
    df["neg_ratio"] = utils.compareToDailyCycle(df, "neg")
    df["spam_ratio"] = utils.compareToDailyCycle(df, "spam")
    df["price_mod_10"] = df['Close Price'] % 10

    return df.dropna()

if __name__ == "__main__":
    main()
