from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sl
from sklearn.naive_bayes import GaussianNB

import os, sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(utils_path)
import utils.general as utils

def main():
    price = utils.getPriceData()
    labelledTweets = utils.getAllTweetsAggregated() 
    labelledTweets["total"] = labelledTweets["pos"] + labelledTweets["neg"] + labelledTweets["spam"]

    df = labelledTweets.join(price).dropna()
    
    df['return'] = (df['Close Price'].shift(-1) - df['Close Price']) / df['Close Price']
    df['increase'] = df['return'] > 0

    df = df.dropna()
    
    df["ratio"] = utils.compareToDailyCycle(df)
    df["pos_ratio"] = utils.compareToDailyCycle(df, "pos")
    df["neg_ratio"] = utils.compareToDailyCycle(df, "neg")
    df["spam_ratio"] = utils.compareToDailyCycle(df, "spam")
    
    features = df.as_matrix(columns = ["hour", "pos"])
    labels = df.as_matrix(columns = ["increase"]).flatten()
    
    cross_validate(features, labels, 5)

def cross_validate(features, labels, noSections):
    row_count = features.shape[0]
    section_size = int(row_count / noSections)
    accuracies = np.zeros(noSections)

    for section in range(0,noSections):
        start = section * section_size
        end = (section + 1) * section_size

        testIndexes = range(start, end+1)
        trainIndexes = [n for n in range(0, row_count) if n not in set(testIndexes)]

        test = features[testIndexes,:]
        train = features[trainIndexes,:]

        train_labels = labels[trainIndexes]
        test_labels = labels[testIndexes]
        
        gnb = GaussianNB()
        y_pred = gnb.fit(train, train_labels).predict(test)

        mistakes = (test_labels != y_pred).sum()
        predictions = test.shape[0] 

        accuracy = (predictions - mistakes) / predictions
        accuracies[section] = accuracy
        print("Accuracy Split %d:\t %f" % (section + 1, accuracy))

    print("\nMean accuracy:\t %f" % accuracies.mean())

if __name__ == "__main__":
    main()
