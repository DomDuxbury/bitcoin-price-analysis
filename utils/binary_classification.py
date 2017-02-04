from __future__ import division
from multiprocessing import Pool
from functools import partial
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
import pandas as pd
import numpy as np
import pydotplus
import math

def cross_validate(model, features, labels, noSections, should_sample = False):

    validate_split_partial = partial(validate_split, 
            model, features, labels, noSections, should_sample)
    
    p = Pool(noSections) 
    
    results = p.map(validate_split_partial, range(0, noSections))

    return np.array(results)

def validate_split(model, features, labels, noSections, should_sample, section):
    
    sampled = split_train_test(features, labels, section, noSections)

    if should_sample:
        sampled = sample(sampled)

    sampled = scale_data(sampled)
    
    fit_model = model.fit(sampled["train"]["data"], sampled["train"]["labels"])
    y_pred = fit_model.predict(sampled["test"]["data"])
    
    return calc_confusion_matrix(sampled["test"]["labels"], y_pred)


def split_train_test(features, labels, section, noSections, returns = []):

    if (len(returns) == 0):
        returns = np.zeros(features.shape[0]) 

    row_count = features.shape[0]
    section_size = int(row_count / noSections)

    start = section * section_size
    end = (section + 1) * section_size

    testIndexes = range(start, end)
    trainIndexes = [n for n in range(0, row_count) if n not in set(testIndexes)]

    train_data = features[trainIndexes,:]
    train_labels = labels[trainIndexes]

    test_data = features[testIndexes,:]
    test_labels = labels[testIndexes]
    test_returns = returns[testIndexes]

    data = {
        "train": {
            "data": train_data,
            "labels": train_labels
        },
        "test": {
            "data": test_data,
            "labels": test_labels,
            "returns": test_returns
        }
    }

    return data 

def scale_data(data):

    tr_data = data["train"]["data"] 
    ts_data = data["test"]["data"] 

    scaler = StandardScaler()
    scaler.fit(tr_data) 

    data["train"]["data"] = scaler.transform(tr_data)
    data["test"]["data"] = scaler.transform(ts_data)

    return data

def sample(data):

    tr_data = data["train"]["data"] 
    tr_labels = data["train"]["labels"] 

    ros = ADASYN()
    sampled_tr_data, sampled_tr_labels = ros.fit_sample(tr_data, tr_labels)
   
    data["train"]["data"] = sampled_tr_data 
    data["train"]["labels"] = sampled_tr_labels 

    return data


def calc_confusion_matrix(test_labels, y_pred):

    true_pos = np.logical_and(test_labels == y_pred, y_pred == True).sum()
    true_neg = np.logical_and(test_labels == y_pred, y_pred == False).sum() 

    false_pos = np.logical_and(test_labels != y_pred, y_pred == True).sum()
    false_neg = np.logical_and(test_labels != y_pred, y_pred == False).sum()
    
    return [[true_pos, true_neg], [false_pos, false_neg]]


def report_results(confusion_matrices):

    accuracies = []
    precisions = []
    recalls = []

    for index, conf in enumerate(confusion_matrices):

        accuracy = (conf[0].sum()) / (conf[0].sum() + conf[1].sum())
        precision = conf[0][0] / (conf[0][0] + conf[1][0])
        recall = conf[0][0] / (conf[0][0] + conf[1][1])

        if (math.isnan(accuracy)):
            accuracy = 0

        if (math.isnan(precision)):
            precision = 0
        
        if (math.isnan(recall)):
            recall = 0

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

        print("Split %d:" % (index+1))
        print("\tTrue Positives: %d"  % (conf[0][0]))
        print("\tTrue Negatives: %d"  % (conf[0][1]))
        print("\tFalse Positives: %d"  % (conf[1][0]))
        print("\tFalse Negatives: %d"  % (conf[1][1]))
        print("\tAccuracy: %f"  % (accuracy))
        print("\tPrecision: %f"  % (precision))
        print("\tRecall: %f"  % (recall))


    accuracies = np.array(accuracies)
    precisions = np.array(precision)
    recalls = np.array(recall)
    
    print("Overview:")
    print("\tMean Accuracy:\t %f" % accuracies.mean())
    print("\tMean Precision:\t %f" % precisions.mean())
    print("\tMean Recall:\t %f" % recalls.mean())

