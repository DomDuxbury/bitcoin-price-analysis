from __future__ import division
from multiprocessing import Pool
from functools import partial
import pandas as pd
import numpy as np
import pydotplus
import math

def cross_validate(model, features, labels, noSections):

    results = []

    # for section in range(0,noSections):

    #     sampled = split_train_test(features, labels, section, noSections)
    #     sampled = under_sample(split)
      
    #     fit_model = model.fit(sampled["train"]["data"], sampled["train"]["labels"])
    #     y_pred = fit_model.predict(sampled["test"]["data"])
        
    #     confusion_matrix = calc_confusion_matrix(sampled["test"]["labels"], y_pred)
    #     results.append(confusion_matrix)
    validate_split_partial = partial(validate_split, model, features, labels, noSections)
    p = Pool(8) 
    results = p.map(validate_split_partial, range(0, noSections))

    return np.array(results)

def validate_split(model, features, labels, noSections, section):
    
    sampled = split_train_test(features, labels, section, noSections)
    # sampled = under_sample(split)
    
    fit_model = model.fit(sampled["train"]["data"], sampled["train"]["labels"])
    y_pred = fit_model.predict(sampled["test"]["data"])
    
    return calc_confusion_matrix(sampled["test"]["labels"], y_pred)


def split_train_test(features, labels, section, noSections):

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

    data = {
        "train": {
            "data": train_data,
            "labels": train_labels
        },
        "test": {
            "data": test_data,
            "labels": test_labels
        }
    }

    return data 


def under_sample(data):

    tr_data = data["train"]["data"] 
    tr_labels = data["train"]["labels"] 
    ts_data = data["test"]["data"] 
    ts_label = data["test"]["labels"] 

    neg_examples = np.where(np.logical_not(tr_labels))[0]
    no_neg_examples = neg_examples.shape[0]

    pos_examples = np.where(tr_labels)[0][0:no_neg_examples]

    newTrainIndexes = np.concatenate((neg_examples, pos_examples), axis = 0) 

    data["train"]["data"] = tr_data[newTrainIndexes, :]
    data["train"]["labels"] = tr_labels[newTrainIndexes]
       
    return data


def calc_confusion_matrix(test_labels, y_pred):

    mistakes = (test_labels != y_pred).sum()
    correct = (test_labels == y_pred).sum()

    true_pos = np.logical_and(test_labels == y_pred, y_pred).sum()
    true_neg = correct - true_pos 

    false_pos = np.logical_and(test_labels != y_pred, y_pred).sum()
    false_neg = mistakes - false_pos 
    
    return [[true_pos, true_neg], [false_pos, false_neg]]



def report_results(confusion_matrices):

    accuracies = []

    for index, conf in enumerate(confusion_matrices):

        accuracy = (conf[0].sum()) / (conf[0].sum() + conf[1].sum())
        if (math.isnan(accuracy)):
            accuracy = 0

        accuracies.append(accuracy)

        print("Split %d:" % (index+1))
        print("\tTrue Positives: %d"  % (conf[0][0]))
        print("\tTrue Negatives: %d"  % (conf[0][1]))
        print("\tFalse Positives: %d"  % (conf[1][0]))
        print("\tFalse Negatives: %d"  % (conf[1][1]))
        print("\tAccuracy: %f"  % (accuracy))


    accuracies = np.array(accuracies)
    
    print("Overview:")
    print("\tMean Accuracy:\t %f" % accuracies.mean())
