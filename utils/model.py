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

def validate_split(model, features, labels, noSections, should_under_sample, section):
    
    sampled = split_train_test(features, labels, section, noSections)

    if should_sample:
        sampled = sample(sampled)

    sampled = scale_data(sampled)
    
    fit_model = model.fit(sampled["train"]["data"], sampled["train"]["labels"])
    y_pred = fit_model.predict(sampled["test"]["data"])
    
    return calc_confusion_matrix(sampled["test"]["labels"], y_pred)

def cross_validate_multi(model, features, labels, noSections, should_sample = False):

    validate_split_partial = partial(validate_split_multi, 
            model, features, labels, noSections, should_sample)
    
    p = Pool(noSections) 
    
    results = p.map(validate_split_partial, range(0, noSections))

    return np.array(results)

def validate_split_multi(model, features, labels, noSections, should_sample, section):
    
    sampled = split_train_test(features, labels, section, noSections)

    if should_sample:
        sampled = sample(sampled)

    sampled = scale_data(sampled)
    
    fit_model = model.fit(sampled["train"]["data"], sampled["train"]["labels"])
    y_pred = fit_model.predict(sampled["test"]["data"])
    
    return calc_confusion_matrix_multi(sampled["test"]["labels"], y_pred)


def calculate_returns(model, features, labels, noSections, returns):

    calc_returns_partial = partial(calculate_returns_section, 
            model, features, labels, noSections, returns)
    
    p = Pool(noSections) 
    
    results = p.map(calc_returns_partial, range(0, noSections))

def calculate_returns_multi(model, features, labels, noSections, returns):

    calc_returns_partial = partial(calculate_returns_section_multi, 
            model, features, labels, noSections, returns)
    
    p = Pool(noSections) 
    
    results = p.map(calc_returns_partial, range(0, noSections))

    return np.array(results)

def calculate_returns_section(model, features, labels, noSections, returns, section):
    
    split = split_train_test(features, labels, section, noSections, returns)

    split = scale_data(split)
    fit_model = model.fit(split["train"]["data"], split["train"]["labels"])
    y_pred = fit_model.predict(split["test"]["data"])

    test_returns = split["test"]["returns"]

    return_df = pd.DataFrame(test_returns, y_pred).reset_index()
    return_df.columns = ["pred", "return"]
    return_df["last_pred"] = return_df["pred"].shift(1)
    return_df = return_df.dropna()
    return_df["buy"] = (~return_df["last_pred"]) & return_df["pred"]
    return_df["sell"] = return_df["last_pred"] & (~return_df["pred"])
    return_df["trade"] = (return_df["sell"] | return_df["buy"]).astype(int)
    return_df["weighted_return"] = return_df["return"] - (return_df["trade"] * 0.0014)

    weighted_returns = return_df.as_matrix(columns = ["weighted_return"]).flatten()
    normal_returns = return_df.as_matrix(columns = ["return"]).flatten()

    return [np.prod(weighted_returns), np.prod(test_returns), np.prod(normal_returns)]  

def calculate_returns_section_multi(model, features, labels, noSections, returns, section):
    
    split = split_train_test(features, labels, section, noSections, returns)

    split = scale_data(split)
    fit_model = model.fit(split["train"]["data"], split["train"]["labels"])
    y_pred = fit_model.predict(split["test"]["data"])

    test_returns = split["test"]["returns"]

    return_df = pd.DataFrame(test_returns, y_pred).reset_index()
    return_df.columns = ["pred", "return"]

    holding = False
    weighted_returns = []
    trading_fees = 0
    for index, row in return_df.iterrows():
        if not holding and row["pred"] == "buy":
            weighted_returns.append(row["return"] - trading_fees)
            holding = True
        elif not holding:
            weighted_returns.append(1)
        elif holding and (row["pred"] == "buy" or row["pred"] == "hold"):
            weighted_returns.append(row["return"])
        elif holding and row["pred"] == "sell":
            weighted_returns.append(1 - trading_fees)
            holding = False

    weighted_returns = np.array(weighted_returns)
    return [np.prod(weighted_returns), np.prod(test_returns)]


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

    mistakes = (test_labels != y_pred).sum()
    correct = (test_labels == y_pred).sum()

    true_pos = np.logical_and(test_labels == y_pred, y_pred).sum()
    true_neg = correct - true_pos 

    false_pos = np.logical_and(test_labels != y_pred, y_pred).sum()
    false_neg = mistakes - false_pos 
    
    return [[true_pos, true_neg], [false_pos, false_neg]]

def calc_confusion_matrix_multi(test_labels, y_pred):

    true_buy = np.logical_and(test_labels == y_pred, y_pred == 'buy').sum()
    true_sell = np.logical_and(test_labels == y_pred, y_pred == 'sell').sum() 

    false_buy = np.logical_and(test_labels != y_pred, y_pred == 'buy').sum()
    false_sell = np.logical_and(test_labels != y_pred, y_pred == 'sell').sum()
    
    return [[true_buy, true_sell], [false_buy, false_sell]]


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
            pos_accuracy = 0
        
        if (math.isnan(recall)):
            pos_accuracy = 0

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


def report_results_multi(confusion_matrices):

    accuracies = []

    for index, conf in enumerate(confusion_matrices):

        accuracy = (conf[0].sum()) / (conf[0].sum() + conf[1].sum())

        if (math.isnan(accuracy)):
            accuracy = 0

        accuracies.append(accuracy)

        print("Split %d:" % (index+1))

        print("\tTrue Buy: %d"  % (conf[0][0]))
        print("\tTrue Hold: %d"  % (conf[0][1]))
        print("\tTrue Sell: %d"  % (conf[0][2]))

        print("\tFalse Buy: %d"  % (conf[1][0]))
        print("\tFalse Hold: %d"  % (conf[1][1]))
        print("\tFalse Sell: %d"  % (conf[1][2]))
        print("\tAccuracy: %f"  % (accuracy))


    accuracies = np.array(accuracies)
    
    print("Overview:")
    print("\tMean Accuracy:\t %f" % accuracies.mean())
