import pandas as pd
import numpy as np
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier 
import csv

import os, sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(utils_path)
import utils.general as utils
import utils.sentiment as su

def main():
    
    all_tweets = utils.getAllTweets()
    label_pipeline(all_tweets)

def label_pipeline(unlabelledTweets):
    
    unlabelledTweets["label"] = "n/a"
    spam_labelled_tweets = labelSpam(unlabelledTweets)
    
    groups = spam_labelled_tweets.groupby("label")

    spam = groups.get_group("spam")
    legitimate = groups.get_group("leg")

    legitimate = spam_labelled_tweets.loc[groups.groups["leg"]]
    legitimate["label"] = "n/a"

    spam = spam_labelled_tweets.loc[groups.groups["spam"]]

    pos_neg_labelled_tweets = labelPosNeg(legitimate)

    all_labelled = pos_neg_labelled_tweets.append(spam)
    all_labelled.index = all_labelled["id"]
    all_labelled.to_csv(utils.getAbsPath("data/output/all_labelled.csv"), index = False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)

def labelSpam(unlabelledTweets):
    spamLabelledTweets = utils.getSpamLabelledTweets()     
    full_spam_training_set = su.extract_tweet_features(spamLabelledTweets, 3) 
    spam_classifier = NaiveBayesClassifier.train(full_spam_training_set) 
    return su.classifyDataframe(spam_classifier, unlabelledTweets, 3)

def labelPosNeg(unlabelledTweets):
    posNegLabelledTweets = utils.getPosNeglabelledTweets()
    full_pos_neg_training_set = su.extract_tweet_features(posNegLabelledTweets, 2) 
    pos_neg_classifier = DecisionTreeClassifier.train(full_pos_neg_training_set) 
    return su.classifyDataframe(pos_neg_classifier, unlabelledTweets, 2)

if __name__ == "__main__":
    main()
