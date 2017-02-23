import pandas as pd
import numpy as np
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier 
from sklearn.externals import joblib
import csv

import os, sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(utils_path)
import utils.general as utils
import utils.sentiment as su

def main():
    
    exportSpamModel()
    exportPosNegModel()

def exportSpamModel():
    spamLabelledTweets = utils.getSpamLabelledTweets()     
    full_spam_training_set = su.extract_tweet_features(spamLabelledTweets, 3) 
    spam_classifier = NaiveBayesClassifier.train(full_spam_training_set) 
    joblib.dump(spam_classifier, 'spam_classifier.pkl')

def exportPosNegModel():
    posNegLabelledTweets = utils.getPosNeglabelledTweets()
    full_pos_neg_training_set = su.extract_tweet_features(posNegLabelledTweets, 2) 
    pos_neg_classifier = DecisionTreeClassifier.train(full_pos_neg_training_set) 
    joblib.dump(pos_neg_classifier, 'pos_neg_classifier.pkl')

if __name__ == "__main__":
    main()
