from __future__ import division
import pandas as pd
import numpy as np
import nltk.classify.util
from nltk import ngrams
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier 

import os, sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(utils_path)
import utils.general as utils
import utils.sentiment as su

def main():
    unlabelledTweets = utils.getUnlabelledTweets() 
    unlabelledTweets["label"] = "n/a"
    
    spamLabelledTweets = utils.getSpamLabelledTweets() 
    posNegLabelledTweets = utils.getPosNeglabelledTweets()

    test_spam_classifier(spamLabelledTweets) 
    test_pos_neg_classifier(posNegLabelledTweets) 
    
    
def test_spam_classifier(labelledTweets):
    
    labelledTweets = labelledTweets.iloc[np.random.permutation(len(labelledTweets))]
    
    groups = labelledTweets.groupby("label")

    legitimate_tweets = groups.get_group("leg")
    spam_tweets = groups.get_group("spam")

    legfeats = su.extract_tweet_features(legitimate_tweets, 3)
    spamfeats = su.extract_tweet_features(spam_tweets, 3)
     
    legcutoff = int(len(legfeats)*3/4)
    spamcutoff = int(len(spamfeats)*3/4)

    trainfeats = legfeats[:legcutoff] + spamfeats[:spamcutoff]
    testfeats = legfeats[legcutoff:] + spamfeats[spamcutoff:]
    
    classifier = NaiveBayesClassifier.train(trainfeats) 
    
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
    classifier.show_most_informative_features()

def test_pos_neg_classifier(labelledTweets):

    labelledTweets = labelledTweets.iloc[np.random.permutation(len(labelledTweets))]
    
    groups = labelledTweets.groupby("label")

    pos_tweets = groups.get_group("pos")
    neg_tweets = groups.get_group("neg")

    posfeats = su.extract_tweet_features(pos_tweets, 2)
    negfeats = su.extract_tweet_features(neg_tweets, 2)
     
    poscutoff = int(len(posfeats)*3/4)
    negcutoff = int(len(negfeats)*3/4)

    trainfeats = posfeats[:poscutoff] + negfeats[:negcutoff]
    testfeats = posfeats[poscutoff:] + negfeats[negcutoff:]

    classifier = DecisionTreeClassifier.train(trainfeats) 
    
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
    print classifier.pretty_format()

    classifier = DecisionTreeClassifier.train(trainfeats + testfeats)
    print classifier.pretty_format()

def merge_two_dicts(x, y):
        '''Given two dicts, merge them into a new dict as a shallow copy.'''
        z = x.copy()
        z.update(y)
        return z

if __name__ == "__main__":
    main()
