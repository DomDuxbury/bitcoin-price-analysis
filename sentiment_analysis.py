from __future__ import division
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk.classify.util
from nltk import ngrams
from nltk.classify import NaiveBayesClassifier
 
def main():
    unlabelledTweets = utils.getUnlabelledTweets() 
    unlabelledTweets["label"] = "n/a"
    
    labelledTweets = utils.getLabelledTweets() 
    labelledTweets = labelledTweets.iloc[np.random.permutation(len(labelledTweets))]
    
    groups = labelledTweets.groupby("label")

    negative_tweets = groups.get_group("neg")
    positive_tweets = groups.get_group("pos")
    spam_tweets = groups.get_group("spam")

    negfeats = extract_tweet_features(negative_tweets)
    posfeats = extract_tweet_features(positive_tweets)
    spamfeats = extract_tweet_features(spam_tweets)
     
    negcutoff = int(len(negfeats)*3/4)
    poscutoff = int(len(posfeats)*3/4)
    spamcutoff = int(len(posfeats)*3/4)

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff] + spamfeats[:spamcutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:] + spamfeats[spamcutoff:]
    
    classifier = NaiveBayesClassifier.train(trainfeats) 
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
    classifier.show_most_informative_features()

    newlylabelleddf = classifyDataframe(classifier, unlabelledTweets)
    newlylabelleddf.to_csv("data/output/labelled_tweets.csv")
    
def extract_tweet_features(tweets):
    features = []

    for index, row in tweets.iterrows():
        base_features = {
                "user": row["user_id"],
                }
        ngram_features = extract_ngram_features(row["status"])
        row_features = merge_two_dicts(base_features, ngram_features)

        features.append((row_features, row["label"]))
        
    return features

def extract_ngram_features(text):
    n = 2
    features = {}
    for gram in ngrams(text.split(), n):
        features[gram] = True
    return features

def merge_two_dicts(x, y):
        '''Given two dicts, merge them into a new dict as a shallow copy.'''
        z = x.copy()
        z.update(y)
        return z

def classifyDataframe(classifier, df):
    features = []
    for index, row in df.iterrows():
        base_features = {
                "user": row["user_id"],
                }
        ngram_features = extract_ngram_features(row["status"])
        row_features = merge_two_dicts(base_features, ngram_features)
        features.append(row_features)
    labels = classifier.classify_many(features)
    df["label"] = labels
    return df

if __name__ == "__main__":
    main()
