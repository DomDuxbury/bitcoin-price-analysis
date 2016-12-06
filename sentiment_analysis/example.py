from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

import os, sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(utils_path)
import utils.general as utils

def word_feats(words):
    return dict([(word, True) for word in words])
 
def main():
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')
     
    first_review = movie_reviews.words(fileids=(negids[0]))
    
    negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
     
    negcutoff = int(len(negfeats)*3/4)
    poscutoff = int(len(posfeats)*3/4)
    
    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
     
    classifier = NaiveBayesClassifier.train(trainfeats)
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
    classifier.show_most_informative_features()

    result = classifier.classify({
        "bad": True,
        "movie": True,
        "magnificent": True
        })

    print(result)

if __name__ == "__main__":
    main()
