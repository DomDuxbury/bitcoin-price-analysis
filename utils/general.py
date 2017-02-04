from __future__ import division
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
import os, sys

def getAbsPath(data_path):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    return os.path.join(project_root, data_path)

def getTweetsData():
    df = pd.read_csv(getAbsPath("data/tweet-sum/tweets-timestamp.csv"), index_col=["hour"])
    df.set_index(pd.to_datetime(df.index), inplace = True)
    return df

def getBlockData():
    df = pd.read_csv(getAbsPath("data/block-chain/block_chain_hourly.csv"), index_col=["hour"])
    df.set_index(pd.to_datetime(df.index), inplace = True)
    return df

def getPriceData():
    oct_df = pd.read_csv(getAbsPath("data/bitcoin/coindesk-oct.csv"), index_col=["Date"])
    nov_df = pd.read_csv(getAbsPath("data/bitcoin/coindesk-nov.csv"), index_col=["Date"])
    dec_df = pd.read_csv(getAbsPath("data/bitcoin/coindesk-dec.csv"), index_col=["Date"])
    df = pd.concat([oct_df, nov_df, dec_df])
    df.set_index(pd.to_datetime(df.index), inplace = True)
    return df

def getSpamLabelledTweets():
    df = pd.read_csv(getAbsPath("data/sample/spam-labelled.csv"))
    return df

def getPosNeglabelledTweets():
    df = pd.read_csv(getAbsPath("data/sample/labelled_non_spam_tweets.csv"))
    return df

def getUnlabelledTweets():
    df = pd.read_csv(getAbsPath("data/sample/unlabelled.csv"))
    return df

def getAllTweets():
    df = pd.read_csv(getAbsPath("data/all/08-12-2016.csv"))
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df

def getAllTweetsAggregated():
   
    df = pd.read_csv(getAbsPath("data/all/08-12-2016-labelled.csv"))
    df["created_at"] = pd.to_datetime(df["created_at"])
    
    df["hour"] = df["created_at"].values.astype('<M8[h]')
    aggregate = df.groupby(["hour", "label"]).count()
    aggregate = aggregate["id"].reset_index()
    aggregate.index = aggregate["hour"]
    
    groups = aggregate.groupby("label")

    neg = groups.get_group("neg")["id"]
    pos = groups.get_group("pos")["id"]
    spam = groups.get_group("spam")["id"]
    
    return pd.DataFrame(dict(neg = neg, pos = pos, spam = spam))

def getAllTweetsLabelled():
    df = pd.read_csv(getAbsPath("data/all/08-12-2016-labelled.csv"))
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df

def compareToDailyCycle(df, column = "total"):
    
    dailyCycle = df.groupby(df.index.hour).mean()[column]
    
    df["hour"] = df.index.hour
    df = df.join(dailyCycle, on = "hour", how="outer", rsuffix="_mean")
    df = df.sort_index()
    
    return df[column] / df[column + "_mean"]

def distcorr(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor
