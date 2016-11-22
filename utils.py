from __future__ import division
import pandas as pd
import numpy as np

def getTweetsData():
    df = pd.read_csv("data/tweet-sum/tweets-timestamp.csv", index_col=["hour"])
    df.set_index(pd.to_datetime(df.index), inplace = True)
    return df

def getPriceData():
    oct_df = pd.read_csv("data/bitcoin/coindesk-oct.csv", index_col=["Date"])
    nov_df = pd.read_csv("data/bitcoin/coindesk-nov.csv", index_col=["Date"])
    df = pd.concat([oct_df, nov_df])
    df.set_index(pd.to_datetime(df.index), inplace = True)
    return df

def getSpamLabelledTweets():
    df = pd.read_csv("data/sample/spam-labelled.csv")
    return df

def getPosNeglabelledTweets():
    df = pd.read_csv("data/sample/labelled_non_spam_tweets.csv")
    return df

def getUnlabelledTweets():
    df = pd.read_csv("data/sample/unlabelled.csv")
    return df

def getAllTweets():
    df = pd.read_csv("data/all/21-11-2016.csv")
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df

def getAllTweetsAggregated():
    
    df = pd.read_csv("data/all/21-11-2016-labelled.csv")
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
    df = pd.read_csv("data/all/21-11-2016-labelled.csv")
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df

def compareToDailyCycle(df):
    
    dailyCycle = df.groupby(df.index.hour).mean()['total']
    
    df["hour"] = df.index.hour
    df = df.join(dailyCycle, on = "hour", how="outer", rsuffix="_mean")
    df = df.sort_index()
    
    return df["total"] / df["total_mean"]
