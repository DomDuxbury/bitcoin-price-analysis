from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getTweetsData():
    df = pd.read_csv("data/tweets-timestamp.csv", index_col=["hour"])
    df.set_index(pd.to_datetime(df.index), inplace = True)
    return df

def getPriceData():
    oct_df = pd.read_csv("data/coindesk-oct.csv", index_col=["Date"])
    nov_df = pd.read_csv("data/coindesk-nov.csv", index_col=["Date"])
    df = pd.concat([oct_df, nov_df])
    df.set_index(pd.to_datetime(df.index), inplace = True)
    return df

def compareToDailyCycle(df):
    
    dailyCycle = df.groupby(df.index.hour).mean()['total']
    
    df["hour"] = df.index.hour
    df = df.join(dailyCycle, on = "hour", how="outer", rsuffix="_mean")
    df = df.sort_index()
    
    return df["total"] / df["total_mean"]

def getLabelledTweets():
    df = pd.read_csv("data/sample/labelled.csv")
    return df

def getUnlabelledTweets():
    df = pd.read_csv("data/sample/unlabelled.csv")
    return df
