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
