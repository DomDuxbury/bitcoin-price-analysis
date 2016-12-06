import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.spatial as sc

import os, sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(utils_path)
import utils.general as utils

def main():
    price = utils.getPriceData()
    labelledTweets = utils.getAllTweetsAggregated() 
    labelledTweets["total"] = labelledTweets["pos"] + labelledTweets["neg"] + labelledTweets["spam"]

    df = labelledTweets.join(price).dropna()
    
    for hour in range(1, 25):
        df['hour ' + str(hour)] = (df['Close Price'].shift(-1 * hour) - df['Close Price']) / df['Close Price']

    df = df.dropna()
    
    df["ratio"] = utils.compareToDailyCycle(df)
    df["pos_ratio"] = utils.compareToDailyCycle(df, "pos")
    df["neg_ratio"] = utils.compareToDailyCycle(df, "neg")
    df["spam_ratio"] = utils.compareToDailyCycle(df, "spam")
  
    linear_corr = lambda x,y : np.corrcoef(x,y)[0,1]
    correlation(df, linear_corr).plot(color = ["red", "green", "black", "blue"])
    correlation(df, utils.distcorr).plot()

    plt.ylabel("Correlation Coefficient")
    plt.show()
    
def correlation(df, func):
    neg, pos, spam, total = [], [], [], []
    for hour in range(1, 25):
        hourString = "hour " + str(hour)  
        neg.append(func(df[hourString], df['neg_ratio']))
        pos.append(func(df[hourString], df['pos_ratio']))
        spam.append(func(df[hourString], df['spam_ratio']))
        total.append(func(df[hourString], df['ratio']))
            
    return pd.DataFrame(dict(
        pos = pos,
        neg = neg,
        spam = spam,
        total = total))

if __name__ == "__main__":
    main()
