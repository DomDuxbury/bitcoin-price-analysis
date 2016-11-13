import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils

def plotTweetDF(df):
    
    ax = df.plot(x = df.index, y = "total")
    
    df.plot(ax = ax, x = df.index, y = "peaks", 
            marker = "o", color = "red", markersize = 10)
    
    return ax

def main():
    
    tweet_df = utils.getTweetsData()
    price_df = utils.getPriceData()
    
    dailyCycle = tweet_df.groupby(tweet_df.index.hour).mean()
    dailyCycle.columns = ["mean tweets"]
    
    tweet_df["hour"] = tweet_df.index.hour
    tweet_df = tweet_df.join(dailyCycle, on = "hour", how="outer")
    tweet_df["ratio"] = tweet_df["total"] / tweet_df["mean tweets"]
    tweet_df["peaks"] = tweet_df.where(tweet_df.ratio > 1.7)["total"]

    ax = plotTweetDF(tweet_df) 
    price_df.plot(ax = ax, sharey = False, secondary_y = True)
    
    plt.show()

if __name__ == "__main__":
    main()

