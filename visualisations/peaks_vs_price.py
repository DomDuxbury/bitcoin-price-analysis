from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os, sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(utils_path)
import utils.general as utils


def plotTweetDF(df):
   
    # Plot total tweets
    ax = df.plot(x = df.index, y = "total", label = "Total Tweets")

    plt.ylabel("Total Tweets")
   
    # Plot the peaks
    df.plot(ax = ax, x = df.index, y = "peaks", linestyle="None", 
            marker = "*", color = "red", markersize = 15, label = "Detected Peaks")
   
    # Plot the price of bitcoin
    df.plot(ax = ax, x = df.index, y = "Close Price", 
            sharey = False, secondary_y = True)

    plt.ylabel("Price ($)")
    plt.title("Price of Bitcoin vs Total Bitcoin Tweets")

    return ax

def main():
   
    tweet_df = utils.getTweetsData()
    price_df = utils.getPriceData()
    
    # Create a daily cycle dataframe 
    dailyCycle = tweet_df.groupby(tweet_df.index.hour).mean()
    dailyCycle.columns = ["mean tweets"]
   
    # Create an hour column for each day to join on
    tweet_df["hour"] = tweet_df.index.hour
   
    # Join the daily cycle and create a ratio of each hour to the cycle
    tweet_df = tweet_df.join(dailyCycle, on = "hour", how="outer")
    tweet_df["ratio"] = tweet_df["total"] / tweet_df["mean tweets"]
   
    # Label peaks as hours where the ratio is high
    tweet_df["peaks"] = tweet_df.where(tweet_df.ratio > 1.5)["total"]

    # Join price to our data frame and plot it
    tweet_df = tweet_df.join(price_df) 
    ax = plotTweetDF(tweet_df)  
    plt.show()

if __name__ == "__main__":
    main()
