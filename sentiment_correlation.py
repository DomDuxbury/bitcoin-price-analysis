import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils

def main():
    price = utils.getPriceData()
    labelledTweets = utils.getAllTweetsAggregated() 

    df = labelledTweets.join(price).dropna()
    
    for hour in range(1, 25):
        df['hour ' + str(hour)] = df['Close Price'].shift(-1 * hour) - df['Close Price']

    df = df.dropna()

    df["pos_ratio"] = utils.compareToDailyCycle(df, "pos")
    df["neg_ratio"] = utils.compareToDailyCycle(df, "neg")
    df["spam_ratio"] = utils.compareToDailyCycle(df, "spam")
  
    negcorrelation = []
    poscorrelation = []
    spamcorrelation = []
    peakcorrelation = []

    for hour in range(1, 25):
        negcorrelation.append(np.corrcoef(df['hour ' + str(hour)], df['neg_ratio'])[0,1])
        poscorrelation.append(np.corrcoef(df['hour ' + str(hour)], df['pos_ratio'])[0,1])
        spamcorrelation.append(np.corrcoef(df['hour ' + str(hour)], df['spam_ratio'])[0,1])
    
    pd.DataFrame(dict(
        pos = poscorrelation, 
        neg = negcorrelation, 
        spam = spamcorrelation)).plot()

    plt.show()
    
if __name__ == "__main__":
    main()
