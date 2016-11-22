from __future__ import division
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():

    tweet_df = utils.getTweetsData()
    price_df = utils.getPriceData()
    
    df = tweet_df.join(price_df) 
    df = df.dropna()
    for hour in range(1, 25):
        df['hour ' + str(hour)] = df['Close Price'].shift(-1 * hour) - df['Close Price'].shift(-1 * (hour - 1))
   
    df['ratio'] = utils.compareToDailyCycle(df)
    df = df.dropna()
    df['peaks'] = df['ratio'] > 1.5
    
    allcorrelation = []
    peakcorrelation = []
    for hour in range(1, 25):
        allcorrelation.append(np.corrcoef(df['hour ' + str(hour)], df['ratio'])[0,1])
        peakcorrelation.append(np.corrcoef(df['hour ' + str(hour)], df['peaks'])[0,1])
    
    allcorrelation = pd.DataFrame(allcorrelation)
    allcorrelation.columns = ['All Data']
    allcorrelation.index = range(1,len(allcorrelation)+1)
    
    peakcorrelation = pd.DataFrame(peakcorrelation) 
    peakcorrelation.columns = ['Peaks']
    peakcorrelation.index = range(1,len(peakcorrelation)+1)
    
    ax = allcorrelation.plot()
    peakcorrelation.plot(ax = ax)
    plt.show()

if __name__ == "__main__":
    main()
