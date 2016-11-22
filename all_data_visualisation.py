import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils
from pandas.lib import Timestamp

def main():
    labelledTweets = utils.getAllTweetsAggregated() 
    
    fig, ax = plt.subplots()
    
    fields = ax.stackplot(
            labelledTweets.index, 
            labelledTweets["neg"], 
            labelledTweets["pos"], 
            labelledTweets["spam"], 
            colors = ["red", "green", "black"])

    colors = [field.get_facecolor()[0] for field in fields]
    patch1=mpl.patches.Patch(color=colors[0], label= 'Negative')
    patch2=mpl.patches.Patch(color=colors[1], label ='Positive')
    patch3=mpl.patches.Patch(color=colors[2], label ='Spam')
    plt.legend(handles=[patch1,patch2, patch3])
    plt.show()

if __name__ == "__main__":
    main()
