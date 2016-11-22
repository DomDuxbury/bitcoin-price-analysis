import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils
from pandas.lib import Timestamp

def main():
    labelledTweets = utils.getAllTweetsLabelled()
    labelledTweets["hour"] = labelledTweets["created_at"].values.astype('<M8[h]')
    aggregate = labelledTweets.groupby(["hour", "label"]).count()
    aggregate = aggregate["id"].reset_index()
    aggregate.index = aggregate["hour"]

    groups = aggregate.groupby("label")

    neg = groups.get_group("neg")["id"]
    pos = groups.get_group("pos")["id"]
    spam = groups.get_group("spam")["id"]

    fig, ax = plt.subplots()
    fields = ax.stackplot(neg.index, neg, pos, spam, colors = ["red", "green", "black"])
    colors = [field.get_facecolor()[0] for field in fields]
    patch1=mpl.patches.Patch(color=colors[0], label= 'Negative')
    patch2=mpl.patches.Patch(color=colors[1], label ='Positive')
    patch3=mpl.patches.Patch(color=colors[2], label ='Spam')
    plt.legend(handles=[patch1,patch2, patch3])
    plt.show()

    # ax = neg.plot(color = "red", label = "neg")
    # ax = pos.plot(color = "green", ax = ax, label = "pos")
    # spam.plot(color = "black", ax = ax, label = "spam")

    # plt.show()

def to_hour(ts):
    hour = 60*60*1000000000
    return Timestamp(long(round(ts.value, -5)))

if __name__ == "__main__":
    main()
