import os, sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(utils_path)

from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
import utils.general as utils
import utils.binary_classification as model_utils

def main():
    # Build a classification task using 3 informative features
    df = prep_data()

    feature_columns = [
                "Close Price", "output", "fee", "pos_ratio", "total"
            ]
    
    # feature_columns = [
    #             "Close Price", "recent_change", 
    #             "output", "fee", "transactions", "reward",
    #             "pos_ratio", "total"
    #         ]


    features = df.as_matrix(columns = feature_columns)
    labels = df.as_matrix(columns = ["increase"]).flatten()

    scaler = StandardScaler()
    scaler.fit(features) 

    scaled_features = scaler.transform(features)

    ros = ADASYN()
    sampled_features, sampled_labels = ros.fit_sample(scaled_features, labels)

    clf = svm.SVC()

    param_grid = [
        {
            'C': [1, 10, 100, 250, 1000], 
            'gamma': [1, 0.25, 0.1, 0.01, 0.001], 
            'kernel': ['rbf']
        },
        {
            'C': [1, 10, 100, 250, 1000], 
            'gamma': [0.1, 0.01, 0.001],
            'kernel': ['poly'], 
            'decision_function_shape': ['ovr', 'ovo', None],
            'degree': [2, 3, 4, 5]
        }
    ] 
    
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs = 8, cv= 10)
    CV_rfc.fit(sampled_features, sampled_labels)
    print CV_rfc.best_params_

def prep_data():

    price = utils.getPriceData()
    labelledTweets = utils.getAllTweetsAggregated() 
    blocks = utils.getBlockData()

    labelledTweets["total"] = labelledTweets["pos"] + labelledTweets["neg"] + labelledTweets["spam"]

    df = labelledTweets.join(price).join(blocks).dropna()
    
    df['return'] = (df['Close Price'].shift(-9) - df['Close Price']) / df['Close Price']

    df["hourly_return"] = ((df['Close Price'].shift(-1) - df['Close Price']) / df['Close Price']) + 1
    df['increase'] = df['return'] > 0

    df["ratio"] = utils.compareToDailyCycle(df)
    df["pos_ratio"] = utils.compareToDailyCycle(df, "pos")
    df["neg_ratio"] = utils.compareToDailyCycle(df, "neg")
    df["spam_ratio"] = utils.compareToDailyCycle(df, "spam")

    df['recent_change'] = (df['Close Price'] - df['Close Price'].shift(1)) / df['Close Price']
    
    return df.dropna()

if __name__ == "__main__":
    main()
