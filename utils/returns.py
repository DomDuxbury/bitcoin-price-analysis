from __future__ import division
from multiprocessing import Pool
from functools import partial
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
import pandas as pd
import numpy as np
import pydotplus
import math

def calculate_returns(model, features, labels, noSections, returns):

    calc_returns_partial = partial(calculate_returns_section, 
            model, features, labels, noSections, returns)
    
    p = Pool(noSections) 
    
    results = p.map(calc_returns_partial, range(0, noSections))

def calculate_returns_section(model, features, labels, noSections, returns, section):
    
    split = split_train_test(features, labels, section, noSections, returns)

    split = scale_data(split)
    fit_model = model.fit(split["train"]["data"], split["train"]["labels"])
    y_pred = fit_model.predict(split["test"]["data"])

    test_returns = split["test"]["returns"]

    return_df = pd.DataFrame(test_returns, y_pred).reset_index()
    return_df.columns = ["pred", "return"]
    return_df["last_pred"] = return_df["pred"].shift(1)
    return_df = return_df.dropna()
    return_df["buy"] = (~return_df["last_pred"]) & return_df["pred"]
    return_df["sell"] = return_df["last_pred"] & (~return_df["pred"])
    return_df["trade"] = (return_df["sell"] | return_df["buy"]).astype(int)
    return_df["weighted_return"] = return_df["return"] - (return_df["trade"] * 0.0014)

    weighted_returns = return_df.as_matrix(columns = ["weighted_return"]).flatten()
    normal_returns = return_df.as_matrix(columns = ["return"]).flatten()

    return [np.prod(weighted_returns), np.prod(test_returns), np.prod(normal_returns)]  


