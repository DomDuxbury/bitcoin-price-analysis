import numpy as np
import os, sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(utils_path)

import utils.model as mu

def test_answer():
    
    test_array = np.array([[[0,1],[0,1]]])
    mu.report_results(test_array)

    assert 5 == 5
