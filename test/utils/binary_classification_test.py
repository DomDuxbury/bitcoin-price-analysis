import numpy as np
import numpy.testing as tst
import pytest

import os, sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(utils_path)

import utils.binary_classification as bc

# def test_answer():
    
#     test_array = np.array([[[0,1],[0,1]]])
#     mu.report_results(test_array)

#     assert 5 == 5

class TestSplitting:

    features = np.array([[1,2],[3,4],[5,6],[7,8]])
    labels = np.array([1,2,3,4])

    @pytest.mark.parametrize("section,expected", [
        (0, [1]),
        (1, [2]),
        (2, [3]),
        (3, [4])
    ])
    def test_test_labels(self, section, expected):

        split = bc.split_train_test(self.features, self.labels, section, 4)
        tst.assert_array_equal(expected, split["test"]["labels"])


    @pytest.mark.parametrize("section,expected", [
        (0, [[1,2]]),
        (1, [[3,4]]),
        (2, [[5,6]]),
        (3, [[7,8]])
    ])
    def test_test_data(self, section, expected):

        split = bc.split_train_test(self.features, self.labels, section, 4)
        tst.assert_array_equal(expected, split["test"]["data"])


    @pytest.mark.parametrize("section,expected", [
        (0, [2,3,4]),
        (1, [1,3,4]),
        (2, [1,2,4]),
        (3, [1,2,3])
    ])
    def test_train_labels(self, section, expected):

        split = bc.split_train_test(self.features, self.labels, section, 4)
        tst.assert_array_equal(expected, split["train"]["labels"])

    @pytest.mark.parametrize("section,expected", [
        (0, [[3,4],[5,6],[7,8]]),
        (1, [[1,2],[5,6],[7,8]]),
        (2, [[1,2],[3,4],[7,8]]),
        (3, [[1,2],[3,4],[5,6]])
    ])
    def test_train_data(self, section, expected):

        split = bc.split_train_test(self.features, self.labels, section, 4)
        tst.assert_array_equal(expected, split["train"]["data"])


class TestCalcConfusionMatrix:

    @pytest.mark.parametrize("test_labels, y_pred, expected", [
        (
            np.array([True, True, True]), 
            np.array([False, False, False]), 
            np.array([[0, 0], [0, 3]])
        ),
        (
            np.array([True, True, True]), 
            np.array([True, False, False]), 
            np.array([[1, 0], [0, 2]])
        ),
        (
            np.array([False, True, True]), 
            np.array([False, False, False]), 
            np.array([[0, 1], [0, 2]])
        ),
        (
            np.array([False, False, False]), 
            np.array([True, True, True]), 
            np.array([[0, 0], [3, 0]])
        ),
        (
            np.array([True, True, False, False]), 
            np.array([True, False, True, False]), 
            np.array([[1, 1], [1, 1]])
        ),
        (
            np.array([False]), 
            np.array([False]), 
            np.array([[0, 1], [0, 0]])
        )])
    def test_labels(self, test_labels, y_pred, expected):
        
        confusion_matrix = bc.calc_confusion_matrix(test_labels, y_pred)
        tst.assert_array_equal(expected, confusion_matrix)

class TestReporting:

    output_template = """Split %d:
                            True Positives: %d
                            True Negatives: %d
                            False Positives: %d
                            False Negatives: %d
                            Accuracy: %f
                            Precision: %f
                            Recall: %f
                        Overview:
                            Mean Accuracy: %f
                            Mean Precision: %f
                            Mean Recall: %f"""

    @pytest.mark.parametrize("matrices, split, tp, tf, fp, ff, acc, prec, rec, mean_acc, mean_prec, mean_rec", [
        (
            np.array([[[0, 0], [0, 0]]]),
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ),
        (
            np.array([[[1, 1], [1, 1]]]),
            1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        ),
        (
            np.array([[[0, 0], [1, 1]]]),
            1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0
        ),
        (
            np.array([[[1, 1], [0, 0]]]),
            1, 1, 1, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ),
        (
            np.array([[[3, 5], [1, 1]]]),
            1, 3, 5, 1, 1, 0.8, 0.75, 0.75, 0.8, 0.75, 0.75
        )])
    def test_basic(self, capsys, matrices, split, tp, tf, fp, ff, acc, prec, rec, mean_acc, mean_prec, mean_rec):

        bc.report_results(matrices)
        out, err = capsys.readouterr()
        expected = "".join((self.output_template % (split, tp, tf, fp, ff, acc, prec, rec, mean_acc, mean_prec, mean_rec)).split())

        assert "".join(out.split()) == expected

