import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def dummy_transform(data):
    """
    Replaces all categorical features in the dataset with dummies
    in a very naive way
    """
    transformed_data = np.zeros((data.shape[0], 23))
    transformed_data[:, 0] = data[:, 0]
    transformed_data[:, 1:4] = pd.get_dummies(data[:, 1]).values
    transformed_data[:, 4:15] = pd.get_dummies(data[:, 2]).values
    transformed_data[:, 15:19] = data[:, 3:7]
    transformed_data[:, 19:22] = pd.get_dummies(data[:, 7]).values
    transformed_data[:, 22:23] = data[:, 8:9]
    return transformed_data


class CustomClassifier(LogisticRegression):
    """
    Custom Logistic Regression implementation with preset coefficients
    """

    def __init__(self):

        super(CustomClassifier, self).__init__()
        self.coef_ = np.array([[
            -0.21404096, -2.33748762, -2.70184235, -2.73066579, -1.54875568,
            -2.06471249, -1.00291385, -1.43804488, -1.63824906, -1.0017922,
            -0.48928441, -0.62414559, -0.27359805,  0.98648587,  1.32501456,
            -0.80052908, -0.283991,  1.34641143,  0.02758078, -2.6099112,
            -1.63476157, -3.525323,  0.60030625
        ]])
        self.intercept_ = np.array([-7.76999576])
        self.classes_ = np.array([0, 1])

    def predict(self, X):
        assert len(X.shape) == 2, 'Invalid tensor shape'
        if X.shape[1] == 23:
            return super().predict(X)
        else:
            return super().predict(dummy_transform(X))

clf = CustomClassifier()
