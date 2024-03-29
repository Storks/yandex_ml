from scipy.stats import mode
from sklearn.base import ClassifierMixin
import numpy as np


def most_frequent(nums):
    """
    Find the most frequent value in an array
    :param nums: array of ints
    :return: the most frequent value
    """
    unique, counts = np.unique(nums, return_counts=True)
    max_count_index = np.argmax(counts)
    return unique[max_count_index]


class MostFrequentClassifier(ClassifierMixin):
    # Predicts the rounded (just in case) median of y_train
    def fit(self, X=None, y=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        '''
        self.most_freq = most_frequent(y)

    def predict(self, X=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''
        return np.repeat(self.most_freq, len(X))
