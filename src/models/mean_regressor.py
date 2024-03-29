from scipy.stats import mode
from sklearn.base import RegressorMixin
import numpy as np

class MeanRegressor(RegressorMixin):
    # Predicts the mean of y_train
    def __init__(self):
        self.mean_value = 0
    def fit(self, X=None, y=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        '''
        self.mean_value = np.mean(y)
        return self

    def predict(self, X=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''
        return np.repeat(self.mean_value, len(X))
