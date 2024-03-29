from sklearn.base import RegressorMixin
import numpy as np


class CityMeanRegressor(RegressorMixin):
    # Predicts the mean of y_train
    def __init__(self):
        self.city_mean_value = np.zeros(2)

    def fit(self, X=None, y=None):
        # X = np.array(X)
        y = np.array(y)
        # self.city_mean_value = np.zeros(2)
        self.city_mean_value[0] = y[X['city'] == 'msk'].mean()
        self.city_mean_value[1] = y[X['city'] == 'spb'].mean()

        return self

    def predict(self, X=None):
        # X = np.array(X)
        y_pred = np.zeros(len(X))
        y_pred[X['city'] == 'msk'] = self.city_mean_value[0]
        y_pred[X['city'] == 'spb'] = self.city_mean_value[1]
        return y_pred
