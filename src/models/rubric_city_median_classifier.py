from sklearn.base import ClassifierMixin
import numpy as np


class RubricCityMedianClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        '''
        self.median = y.groupby(by = [X['city'], X['modified_rubrics']]).median(numeric_only=True)
        return self

    def predict(self, X=None):
        length = len(X)
        y_pred = np.zeros(length)
        for i in range(length):
            x = X.iloc[i]
            y_pred[i] = self.median[x['city']][x['modified_rubrics']]
        return y_pred
