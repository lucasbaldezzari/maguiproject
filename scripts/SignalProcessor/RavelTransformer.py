from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class RavelTransformer(BaseEstimator, TransformerMixin):
    """Esta clase recibe datos en la forma [n_trials, n_channels or n_components, n_samples].
    Retorna datos en la forma [n_trials, n_components x n_samples]."""
    def __init__(self, method = "reshape"):

        self.method = method


    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):

        if self.method == "reshape":
            X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
        elif self.method == "mean":
            X = np.mean(X, axis = 1)

        return X
