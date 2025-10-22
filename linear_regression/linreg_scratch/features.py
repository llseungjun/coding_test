import numpy as np


class StandardScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]
