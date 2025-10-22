import numpy as np


class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.W = None

    def fit(self, X, y):
        X = X.astype(float)
        y = y.reshape(-1, 1).astype(float)
        n, d = X.shape
        self.W = np.zeros((d, 1))
        for _ in range(self.epochs):
            pred = X @ self.W
            grad = (2 / n) * (X.T @ (pred - y))
            self.W -= self.lr * grad
        return self

    def predict(self, X):
        return (X @ self.W).ravel()


class LinearRegressionNE:
    def __init__(self, l2=0.0):
        self.l2 = l2
        self.W = None

    def fit(self, X, y):
        X = X.astype(float)
        y = y.reshape(-1, 1).astype(float)
        d = X.shape[1]
        I = np.eye(d)
        self.W = np.linalg.pinv(X.T @ X + self.l2 * I) @ X.T @ y
        return self

    def predict(self, X):
        return (X @ self.W).ravel()
