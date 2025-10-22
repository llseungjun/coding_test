from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def build_model(normalize: bool = True, fit_intercept: bool = True):
    steps = []
    if normalize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("lr", LinearRegression(fit_intercept=fit_intercept)))
    return Pipeline(steps)
