from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class RegressorEnsemble(RegressorMixin, BaseEstimator):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        preds = []
        for model in self.models:
            pred = model.predict(X)
            preds.append(pred)

        sum_pred = np.zeros_like(preds[0])
        for pred in preds:
            sum_pred += pred
        return sum_pred/len(self.models)