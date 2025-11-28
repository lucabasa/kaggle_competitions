import numpy as np

from sklearn.base import RegressorMixin, BaseEstimator


class DailyModel(BaseEstimator, RegressorMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X, y=None):
        preds = np.empty(len(X), dtype=float)
        for _, group in X.groupby("date_id"):
            idx = group.index
            preds[idx] = self.model.predict(group)
        return preds