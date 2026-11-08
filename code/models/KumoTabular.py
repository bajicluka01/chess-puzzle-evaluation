# https://scikit-learn.org/stable/developers/develop.html
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import pandas as pd
import numpy as np
import torch
import sdm

class KumoTabular(BaseEstimator):
    def __init__(self, task, quantile=None, device="cpu", random_state=None):
        self.task = task
        self.quantile = quantile
        self.device = device
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        df = X.copy()
        df["__target__"] = pd.Series(y).values
        self.dtype = df["__target__"].dtype

        self.stypes_ = sdm.infer_stypes(df, overrides={"__target__": "numerical" if self.task == "regression" else "categorical"})
        self.feature_stypes_ = {k: v for k, v in self.stypes_.items() if k != "__target__"}

        t = sdm.TableTensor.from_pandas(df, stypes=self.stypes_, device=self.device)
        self.x_context_ = t.drop_columns("__target__")
        self.y_context_ = t[:, "__target__"]
        self.model_ = sdm.models.KumoTabular(task=self.task, device=self.device)
        return self

    def predict(self, X: pd.DataFrame):
        q = sdm.TableTensor.from_pandas(X, stypes=self.feature_stypes_, device=self.device)

        if getattr(self, "generator", None) is None:
            if self.random_state is None:
                self.generator = None
            else:
                self.generator = torch.Generator(device=self.device).manual_seed(self.random_state)

        pred = self.model_(
            x_context=self.x_context_,
            y_context=self.y_context_,
            x_query=q,
            generator=self.generator
        )
        pred_df = pred.to_pandas()
        self.pred_columns_ = list(pred_df.columns)

        if self.quantile is None:
            return pred_df.to_numpy()
            #return pred_df
        return pred_df[self.quantile].to_numpy()
        #return pred_df[self.quantile]

class KumoRegressor(RegressorMixin, KumoTabular):
    task = "regression"

    def __init__(self, quantile="q500", device="cpu", random_state=None):
        self.quantile = quantile
        self.device = device
        self.random_state = random_state

class KumoClassifier(ClassifierMixin, KumoTabular):
    task = "classification"
    quantile = None

    def __init__(self, device="cpu", random_state=None):
        self.device = device
        self.random_state = random_state

    def fit(self, X, y):
        super().fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        proba = super().predict(X)
        cols = pd.Index(np.asarray(self.pred_columns_).astype(self.classes_.dtype))
        order = cols.get_indexer(self.classes_)
        return proba[:, order]

    def predict(self, X):
        proba = super().predict(X)
        return np.asarray(self.pred_columns_, dtype=self.dtype)[proba.argmax(axis=1)]
