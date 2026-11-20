import numpy as np
import xgboost as xgb

class XGBoostModel():
    def __init__(self, lr=0.01, max_depth=6, subsample=1, reg_lambda=0.01, seed=13, num_round=500):
        # paramters
        self.lr = lr
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.seed = seed 
        self.num_round = num_round
        self.model = None

    def fit(self, X_train, y_train):
        dtrain = xgb.DMatrix(X_train, label=y_train)

        params = {
            'objective': 'reg:squarederror',
            'learning_rate': self.lr,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'reg_lambda': self.reg_lambda,
            'seed': self.seed,
        }

        self.model = xgb.train(params=params, dtrain=dtrain, num_boost_round=self.num_round)

    def predict(self, X_val):
        dval = xgb.DMatrix(X_val)
        preds = self.model.predict(dval)
        preds = np.clip(preds, a_min=0, a_max=None)  # no negative rating 
        return preds
