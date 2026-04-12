# basic libraries
import pandas as pd
import numpy as np

# train test 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# model libraries
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_test(X, y):

    # create 80:20 sets
    # probably order doesn't matter since rows are independent 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)


    # standardize

    # we do not standardize one-hot encoded ones / or like -1 1 ones
    dont_std = ["to_move"]
    X_train_dont_std = X_train[dont_std].copy()
    X_test_dont_std = X_test[dont_std].copy()

    X_train_to_std = X_train.drop(columns=dont_std)
    X_test_to_std = X_test.drop(columns=dont_std)

    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train_to_std)
    X_test_standardized = scaler.transform(X_test_to_std)

    # rejoin
    X_train_final = np.concatenate([X_train_standardized, X_train_dont_std.values], axis=1)
    X_test_final = np.concatenate([X_test_standardized, X_test_dont_std.values], axis=1)


    
    return X_train_final, X_test_final, y_train, y_test, scaler




class XGBoostModel():

    def __init__(self, lr=0.01, max_depth=6, subsample=1, reg_lambda=0.1, seed=13, num_round=150):
        
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

        self.model = xgb.train(
            params=params,
            dtrain=dtrain, 
            num_boost_round=self.num_round
        )

    def predict(self, X_val):
        dval = xgb.DMatrix(X_val)
        preds = self.model.predict(dval)
        preds = np.clip(preds, a_min=0, a_max=None)  # no negative rating 
        return preds




if __name__ == '__main__':

    # read csv file
    data = pd.read_csv("dataset_upgraded.csv")
    #print(data)


    # split attributes + target variables
    attribute_list = data.columns.to_list()

    extra_cols = ["epd", "rating", "rating_dev"]
    for col in extra_cols:
        attribute_list.remove(col)

    attribute_data = data[attribute_list]
    target_data = data["rating"]

    #print(attribute_list)
    #print(attribute_data)
    #print(target_data)


    # train test split 
    X_train, X_test, y_train, y_test, scaler = train_test(attribute_data, target_data)
    #print(X_train)
    #print(X_test)
    #print(y_train)
    #print(y_test)

    # model 
    model = XGBoostModel()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)


    # evaluate
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"MSE: {mse}, MAE: {mae}")
    print()

    feature_importance = model.model.get_score(importance_type='gain')
    #print(feature_importance)


    #"""
    idx = 0
    print(f"{'Real':>10} | {'Predicted':>10}")
    print("-" * 25)
    for pred, y in zip(preds, y_test):
        print(f"{float(y):10.2f} | {pred:10.2f}     {'!' if abs(pred - y) > 300 else ''}")  # ! ... critical 

        idx += 1
        if idx > 20:
            break     
    #"""
