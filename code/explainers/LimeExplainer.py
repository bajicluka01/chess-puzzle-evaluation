import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
from data_preprocessing.plots import lime_plot

class LimeExplainer:
    def __init__(self, model, columns=None):
        # model should be trained already
        self.model = model
        # which columns are to be used for the explanation (must be present in X later!), default: all columns
        self.columns = columns 

    # for the default path to work, run from "code" folder
    def explain(self, X_train, target, X_test, verbose=True, mode="regression", plot=True, plot_location="../plots/explanations/lime", 
                    num_features=20):
        if self.columns is not None:
            X_train = X_train[self.columns]
            X_test = X_test[self.columns]
        else:
            self.columns = X_train.columns

        def predict_fn(arr):
            X = pd.DataFrame(arr, columns=self.columns)
            return np.asarray(self.model.predict(X)).ravel()

        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns.values.tolist(),
                                                            class_names=[target], verbose=verbose, mode=mode)
        
        for i in range(X_test.shape[0]):
            exp = explainer.explain_instance(X_test.values[i], predict_fn, num_features=num_features)
            exp_list = exp.as_list()
            print(exp_list)

            if plot:
                lime_plot(exp, i) #TODO location
