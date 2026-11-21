import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
from utilities.plots import *

class ShapExplainer:
    def __init__(self, model, type_="agn", columns=None):
        # model should be trained already
        self.model = model
        # which columns are to be used for the explanation (must be present in X later!), default: all columns
        self.columns = columns 
        self.type = type_

    def do_shap_classifiers(self, model, X_test, y_test, y_pred, instances, scaler):
        model_name = type(model).__name__
        print(f"\nSHAP for {model_name}")

        classes = range(len(model.classes_))

        if scaler is not None:
            X_test = (pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index))

        if self.type == "tree":
            explainer = shap.TreeExplainer(model)

        elif self.type == "lin": 
            explainer = shap.LinearExplainer(model, X_test)

        elif self.type == "agn":
            # slow af, only a few samples will be collected        
            X_background = shap.sample(X_test, min(100, len(X_test)))
            explainer = shap.Explainer(model.predict_proba, X_background)
            
        shap_values = explainer(X_test)

        # we do shap for each of the classes
        for class_idx in classes:
            class_shap_values = shap_values[:, :, class_idx]

            beeswarm(class_shap_values, model_name, class_idx)

            for instance in instances:
                actual = y_test.iloc[instance] if hasattr(y_test, "iloc") else y_test[instance]
                predicted = y_pred[instance]

                sv = class_shap_values[instance]
                if np.asarray(sv.base_values).size == 1:
                    sv.base_values = np.asarray(sv.base_values).item()

                waterfall(sv, model_name, instance, actual, predicted, class_idx=class_idx, mode="classification")

    def do_shap_regressors(self, model, X_test, y_test, y_pred, instances, scaler):
        model_name = type(model).__name__
        print(f"\nSHAP for {model_name}")

        if scaler is not None:
            X_test = (pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index))

        if self.type == "tree":
            explainer = shap.TreeExplainer(model)
        elif self.type == "lin": 
            explainer = shap.LinearExplainer(model, X_test)
        elif self.type == "agn":
            # slow af, only a few samples will be collected        
            X_background = shap.sample(X_test, min(100, len(X_test)))
            explainer = shap.Explainer(model.predict, X_background)
        else:
            raise("evaluation.py: wrong shap type")

        shap_values = explainer(X_test)

        beeswarm(shap_values, model_name)

        for instance in instances:
            actual = y_test.iloc[instance] if hasattr(y_test, "iloc") else y_test[instance]
            predicted = y_pred[instance]

            sv = shap_values[instance]
            if np.asarray(sv.base_values).size == 1:
                sv.base_values = np.asarray(sv.base_values).item()

            waterfall(sv, model_name, instance, actual, predicted, mode="regression")

    # for the default path to work, run from "code" folder
    def explain(self, X_train, target, X_test, y_pred=None, instances=None, verbose=True, mode="regression", plot=True, plot_location="../plots/explanations/shap", 
                    num_features=20, scaler=None):
        # TODO adjust for shap
        if self.columns is not None:
            X_train = X_train[self.columns]
            X_test = X_test[self.columns]
        else:
            self.columns = X_train.columns

        def predict_fn(arr):
            X = pd.DataFrame(arr, columns=self.columns)
            return np.asarray(self.model.predict(X)).ravel()

        # TODO fix parameters
        if self.mode == "regression":
            self.do_shap_regressors(self.model, X_test, target, y_pred, instances, scaler)
        elif self.mode == "classification":
            self.do_shap_classifiers(self.model, X_test, target, y_pred, instances, scaler)
        else:
            raise "Unknown shap mode in ShapExplainer.py"
