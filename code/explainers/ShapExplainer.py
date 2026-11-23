import shap
import pandas as pd
import numpy as np

from utilities.plots import *


SHAP_TYPES = {
    # Linear models
    "LinearRegression": "lin",
    "Ridge": "lin",
    "BayesianRidge": "lin",
    "LogisticRegression": "lin",
    "SGDRegressor": "lin",
    "SGDClassifier": "lin",

    # Tree-based models
    "DecisionTreeRegressor": "tree",
    "RandomForestClassifier": "tree",
    "GradientBoostingClassifier": "tree",
    "XGBRegressor": "tree",
    "XGBClassifier": "tree",
    "XGBRanker": "tree",
    "LGBMRegressor": "tree",
    "CatBoostRegressor": "tree",

    # Model-agnostic
    "KumoRegressor": "agn",
    "KumoClassifier": "agn",
    "TabFMRegressor": "agn",
    "TabFMClassifier": "agn",
    "RegressorEnsemble": "agn",
    "StackingRegressor": "agn",
    "VotingRegressor": "agn",
    "VotingClassifier": "agn",
    "MLPRegressor": "agn",
    "MLPClassifier": "agn",
    "SVC": "agn",
    "GaussianNB": "agn",
    "KNeighborsClassifier": "agn",
}


class ShapExplainer:
    def __init__(self, model, columns=None, type_=None):
        self.model = model

        # columns to display in plots 
        self.columns = columns 

        # based on the model name, automatically set type 
        # (unless given -> given only if you want to do comparison)
        self.type = (
            type_ if type_ is not None
            else SHAP_TYPES.get(type(model).__name__, "agn")
        )

    def do_shap_classifiers(
            self,
            model,
            X_train, X_test, y_test, y_pred,
            instances,
            epd_test,
            scaler,
            show_plot, save_plot,
            plot_dir
        ):

        model_name = type(model).__name__
        print(f"\nSHAP for {model_name}")

        if scaler is not None:
            X_train = pd.DataFrame(
                scaler.transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

        if self.type == "tree":
            explainer = shap.TreeExplainer(model)

        elif self.type == "lin": 
            explainer = shap.LinearExplainer(model, X_train)

        elif self.type == "agn":
            # slow af, only a few samples will be collected        
            X_background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
            explainer = shap.Explainer(
                model.predict_proba,
                X_background
            )
        else:
            raise ValueError("Unknown SHAP type")

        shap_values = explainer(X_test)

        # handle some different formats of shap_values 
        if len(shap_values.shape) == 2:
            # Binary classifier with one explained output
            if len(model.classes_) != 2:
                raise ValueError(
                    f"Expected binary classifier, got {len(model.classes_)} classes"
                )

            class_explanations = [(model.classes_[1], shap_values)]
            
        elif len(shap_values.shape) == 3:
            class_explanations = [
                (class_label, shap_values[:, :, class_idx])
                for class_idx, class_label in enumerate(model.classes_)
            ]

        else:
            raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

        # we do shap for each of the classes
        for class_label, class_shap_values in class_explanations:

            beeswarm(
                class_shap_values=class_shap_values,
                model_name=model_name,
                class_idx=class_label,
                columns=self.columns,
                show_plot=show_plot,
                save_plot=save_plot,
                plot_dir=plot_dir
            )
            bar(
                class_shap_values=class_shap_values,
                model_name=model_name,
                columns=self.columns,
                show_plot=show_plot,
                save_plot=save_plot,
                plot_dir=plot_dir,
                instance=None,
                class_idx=class_label
            )

            for instance in instances:
                actual = y_test.iloc[instance] if hasattr(y_test, "iloc") else y_test[instance]
                predicted = y_pred[instance]

                # get epd if available
                epd = epd_test.iloc[instance] if epd_test is not None else None

                sv = class_shap_values[instance]

                waterfall(
                    sv=sv,
                    model_name=model_name,
                    instance=instance,
                    actual=actual,
                    predicted=predicted,
                    epd=epd,
                    class_idx=class_label,
                    columns=self.columns,
                    mode="classification",
                    show_plot=show_plot,
                    save_plot=save_plot,
                    plot_dir=plot_dir
                )
                bar(
                    class_shap_values=sv,
                    model_name=model_name,
                    instance=instance,
                    class_idx=class_label,
                    columns=self.columns,
                    epd=epd,
                    show_plot=show_plot,
                    save_plot=save_plot,
                    plot_dir=plot_dir
                )

    def do_shap_regressors(
            self,
            model,
            X_train, X_test, y_test, y_pred,
            instances,
            epd_test,
            scaler,
            show_plot, save_plot,
            plot_dir
        ):

        model_name = type(model).__name__
        print(f"\nSHAP for {model_name}")

        if scaler is not None:
            X_train = pd.DataFrame(
                scaler.transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

        if self.type == "tree":
            explainer = shap.TreeExplainer(model)
        elif self.type == "lin": 
            explainer = shap.LinearExplainer(model, X_train)
        elif self.type == "agn":
            # slow af, only a few samples will be collected        
            X_background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
            explainer = shap.Explainer(model.predict, X_background)
        else:
            raise ValueError("Unknown SHAP type")

        shap_values = explainer(X_test)

        beeswarm(
            class_shap_values=shap_values,
            model_name=model_name,
            columns=self.columns,
            show_plot=show_plot,
            save_plot=save_plot,
            plot_dir=plot_dir
        )
        bar(
            class_shap_values=shap_values,
            model_name=model_name,
            columns=self.columns,
            show_plot=show_plot,
            save_plot=save_plot,
            plot_dir=plot_dir,
            instance=None
        )

        for instance in instances:
            actual = y_test.iloc[instance] if hasattr(y_test, "iloc") else y_test[instance]
            predicted = y_pred[instance]

            # get epd if available
            epd = epd_test.iloc[instance] if epd_test is not None else None

            sv = shap_values[instance]
            if np.asarray(sv.base_values).size == 1:
                sv.base_values = np.asarray(sv.base_values).item()

            waterfall(
                sv=sv,
                model_name=model_name,
                instance=instance,
                actual=actual,
                predicted=predicted,
                epd=epd,
                mode="regression",
                columns=self.columns,
                show_plot=show_plot,
                save_plot=save_plot,
                plot_dir=plot_dir
            )

            bar(
                class_shap_values=sv,
                model_name=model_name,
                instance=instance,
                columns=self.columns,
                epd=epd,
                show_plot=show_plot,
                save_plot=save_plot,
                plot_dir=plot_dir
            )


    # for the default path to work, run from "code" folder
    def explain(
            self,
            X_train, X_test, y_test, y_pred,
            mode="regression",
            instances=None,
            epd_test=None,
            scaler=None,
            show_plot=False,
            save_plot=False,
            plot_dir=None
        ):
        """
            X_train         : for background data 
            X_test          : puzzles we want to explain globally 
            y_test, y_pred  : only for plotting, to see predicted and actual class
            instances       : puzzles we want to explain locally 
            scaler          : if model was trained on standardized data, it should 
                              also be explained on them 
                              TODO: perhaps send already standardized data
            show_plot       : if True, it shows plot
            save_plot       : if True, it stores plot 
        """

        if instances is None:
            instances = []

        # choose which columns should be plotted 
        if self.columns is None:
            self.columns = X_train.columns
        
        if mode == "regressor":
            self.do_shap_regressors(
                self.model,
                X_train, X_test, y_test, y_pred,
                instances,
                epd_test,
                scaler,
                show_plot, save_plot,
                plot_dir
            )
        elif mode == "classifier":
            self.do_shap_classifiers(
                self.model,
                X_train, X_test, y_test, y_pred,
                instances,
                epd_test,
                scaler,
                show_plot, save_plot,
                plot_dir
            )
        else:
            raise ValueError("Unknown SHAP mode in ShapExplainer.py")
