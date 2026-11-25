import lime
import lime.lime_tabular
import pandas as pd
import numpy as np

from utilities.plots import lime_plot

class LimeExplainer:
    def __init__(self, model, columns=None):
        self.model = model
        # columns to display in plots 
        self.columns = columns 

    def explain(
            self,
            X_train, X_test, y_test, y_pred,
            mode="regressor",
            instances=None,
            epd_test=None,
            scaler=None,
            show_plot=False,
            save_plot=False,
            plot_dir=None,
            num_features=20,
            num_samples=5000,
            random_state=42
        ):

        model_name = type(self.model).__name__

        if instances is None:
            instances = []


        feature_names = X_train.columns.tolist()
        if self.columns is None:
            self.columns = feature_names


        if mode == "regressor":
            lime_mode = "regression"
            class_names = None

        elif mode == "classifier":
            lime_mode = "classification"
            class_names = [
                str(label) for label in self.model.classes_
            ]

        else:
            raise ValueError("Unknown LIME mode")


        # prediction function used by LIME
        # this function is useful because it standardizes data 
        # but will still plot original data
        def predict_fn(arr):

            X = pd.DataFrame(
                arr,
                columns=feature_names
            )

            if scaler is not None:
                X = pd.DataFrame(
                    scaler.transform(X),
                    columns=feature_names
                )

            if mode == "regressor":
                return np.asarray(
                    self.model.predict(X)
                ).ravel()

            else:
                return np.asarray(
                    self.model.predict_proba(X)
                )

        # LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.to_numpy(),
            feature_names=feature_names,
            class_names=class_names,
            mode=lime_mode,
            random_state=random_state,
            discretize_continuous=True
        )

        # locally explain chosen instances
        for instance in instances:

            actual = (
                y_test.iloc[instance]
                if hasattr(y_test, "iloc")
                else y_test[instance]
            )

            predicted = y_pred[instance]

            epd = (
                epd_test.iloc[instance]
                if epd_test is not None
                else None
            )

            # Explain instance
            kwargs = {}

            if mode == "classifier":
                kwargs["labels"] = tuple(
                    range(len(self.model.classes_))
                )

            exp = explainer.explain_instance(
                data_row=X_test.iloc[instance].to_numpy(),
                predict_fn=predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                **kwargs
            )


            if mode == "regressor":

                lime_plot(
                    exp=exp,
                    model_name=model_name,
                    instance=instance,
                    actual=actual,
                    predicted=predicted,
                    mode="regression",
                    epd=epd,
                    show_plot=show_plot,
                    save_plot=save_plot,
                    plot_dir=plot_dir
                )

            else:
                for class_idx, class_label in enumerate(
                    self.model.classes_
                ):

                    lime_plot(
                        exp=exp,
                        model_name=model_name,
                        instance=instance,
                        actual=actual,
                        predicted=predicted,
                        class_idx=class_idx,
                        class_label=class_label,
                        mode="classification",
                        epd=epd,
                        show_plot=show_plot,
                        save_plot=save_plot,
                        plot_dir=plot_dir
                    )