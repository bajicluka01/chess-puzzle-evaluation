import lime
import lime.lime_tabular
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
                from pathlib import Path
                output_dir = Path(__file__).resolve().parent / "prsezm"
                output_dir.mkdir(parents=True, exist_ok=True)
            
                fig = exp.as_pyplot_figure()
                ax = fig.axes[0]
        
                # Vertical grid lines
                ax.grid(axis="x", color="black", linewidth=0.5, alpha=0.2)
        
                # Horizontal grid lines between bars
                bar_positions = [bar.get_y() + bar.get_height() / 2 for bar in ax.patches]
        
                ax.set_yticks(np.array(bar_positions) + 0.5, minor=True)
                ax.grid(axis="y", which="minor", color="black", linewidth=0.5, alpha=0.2)
                ax.grid(axis="y", which="major", visible=False)
                ax.axvline(x=0, color="black", linewidth=2, alpha=0.6)
                ax.tick_params(axis="y", which="minor", length=0)
                ax.set_axisbelow(True)
                ax.set_title(
                    f"LIME Explanation - Puzzle {j}\n"
                    f"Test position: {j}",
                    fontsize=20,
                    fontweight="bold",
                    pad=20,
                )
        
                fig.set_size_inches(16, 10)
                fig.tight_layout()
        
                fig.savefig(output_dir / f"lime_test_{j}_puzzle_{j}.pdf", bbox_inches="tight")
                plt.close(fig)
