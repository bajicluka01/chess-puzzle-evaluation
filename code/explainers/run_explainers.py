from pathlib import Path
from datetime import datetime

from explainers.ShapExplainer import ShapExplainer
from explainers.LimeExplainer import LimeExplainer
from utilities.plots import plot_boards


class RunExplainers:

    def __init__(
            self,
            results,
            models,
            show_plot=False,
            save_plot=True,
            plot_expl_plus_board=True
        ):
        """
            Explain selected models using SHAP and/or LIME.

            results                     : dictionary returned by evaluate_models()
            models                      : list of model objects to explain
            plot_expl_plus_board        : whether to plot chess boards besides explanation plot 
        """


        self.results = results
        self.models = models

        self.show_plot = show_plot
        self.save_plot = save_plot
        self.plot_expl_plus_board = plot_expl_plus_board

        # Create paths for current session
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        base_dir = (
            Path(__file__).resolve().parents[2]
            / "plots" / "explanations"
        )

        self.shap_dir = base_dir / "shap_plots" / timestamp
        self.lime_dir = base_dir / "lime_plots" / timestamp
        self.boards_dir = base_dir / "instances" / timestamp


    def _select_instances(
            self,
            result,
            model_name,
            instances,
            explain_misclassifications,
            max_instances
        ):
        """
            Select instances for local explanations and chessboards.
        """

        if explain_misclassifications:

            if "misclassifications" not in result:
                raise ValueError(
                    f"Misclassifications not found for {model_name}. "
                    "Run find_misclassifications() first."
                )

            model_instances = [
                idx
                for indices in result["misclassifications"].values()
                for idx in indices
            ]

        else:
            model_instances = [13] if instances is None else instances

        if max_instances is not None:
            if max_instances < 0:
                raise ValueError("max_instances must be non-negative")

            model_instances = model_instances[:max_instances]

        return model_instances


    def _run_explainer(
            self,
            explainer_class,
            result,
            instances,
            plot_dir
        ):
        """
            Run either selected explainer.
        """

        if self.save_plot:
            plot_dir.mkdir(parents=True, exist_ok=True)

        epd_test = (
            result["epd_test"]
            if self.plot_expl_plus_board
            else None
        )

        explainer = explainer_class(
            model=result["model"]
        )

        explainer.explain(
            X_train=result["X_train"],
            X_test=result["X_test"],
            y_test=result["y_test"],
            y_pred=result["y_pred"],
            mode=result["type"],
            instances=instances,
            epd_test=epd_test,
            scaler=result["scaler"],
            show_plot=self.show_plot,
            save_plot=self.save_plot,
            plot_dir=plot_dir
        )


    def _draw_boards(
            self,
            result,
            model_name,
            instances
        ):
        """
            Draw standalone chessboards for selected instances.
        """

        if result["epd_test"] is None:
            print(
                f"Skipping chessboards for {model_name}: "
                "no EPDs available."
            )
            return

        for instance in instances:

            actual = (
                result["y_test"].iloc[instance]
                if hasattr(result["y_test"], "iloc")
                else result["y_test"][instance]
            )

            predicted = result["y_pred"][instance]
            epd = result["epd_test"].iloc[instance]

            mode = (
                "regression"
                if result["type"] == "regressor"
                else "classification"
            )

            filename = f"{model_name}_{instance}_{mode}.pdf"

            plot_boards(
                epd=epd,
                model_name=model_name,
                instance=instance,
                actual=actual,
                predicted=predicted,
                mode=mode,
                show_plot=self.show_plot,
                save_plot=self.save_plot,
                plot_path=self.boards_dir / filename
            )


    def run(
            self,
            do_shap=True,
            do_lime=False,
            instances=None,
            explain_misclassifications=False,
            max_instances=None,
            draw_boards=True
        ):
        """
            Run selected explainers for selected models.

            do_shap                     : whether to generate SHAP explanations
            do_lime                     : whether to generate LIME explanations
            instances                   : indices of puzzles for local explanations
            explain_misclassifications  : whether to explain misclassifications rather than instances
            max_instances               : max misclassifications to explain
            draw_boards                 : whether to draw standalone chessboard plots            
        """

        if not do_shap and not do_lime and not draw_boards:
            print("No plots selected.")
            return

        for model in self.models:

            model_name = type(model).__name__

            if model_name not in self.results:
                print(f"Skipping {model_name}: not found in results.")
                continue

            result = self.results[model_name]

            model_instances = self._select_instances(
                result=result,
                model_name=model_name,
                instances=instances,
                explain_misclassifications=explain_misclassifications,
                max_instances=max_instances
            )

            if do_shap:
                self._run_explainer(
                    explainer_class=ShapExplainer,
                    result=result,
                    instances=model_instances,
                    plot_dir=self.shap_dir
                )

            if do_lime:
                self._run_explainer(
                    explainer_class=LimeExplainer,
                    result=result,
                    instances=model_instances,
                    plot_dir=self.lime_dir
                )

            if draw_boards:
                self._draw_boards(
                    result=result,
                    model_name=model_name,
                    instances=model_instances
                )