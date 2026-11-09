import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shap
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, is_classifier, is_regressor
#from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
#from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor, XGBClassifier, XGBRanker
from sklearn.metrics import mean_absolute_error as MAE
#from lightgbm import LGBMRegressor
#from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
#from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import ndcg_score as ndcg
#from scipy.stats import kendalltau
#from sklearn.svm import SVC
#from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.KumoTabular import KumoRegressor, KumoClassifier
from models.TabFM import TabFMRegressor, TabFMClassifier

from evaluation.evaluation import process_data, evaluate_models


def do_shap_classifiers(model, X_test, y_test, y_pred, instances, shap_type, scaler):

    model_name = type(model).__name__
    print(f"\nSHAP for {model_name}")

    if scaler is not None:
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

    # Select only misclassified puzzles
    X_selected = X_test.iloc[instances]

    if shap_type == "tree":
        explainer = shap.TreeExplainer(model)

    elif shap_type == "lin":
        explainer = shap.LinearExplainer(model, X_test)

    elif shap_type == "agn":
        X_background = shap.sample(X_test, min(100, len(X_test)))

        # Kumo expects a DataFrame, not a NumPy array
        def predict_fn(arr):
            df = pd.DataFrame(arr, columns=X_test.columns)
            return np.asarray(model.predict_proba(df))

        explainer = shap.Explainer(predict_fn, X_background)

    else:
        raise ValueError(f"Unknown SHAP type: {shap_type}")

    shap_values = explainer(X_selected)

    # Explain each class
    classes = range(shap_values.values.shape[2])

    for class_idx in classes:
        class_shap_values = shap_values[:, :, class_idx]

        shap.plots.beeswarm(
            class_shap_values,
            max_display=20,
            show=False
        )

        plt.title(
            f"SHAP Summary: {model_name}\n"
            f"for Class {class_idx}",
            fontsize=17,
            fontweight="bold"
        )
        plt.tight_layout()

        path = f"./shap_classifiers/{model_name}_class_{class_idx}_SHAP_summary.pdf"
        plt.savefig(path, dpi=300, bbox_inches="tight", format="pdf")
        plt.close()

        # Explain individual misclassified puzzles
        for pos, instance in enumerate(instances):

            actual = y_test.iloc[instance]
            predicted = y_pred[instance]

            # pos is the position inside X_selected
            sv = class_shap_values[pos]

            if np.asarray(sv.base_values).size == 1:
                sv.base_values = np.asarray(sv.base_values).item()

            shap.plots.waterfall(
                sv,
                max_display=20,
                show=False
            )

            plt.title(
                f"SHAP Explanation: {model_name}\n"
                f"for Class {class_idx}, Instance {instance}\n"
                f"Actual class: {actual} | Predicted class: {predicted}",
                fontsize=17,
                fontweight="bold"
            )
            plt.tight_layout()

            path = f"./shap_classifiers/{model_name}_class_{class_idx}_waterfall_{instance}.pdf"
            plt.savefig(path, dpi=300, bbox_inches="tight", format="pdf")
            plt.close()


def do_shap_regressors(model, X_test, y_test, y_pred, instances, shap_type, scaler):

    model_name = type(model).__name__
    print(f"\nSHAP for {model_name}")

    if scaler is not None:
        X_test = (pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index))

    if shap_type == "tree":
        explainer = shap.TreeExplainer(model)

    elif shap_type == "lin":
        explainer = shap.LinearExplainer(model, X_test)

    elif shap_type == "agn":
        # slow af, only a few samples will be collected
        X_background = shap.sample(X_test, min(100, len(X_test)))
        explainer = shap.Explainer(model.predict, X_background)

    else:
        print("!!! wrong shap type")

    X_selected = X_test.iloc[instances]
    shap_values = explainer(X_selected)

    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(
        f"SHAP Summary: {model_name}",
        fontsize=17,
        fontweight="bold"
    )
    plt.tight_layout()
    path = f"./shap_regressors/{model_name}_SHAP_summary.pdf"
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        format="pdf"
    )
    plt.close()

    for i, instance in enumerate(instances):

        actual = y_test.iloc[instance] if hasattr(y_test, "iloc") else y_test[instance]
        predicted = y_pred[instance]

        sv = shap_values[i]
        if np.asarray(sv.base_values).size == 1:
            sv.base_values = np.asarray(sv.base_values).item()

        shap.plots.waterfall(sv, max_display=20, show=False)
        plt.title(
            f"SHAP Explanation: {model_name}\n"
            f"for Instance {instance}\n"
            f"Actual rating: {actual:.0f} | Predicted rating: {predicted:.0f}",
            fontsize=17,
            fontweight="bold"
        )
        plt.tight_layout()
        path = f"./shap_regressors/{model_name}_waterfall_{instance}.pdf"
        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            format="pdf"
        )
        plt.close()


if __name__ == "__main__":

    '''
    Easy classified as hard: [ 25  33 138 200 228 230 279 362 395 396 458 474 481 484 554 557 605 669]
    Hard classified as easy: [27]
    '''

    description_cols = [
            'white_material', 'black_material', 'material_balance', 'total_material', 'white_P', 'white_N', 'white_B',
            'white_R', 'white_Q', 'black_p', 'black_n', 'black_b', 'black_r', 'black_q', 'current_forks_L1',
            'threatening_forks_L1', 'all_pins_L1', 'meaningful_L1', 'winning_no_mate', 'pieces_L1', 'meaningful_L2', 'possible_L2',
            'pieces_L2', 'current_forks_L2', 'threatening_forks_L2', 'all_pins_L2', 'meaningful_L3', 'possible_L3', 'pieces_L3',
            'current_forks_L3', 'threatening_forks_L3', 'all_pins_L3', 'meaningful_L4', 'possible_L4', 'pieces_L4', 'current_forks_L4',
            'threatening_forks_L4', 'all_pins_L4', 'meaningful_L5', 'possible_L5', 'pieces_L5',
            'current_forks_L5', 'threatening_forks_L5', 'all_pins_L5', 'all_pieces_involved',
            'possible_L1', 'all_possible_moves', 'discoveredCheck', 'deflection', 'bishopEndgame', 'quietMove', 'oneMove',
            'interference', 'trappedPiece', 'endgame', 'fork', 'queenRookEndgame', 'knightEndgame', 'advancedPawn', 'sacrifice',
            'enPassant', 'castling', 'defensiveMove', 'middlegame', 'kingsideAttack', 'promotion', 'rookEndgame', 'doubleCheck',
            'attackingF2F7', 'exposedKing', 'opening', 'intermezzo', 'mate', 'hangingPiece', 'skewer', 'attraction', 'pawnEndgame',
            'underPromotion', 'zugzwang', 'queenEndgame', 'xRayAttack', 'capturingDefender', 'queensideAttack', 'discoveredAttack',
            "nodes", "move1sel_depth", "move1w", "move1d", "move1l", "move2cp", "move2sel_depth", "move2w", "move2d",
            "move2l", "move3cp", "move3sel_depth", "move3w", "move3d", "move3l", "orig_w", "orig_d", "orig_l"
        ]


    X = process_data(
        "../../datasets/dataset_test_engines.csv",
        remove=False,
        class_split="balanced"
    )

    y = X["difficulty"]
    X = X.drop(columns=["rating", "difficulty"])
    X = X[description_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Keep EPD separately for chess board visualizations
    #X_train_epd = X_train["epd"]
    #X_test_epd = X_test["epd"]

    #X_train = X_train.drop(columns="epd")
    #X_test = X_test.drop(columns="epd")

    # Train KumoClassifier
    model = KumoRegressor(device="cpu", random_state=42)
    model.fit(X_train, y_train)
    y_pred = np.asarray(model.predict(X_test)).ravel()

    # Find misclassified puzzles
    y_true = y_test.to_numpy()

    # Easy -> Hard
    #easy_as_hard_idx = np.where(
    #    (y_true == 0) & (y_pred == 2)
    #)[0]

    # Hard -> Easy
    #hard_as_easy_idx = np.where(
    #    (y_true == 2) & (y_pred == 0)
    #)[0]

    #print("Easy classified as hard:", easy_as_hard_idx)
    #print("Hard classified as easy:", hard_as_easy_idx)


    #easy_as_hard_idx = [200, 228, 362, 415, 554, 605, 669]
    #hard_as_easy_idx = [27, 354, 585]
    # Combine misclassified puzzle indices
    #instances = list(easy_as_hard_idx) + list(hard_as_easy_idx)
    instances = [25, 33, 138, 200, 228, 230, 279, 362, 395, 396, 458, 474, 481, 484, 554, 557, 605, 669] + [27]

    if len(instances) == 0:
        print("No Easy/Hard misclassifications found.")
    else:
        do_shap_regressors(
            model=model,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            instances=instances,
            shap_type="agn",
            scaler=None
        )
