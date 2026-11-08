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
sys.path.append('../')

from models.KumoTabular import KumoRegressor, KumoClassifier
from models.TabFM import TabFMRegressor, TabFMClassifier



def process_data(file, class_split="equidistant", remove=False):
    df = pd.read_csv(file)
    ratings = pd.to_numeric(df["rating"], errors="coerce").dropna()
    remove_attributes = []
    #for c in df["themes"]:
    #    remove_attributes.extend(c.split(" "))

    global cutoff1
    global cutoff2
    
    if class_split == "equidistant":
        classwidth = (ratings.max() - ratings.min()) / 3
        cutoff1 = ratings.min() + classwidth
        cutoff2 = ratings.min() + 2 * classwidth 
        df["difficulty"] = df["rating"].map(elo_to_difficulty)
    elif class_split == "balanced":
        rating_list = ratings.to_list()
        rating_list.sort()
        cutoff1 = rating_list[int(len(rating_list)/3)]
        cutoff2 = rating_list[int(2*len(rating_list)/3)]
        df["difficulty"] = df["rating"].map(elo_to_difficulty)
    else:
        raise("evaluation.py: unknown class_split parameter in process_data.")

    if remove:
        #X = df.drop(columns=["solution", "epd", "rating_dev", "classical_solved",  "sunfish_solved", "bare_solved"] + remove_attributes)
        X = df.drop(columns=["solution", "rating_dev", "classical_solved",  "sunfish_solved", "bare_solved"] + remove_attributes)
    else:
        #X = df.drop(columns=["solution", "epd", "rating_dev", "classical_solved",  "sunfish_solved", "bare_solved"])
        X = df.drop(columns=["solution", "rating_dev", "classical_solved",  "sunfish_solved", "bare_solved"])
    return X

# easy/medium/hard
def elo_to_difficulty(y):
    if y < cutoff1:
        return 0
    elif y < cutoff2:
        return 1
    else: 
        return 2

def heatmap(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred)
    #colors=["#FF0000", "#FFFF00", "#00FF00"]
    #cmap = LinearSegmentedColormap.from_list("custom_green_red", colors)
    fig, ax = plt.subplots(figsize=(6, 5))
    #cax = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    white_yellow = LinearSegmentedColormap.from_list(
        "white_yellow",
        ["#FFFADB", "#FFE221"]
    )
    cax = ax.imshow(cm, interpolation='nearest', cmap=white_yellow)
    fig.colorbar(cax)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", 
                    ha="center", va="center", color="black", fontsize=14)
    
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def evaluate(model, X_train, X_test, y_train, y_test, plots, standardize=True):
    print(type(model).__name__)
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    if is_regressor(model):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print("\tMAE:", MAE(y_test, y_pred))
    elif is_classifier(model):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print("\tCA:", accuracy_score(y_test, y_pred), "\tF1:", f1_score(y_test, y_pred, average="weighted"), "\tAUC:", roc_auc_score(y_test, model.predict_proba(X_test_scaled), average="weighted", multi_class="ovr"))
        if plots:
            heatmap(y_test, y_pred, type(model).__name__)
        else:
            pass
            #print(confusion_matrix(y_test, y_pred))
            #print(classification_report(y_test, y_pred))
    return y_pred, scaler

def evaluate_cv(model, X, y, folds=10, standardize=True):
    X_ = X.copy()
    y_ = y.copy()
    print(type(model).__name__)
    cas = []
    f1s = []
    aucs = []
    for _ in range(folds):
        idx = np.random.permutation(X_.index)
        X_.reindex(idx)
        y_.reindex(idx)

        X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, random_state=42)

        if standardize:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        if is_regressor(model):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            #print("\tMAE:", MAE(y_test, y_pred))
        elif is_classifier(model):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            cas.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, average="weighted"))
            aucs.append(roc_auc_score(y_test, model.predict_proba(X_test_scaled), average="weighted", multi_class="ovr"))
            #print("\tCA:", accuracy_score(y_test, y_pred), "\tF1:", f1_score(y_test, y_pred, average="weighted"), "\tAUC:", roc_auc_score(y_test, model.predict_proba(X_test_scaled), average="weighted", multi_class="ovr"))

    print("\tCA:", sum(cas)/len(cas), "\tF1:", sum(f1s)/len(f1s), "\tAUC:", sum(aucs)/len(aucs))
        

def evaluate_models(models, X, plots=False, cv=False):

    results = {}

    for model in models:
        model_name = type(model).__name__

        if is_regressor(model):
            y = X["rating"]
            model_type = "regressor"
        elif is_classifier(model):
            y = X["difficulty"]
            model_type = "classifier"
        else:
            continue
        
        X_ = X.drop(columns=["rating", "difficulty"])
        if cv:
            evaluate_cv(model, X_, y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2, random_state=42)
            y_pred, scaler = evaluate(model, X_train, X_test, y_train, y_test, plots)

            results[model_name] = {
                "type": model_type,
                "X_test": X_test,
                "y_test": y_test,
                "y_pred": y_pred,
                "scaler": scaler
            }

    # return X_test for shap
    return results

def priors(X):
    counts = np.array(X["difficulty"].value_counts())
    return counts/sum(counts)

class RegressorEnsemble(RegressorMixin, BaseEstimator):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        preds = []
        for model in self.models:
            pred = model.predict(X)
            preds.append(pred)

        sum_pred = np.zeros_like(preds[0])
        for pred in preds:
            sum_pred += pred
        return sum_pred/len(self.models)




def ablation(device="cpu"):
    X = process_data("../../datasets/dataset_test_engines.csv")
    #themes = []
    #for c in X["themes"]:
    #    themes.extend(c.split(" "))
    #X = X.drop(columns=["themes"])

    baseline = ["difficulty","rating","to_move","cp_eval","move1cp","nodes","move1sel_depth","move1w","move1d","move1l","move2cp","move2sel_depth","move2w","move2d","move2l",
                "move3cp","move3sel_depth","move3w","move3d","move3l","orig_w","orig_d","orig_l"]
    baseline = baseline + ["white_material","black_material","material_balance","total_material","white_P","white_N","white_B","white_R","white_Q","black_p","black_n","black_b","black_r","black_q"]
    study = ["meaningful_L1","meaningful_L2","meaningful_L3","branching_L2","branching_L3","avg_branching","narrow_L1","narrow_L2","narrow_L3","distance_L1",
             "distance_L2","distance_L3","pieces_L1","pieces_L2","pieces_L3","all_pieces_involved","winning_no_mate","possible_L1","possible_L2","possible_L3","all_possible_moves",
             "all_narrow_solutions","tree_size","move_ratio_L1","move_ratio_L2","move_ratio_L3","sum_distance","avg_distance"]
    engines = ["Bare","Classical","Maia-3 1320","Maia-3 2700","Otter","PyTuroChamp","Stockfish 1320","Stockfish 2700","Sunfish"]
    themes = ["discoveredCheck","deflection","bishopEndgame","quietMove","oneMove","interference","trappedPiece","endgame","fork","queenRookEndgame","knightEndgame","advancedPawn","sacrifice","enPassant",
                "castling","defensiveMove","middlegame","kingsideAttack","promotion","rookEndgame","doubleCheck","attackingF2F7","exposedKing","opening","intermezzo","mate","hangingPiece",
                "skewer","attraction","pawnEndgame","underPromotion","zugzwang","queenEndgame","xRayAttack","capturingDefender","queensideAttack","discoveredAttack"]
    misc = ["current_forks_L1","threatening_forks_L1","all_pins_L1","current_forks_L2","threatening_forks_L2","all_pins_L2","current_forks_L3","threatening_forks_L3",
            "all_pins_L3","current_forks_L4","threatening_forks_L4","all_pins_L4","current_forks_L5","threatening_forks_L5","all_pins_L5"]

    X1 = X[baseline]
    X2 = X[baseline + study]
    X3 = X[baseline + engines]
    X4 = X[baseline + themes]
    X5 = X[baseline + misc]
    X6 = X[baseline + study + engines]
    X7 = X[baseline + study + themes]
    X8 = X[baseline + study + misc]
    X9 = X[baseline + engines + themes]
    X10 = X[baseline + engines + misc]
    X11 = X[baseline + themes + misc]
    X12 = X[baseline + study + engines + themes + misc]

    kumo1 = KumoClassifier(device=device)
    kumo2 = KumoClassifier(device=device)
    kumo3 = KumoClassifier(device=device)
    kumo4 = KumoClassifier(device=device)
    kumo5 = KumoClassifier(device=device)
    kumo6 = KumoClassifier(device=device)
    kumo7 = KumoClassifier(device=device)
    kumo8 = KumoClassifier(device=device)
    kumo9 = KumoClassifier(device=device)
    kumo10 = KumoClassifier(device=device)
    kumo11 = KumoClassifier(device=device)
    kumo12 = KumoClassifier(device=device)

    evaluate_models([kumo1], X1, plots=False, cv=False)
    evaluate_models([kumo2], X2, plots=False, cv=False)
    evaluate_models([kumo3], X3, plots=False, cv=False)
    evaluate_models([kumo4], X4, plots=False, cv=False)
    evaluate_models([kumo5], X5, plots=False, cv=False)
    evaluate_models([kumo6], X6, plots=False, cv=False)
    evaluate_models([kumo7], X7, plots=False, cv=False)
    evaluate_models([kumo8], X8, plots=False, cv=False)
    evaluate_models([kumo9], X9, plots=False, cv=False)
    evaluate_models([kumo10], X10, plots=False, cv=False)
    evaluate_models([kumo11], X11, plots=False, cv=False)
    evaluate_models([kumo12], X12, plots=False, cv=False)


def do_shap_classifiers(model, X_test, y_test, y_pred, instances, shap_type, scaler):

    model_name = type(model).__name__
    print(f"\nSHAP for {model_name}")

    classes = range(len(model.classes_))

    if scaler is not None:
        X_test = (pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index))

    if shap_type == "tree":
        explainer = shap.TreeExplainer(model)

    elif shap_type == "lin": 
        explainer = shap.LinearExplainer(model, X_test)

    elif shap_type == "agn":
        # slow af, only a few samples will be collected        
        X_background = shap.sample(X_test, min(100, len(X_test)))
        explainer = shap.Explainer(model.predict_proba, X_background)
        
    shap_values = explainer(X_test)

    # we do shap for each of the classes
    for class_idx in classes:
        class_shap_values = shap_values[:, :, class_idx]

        shap.plots.beeswarm(class_shap_values, max_display=20, show=False)
        plt.title(
            f"SHAP Summary: {model_name}\n"
            f"for Class {class_idx}",
            fontsize=17,
            fontweight="bold"
        )
        plt.tight_layout()
        path = f"evaluation/shap_plots/{model_name}_class_{class_idx}_SHAP_summary.png"
        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


        for instance in instances:

            actual = y_test.iloc[instance] if hasattr(y_test, "iloc") else y_test[instance]
            predicted = y_pred[instance]

            sv = class_shap_values[instance]
            if np.asarray(sv.base_values).size == 1:
                sv.base_values = np.asarray(sv.base_values).item()

            shap.plots.waterfall(sv, max_display=20, show=False)
            plt.title(
                f"SHAP Explanation: {model_name}\n"
                f"for Class {class_idx}, Instance {instance}\n"
                f"Actual class: {actual} | Predicted class: {predicted}",
                fontsize=17,
                fontweight="bold"
            )
            plt.tight_layout()
            path = f"evaluation/shap_plots/{model_name}_class_{class_idx}_waterfall_{instance}.png"
            plt.savefig(
                path,
                dpi=300,
                bbox_inches="tight"
            )
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

    shap_values = explainer(X_test)

    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(
        f"SHAP Summary: {model_name}",
        fontsize=17,
        fontweight="bold"
    )
    plt.tight_layout()
    path = f"evaluation/shap_plots/{model_name}_SHAP_summary.png"
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    for instance in instances:

        actual = y_test.iloc[instance] if hasattr(y_test, "iloc") else y_test[instance]
        predicted = y_pred[instance]

        sv = shap_values[instance]
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
        path = f"evaluation/shap_plots/{model_name}_waterfall_{instance}.png"
        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()




if __name__ == "__main__":
    #X = process_data("./dataset_100k.csv")
    X = process_data("../../datasets/dataset_test_engines.csv", remove=True, class_split="balanced")

    
    #ablation(device="cuda")
    #exit()

    #print(X["difficulty"].value_counts())

    device="cpu"
    kumo_cl = KumoRegressor(device=device)
    tabfm_cl = TabFMRegressor(device=device)
    xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, eval_metric="mae")
    linreg = LinearRegression()

    regressors = [linreg, kumo_cl, tabfm_cl, xgb_reg]
    regressors = []
    #reg_ensemble = RegressorEnsemble(regressors)
    #regressors.append(reg_ensemble)

    kumo_cl = KumoClassifier(device=device)
    tabfm_cl = TabFMClassifier(device=device)
    xgb_cl = XGBClassifier(n_estimators=100, learning_rate=0.1)


    classifiers = [kumo_cl, tabfm_cl, xgb_cl]
    #classifiers.reverse()
    cl_ensemble = VotingClassifier([(type(cl).__name__, cl) for cl in classifiers], voting="soft")
    classifiers.append(cl_ensemble)

    models = regressors + classifiers
    results = evaluate_models(models, X, plots=True)


    """
    # ------------------------------
    # SHAP - does not work for evaluate_cv!
    # ------------------------------

    tree_explainers = ["XGBRegressor", "XGBClassifier", "RandomForestClassifier"]
    linear_explainers = ["LinearRegression", "LogisticRegression", "BayesianRidge", "SGDRegressor", "SGDClassifier"]
    model_agnostic_explainers = ["MLPRegressor", "MLPClassifier", "SVC", "GaussianNB", "KNeighborsClassifier", "RegressorEnsemble", "VotingClassifier", "GradientBoostingClassifier"]
    
    # if you want shap for specific puzzles
    instances = [13]

    for model_name, result in results.items():

        shap_type = None 
        if model_name in tree_explainers:
            shap_type = "tree"
        elif model_name in linear_explainers:
            shap_type = "lin"
        elif model_name in model_agnostic_explainers:
            # continue unless ur masochist <3
            shap_type = "agn"
            continue
        else:
            print(f"model {model_name} can't be shapped!")
            continue

    
        if result["type"] == "regressor":

            model = next((m for m in regressors if type(m).__name__ == model_name), None)
            if model is None:
                print(f"Skipping {model_name}: not found in classifiers.")
                continue       

            do_shap_regressors(model, result["X_test"], result["y_test"], result["y_pred"], instances, shap_type, result["scaler"])

        elif result["type"] == "classifier":

            model = next((m for m in classifiers if type(m).__name__ == model_name), None)
            if model is None:
                print(f"Skipping {model_name}: not found in classifiers.")
                continue       

            do_shap_classifiers(model, result["X_test"], result["y_test"], result["y_pred"], instances, shap_type, result["scaler"])
    
    """