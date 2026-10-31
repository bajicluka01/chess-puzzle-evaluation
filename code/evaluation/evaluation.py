import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, is_classifier, is_regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor, XGBClassifier, XGBRanker
from sklearn.metrics import mean_absolute_error as MAE
#from lightgbm import LGBMRegressor
#from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import ndcg_score as ndcg
from scipy.stats import kendalltau
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def process_data(file, remove=False):
    df = pd.read_csv(file)
    remove_attributes = []
    #for c in df["themes"]:
    #    remove_attributes.extend(c.split(" "))
    df["difficulty"] = df["rating"].map(elo_to_difficulty)
    if remove:
        X = df.drop(columns=["solution", "epd", "rating_dev"] + remove_attributes)
    else:
        X = df.drop(columns=["solution", "epd", "rating_dev"])
    return X

# easy/medium/hard
def elo_to_difficulty(y):
    if y < 1200:
        return 0
    elif y < 1900:
        return 1
    else: 
        return 2

def heatmap(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred)
    colors=["#FF0000", "#FFFF00", "#00FF00"]
    cmap = LinearSegmentedColormap.from_list("custom_green_red", colors)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm, interpolation='nearest', cmap=cmap)
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

def ablation():
    X = process_data("./dataset_1k_final.csv")
    themes = []
    for c in X["themes"]:
        themes.extend(c.split(" "))
    X = X.drop(columns=["themes"])

    baseline = ["difficulty","rating","to_move","cp_eval","move1cp","nodes","move1multiPV","move1sel_depth","move1w","move1d","move1l","move2cp","move2multiPV","move2sel_depth","move2w","move2d","move2l","move3cp","move3multiPV","move3sel_depth","move3w","move3d","move3l","orig_w","orig_d","orig_l"]
    counts = ["white_material","black_material","material_balance","total_material","white_P","white_N","white_B","white_R","white_Q","white_K","black_p","black_n","black_b","black_r","black_q","black_k"]
    study = ["collinear","meaningful_L1","meaningful_L2","meaningful_L3","branching_L2","branching_L3","avg_branching","narrow_L1","narrow_L2","narrow_L3","distance_L1","distance_L2","distance_L3","pieces_L1","pieces_L2","pieces_L3","all_pieces_involved","winning_no_mate","possible_L1","possible_L2","possible_L3","all_possible_moves","all_narrow_solutions","tree_size","move_ratio_L1","move_ratio_L2","move_ratio_L3","sum_distance","avg_distance"]
    stockfish = ["solved1320","solved1500","solved1750","solved2000","solved2250","solved2500","solved2750","solved3000"]

    X_baseline = X[baseline]
    X_counts = X[baseline + counts]
    X_study = X[baseline + study]
    X_stockfish = X[baseline + stockfish]
    X_themes = X[baseline + themes]
    X_counts_study = X[baseline + counts + study]
    X_counts_stockfish = X[baseline + counts + stockfish]
    X_counts_themes = X[baseline + counts + themes]
    X_study_stockfish = X[baseline + study + stockfish]
    X_study_themes = X[baseline + study + themes]
    X_themes_stockfish = X[baseline + themes + stockfish]

    k = 10
    knn_baseline = KNeighborsClassifier(n_neighbors=k)
    knn_counts = KNeighborsClassifier(n_neighbors=k)
    knn_study = KNeighborsClassifier(n_neighbors=k)
    knn_stockfish = KNeighborsClassifier(n_neighbors=k)
    knn_themes = KNeighborsClassifier(n_neighbors=k)
    knn_counts_study = KNeighborsClassifier(n_neighbors=k)
    knn_counts_stockfish = KNeighborsClassifier(n_neighbors=k)
    knn_counts_themes = KNeighborsClassifier(n_neighbors=k)
    knn_study_stockfish = KNeighborsClassifier(n_neighbors=k)
    knn_study_themes = KNeighborsClassifier(n_neighbors=k)
    knn_themes_stockfish = KNeighborsClassifier(n_neighbors=k)
    knn_all = KNeighborsClassifier(n_neighbors=k)
    evaluate_models([knn_baseline], X_baseline, plots=False, cv=True)
    evaluate_models([knn_counts], X_counts, plots=False, cv=True)
    evaluate_models([knn_study], X_study, plots=False, cv=True)
    evaluate_models([knn_stockfish], X_stockfish, plots=False, cv=True)
    evaluate_models([knn_themes], X_themes, plots=False, cv=True)
    evaluate_models([knn_counts_study], X_counts_study, plots=False, cv=True)
    evaluate_models([knn_counts_stockfish], X_counts_stockfish, plots=False, cv=True)
    evaluate_models([knn_counts_themes], X_counts_themes, plots=False, cv=True)
    evaluate_models([knn_study_stockfish], X_study_stockfish, plots=False, cv=True)
    evaluate_models([knn_study_themes], X_study_themes, plots=False, cv=True)
    evaluate_models([knn_themes_stockfish], X_themes_stockfish, plots=False, cv=True)
    evaluate_models([knn_all], X, plots=False, cv=True)


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
    X = process_data("../../datasets/dataset_test.csv", remove=True)

    
    #ablation()
    #exit()

    #print(X["difficulty"].value_counts())
    
    xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, eval_metric="mae")
    linreg = LinearRegression()
    bayes_ridge = BayesianRidge()
    sgdreg = SGDRegressor()

    nn = MLPRegressor(
        hidden_layer_sizes=(256, 256, 128, 128, 64),
        activation='relu',
        solver='adam',
        batch_size=256,
        early_stopping=True,
        learning_rate="adaptive",
        max_iter=300,
        random_state=42,
        alpha=0.001
    )

    regressors = [xgb_reg, linreg, bayes_ridge, sgdreg, nn]
    reg_ensemble = RegressorEnsemble([xgb_reg, linreg, bayes_ridge, nn])
    regressors.append(reg_ensemble)

    xgb_cl = XGBClassifier(n_estimators=100, learning_rate=0.1)
    svm = SVC(C=1, class_weight="balanced", probability=True)
    naive_bayes = GaussianNB(priors=priors(X))
    rf = RandomForestClassifier(n_estimators=100)
    knn = KNeighborsClassifier(n_neighbors=10)
    logreg = LogisticRegression(class_weight="balanced")
    sgdcl = SGDClassifier(loss="log_loss")
    gbcl = GradientBoostingClassifier()

    nn1 = MLPClassifier(
        hidden_layer_sizes=(2048, 1024, 512, 256, 128),
        activation='relu',
        solver='adam',
        batch_size=512,
        early_stopping=True,
        max_iter=300,
        random_state=42,
        alpha=0.001
    )

    classifiers = [xgb_cl, svm, naive_bayes, rf, knn, logreg, sgdcl, gbcl, nn1]
    cl_ensemble = VotingClassifier([(type(cl).__name__, cl) for cl in classifiers], voting="soft")
    classifiers.append(cl_ensemble)

    models = regressors + classifiers
    results = evaluate_models(models, X, plots=False)

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
    
