import pandas as pd
import numpy as np
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
from sklearn.ensemble import StackingRegressor, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, VotingRegressor
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score, mean_squared_error as MSE, root_mean_squared_error as RMSE
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
from models.ensembles import RegressorEnsemble
from utilities.plots import heatmap, beeswarm, waterfall

def process_data(file, class_split="equidistant", remove=[], dropepd=True):
    df = pd.read_csv(file)
    ratings = pd.to_numeric(df["rating"], errors="coerce").dropna()

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

    X = df.drop(columns=["solution", "rating_dev"] + remove)
    
    if dropepd:
        X = X.drop(columns=["epd"])
    return X

# easy/medium/hard
def elo_to_difficulty(y):
    if y < cutoff1:
        return 0
    elif y < cutoff2:
        return 1
    else: 
        return 2

def evaluate(model, X_train, X_test, y_train, y_test, plots, standardize=True):
    print(type(model).__name__)
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    if is_regressor(model):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print("\tMAE:", MAE(y_test, y_pred))
        print("\tMSE:", MSE(y_test, y_pred))
        print("\tRMSE:", RMSE(y_test, y_pred))
    elif is_classifier(model):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print("\tCA:", accuracy_score(y_test, y_pred), "\tF1:", f1_score(y_test, y_pred, average="weighted"), "\tAUC:", roc_auc_score(y_test, model.predict_proba(X_test_scaled), average="weighted", multi_class="ovr"))
        if plots:
            heatmap(y_test, y_pred, type(model).__name__)
        else:
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
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



def ablation(device="cpu"):
    X = process_data("../../datasets/dataset_main.csv")

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

if __name__ == "__main__":
    X = process_data("../../datasets/dataset_main.csv", remove=True, class_split="balanced")

    #ablation(device="cuda")
    #exit()

    device="cpu"
    random_state=42
    kumo_reg = KumoRegressor(device=device, random_state=random_state, optimize="mse", quantile=None)
    tabfm_reg = TabFMRegressor(device=device, random_state=random_state)
    xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, eval_metric="mae", random_state=random_state)
    linreg = LinearRegression()

    regressors = [linreg, kumo_reg, tabfm_reg, xgb_reg]
    #regressors = []
    #reg_ensemble = RegressorEnsemble(regressors)
    reg_ensemble = VotingRegressor(regressors)
    #regressors.append(reg_ensemble)

    lreg_cl = LogisticRegression(random_state=random_state)
    kumo_cl = KumoClassifier(device=device, random_state=random_state)
    tabfm_cl = TabFMClassifier(device=device, random_state=random_state)
    xgb_cl = XGBClassifier(n_estimators=100, learning_rate=0.1)

    classifiers = [kumo_cl, tabfm_cl, xgb_cl]
    classifiers = []
    cl_ensemble = VotingClassifier([(type(cl).__name__, cl) for cl in classifiers], voting="soft")
    classifiers.append(cl_ensemble)

    models = regressors + classifiers
    results = evaluate_models([kumo_reg], X, plots=True)

    """ MISCLASSIFICATIONS
    X = process_data(
            "../../datasets/dataset_main.csv",
            remove=False,
            class_split="balanced",
        )
    
        y = X["difficulty"]
        X = X.drop(columns=["rating", "difficulty"])
        #X = X[description_cols]
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        # Keep EPD separately for chess board visualizations
        #X_train_epd = X_train["epd"]
        X_test_epd = X_test["epd"]
        X_train = X_train.drop(columns=["epd"])
        X_test = X_test.drop(columns=["epd"])
    
        #X_train = X_train.drop(columns="epd")
        #X_test = X_test.drop(columns="epd")
    
        # Train KumoClassifier
        model = KumoClassifier(device="cpu", random_state=42)
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test)).ravel()
    
        # Find misclassified puzzles
        y_true = y_test.to_numpy()
    
        # Easy -> Hard
        easy_as_hard_idx = np.where(
            (y_true == 0) & (y_pred == 2)
        )[0]
    
        # Hard -> Easy
        hard_as_easy_idx = np.where(
            (y_true == 2) & (y_pred == 0)
        )[0]
    
        difficulty0_as_difficulty0_idx = np.where(
            (y_true == 0) & (y_pred == 0)
        )[0]
        difficulty1_as_difficulty1_idx = np.where(
            (y_true == 1) & (y_pred == 1)
        )[0]
        difficulty2_as_difficulty2_idx = np.where(
            (y_true == 2) & (y_pred == 2)
        )[0]
    
        heatmap(y_true, y_pred, title="holyheatmap")
    
        print("Easy classified as hard:", easy_as_hard_idx)
        print("Hard classified as easy:", hard_as_easy_idx)
        print()
        print("Easy classified as hard:")
        for x in X_test_epd.iloc[easy_as_hard_idx]:
            print(x)
        print()
        print("Hard classified as easy:")
        for x in X_test_epd.iloc[hard_as_easy_idx]:
            print(x)
        print()
    
        print("Difficulty 0 classified as difficulty 0:")
        for x in X_test_epd.iloc[difficulty0_as_difficulty0_idx].head(5):
            print(x)
        print()
    
        print("Difficulty 1 classified as difficulty 1:")
        for x in X_test_epd.iloc[difficulty1_as_difficulty1_idx].head(5):
            print(x)
        print()
    
        print("Difficulty 2 classified as difficulty 2:")
        for x in X_test_epd.iloc[difficulty2_as_difficulty2_idx].head(5):
            print(x)
        """
