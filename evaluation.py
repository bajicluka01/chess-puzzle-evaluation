import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ndcg_score as ndcg
from scipy.stats import kendalltau
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def process_data(file):
    df = pd.read_csv(file)
    remove_attributes = []
    for c in df["themes"]:
        remove_attributes.extend(c.split(" "))
    df["difficulty"] = df["rating"].map(elo_to_difficulty)
    X = df.drop(columns=["themes", "solution", "epd", "rating_dev"] + remove_attributes)
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
    colors=["#00FF00", "#FFFF00", "#FF0000"]
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
        print("\tMAE:", MAE(y_test, y_pred))
    elif is_classifier(model):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print("\tCA:", accuracy_score(y_test, y_pred))
        if plots:
            heatmap(y_test, y_pred, type(model).__name__)
        else:
            print(confusion_matrix(y_test, y_pred))
            #print(classification_report(y_test, y_pred))

def evaluate_models(models, X, plots=False):
    for model in models:
        if is_regressor(model):
            y = X["rating"]
        elif is_classifier(model):
            y = X["difficulty"]
        
        X_ = X.drop(columns=["rating", "difficulty"])
        X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2, random_state=42)
        evaluate(model, X_train, X_test, y_train, y_test, plots)

def priors(X):
    counts = np.array(X["difficulty"].value_counts())
    return counts/sum(counts)

class RegressorEnsemble(RegressorMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        preds = []
        for model in self.models:
            pred = model.predict(X)
            preds.append(pred)

        sum_pred = np.zeros_like(preds[0])
        for pred in preds:
            sum_pred += pred
        return sum_pred/len(self.models)

if __name__ == "__main__":
    X = process_data("./dataset_1k_final.csv")
    
    xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, eval_metric="mae")
    xgb_cl = XGBClassifier(n_estimators=100, learning_rate=0.1)
    linreg = LinearRegression()
    bayes_ridge = BayesianRidge()
    sgdreg = SGDRegressor()

    regressors = [xgb_reg, linreg, bayes_ridge, sgdreg]
    reg_ensemble = RegressorEnsemble([xgb_reg, linreg, bayes_ridge])
    regressors.append(reg_ensemble)

    svm = SVC(C=1, class_weight="balanced")
    naive_bayes = GaussianNB(priors=priors(X))
    rf = RandomForestClassifier(n_estimators=100)
    knn = KNeighborsClassifier(n_neighbors=10)
    #logreg = LogisticRegression(class_weight="balanced")
    sgdcl = SGDClassifier(loss="log_loss")
    gbcl = GradientBoostingClassifier()

    classifiers = [xgb_cl, svm, naive_bayes, rf, knn, sgdcl, gbcl]
    cl_ensemble = VotingClassifier([(type(cl).__name__, cl) for cl in classifiers], voting="hard")
    classifiers.append(cl_ensemble)

    models = regressors + classifiers
    evaluate_models(models, X, plots=True)
