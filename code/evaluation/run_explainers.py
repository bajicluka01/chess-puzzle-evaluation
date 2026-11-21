import numpy as np
import shap
import sdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from explainers.LimeExplainer import LimeExplainer
from utilities.plots import beeswarm, waterfall, bar, lime_plot

# TODO rewrite this to have the same structure as lime_exp()
def shap_explanation(dataset, model, description_cols, target, device="cpu"):
    df = pd.read_csv(dataset)
    df = df[description_cols]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    stypes = sdm.infer_stypes(df, overrides={target: "numerical"})

    train_t = sdm.TableTensor.from_pandas(train_df, stypes=stypes, device=device)
    test_t  = sdm.TableTensor.from_pandas(test_df,  stypes=stypes, device=device)

    pred = model(
        x_context=train_t.drop_columns(target),
        y_context=train_t[:, target],
        x_query=test_t.drop_columns(target),
    )

    pred_df = pred.to_pandas()
    y_true = test_df[target].to_numpy()
    #y_med = pred_df.to_numpy()
    y_med = pred_df["q500"].to_numpy()

    print("MAE :", mean_absolute_error(y_true, y_med))
    print("RMSE:", root_mean_squared_error(y_true, y_med))
    print("R2  :", r2_score(y_true, y_med))

    # ---------- SHAP for Kumo (median prediction) ----------
    # ---------- SHAP for Kumo (median prediction) ----------
    feature_cols = [c for c in df.columns if c != target]
    feat_stypes = {k: v for k, v in stypes.items() if k in feature_cols}

    # Non-numeric (and bool) columns -> integer codes for SHAP
    cat_cols = [c for c in feature_cols
                if (not pd.api.types.is_numeric_dtype(df[c])) or pd.api.types.is_bool_dtype(df[c])]
    cat_maps = {c: pd.Index(df[c].dropna().unique()) for c in cat_cols}   # code -> label
    print("Categorical columns:", cat_cols)

    def encode(X):
        X = X[feature_cols].copy()
        for c in cat_cols:
            codes = cat_maps[c].get_indexer(X[c]).astype(float)   # -1 for NaN/unknown
            codes[codes == -1] = np.nan
            X[c] = codes
        return X.astype(float)

    def decode(arr):
        X = pd.DataFrame(arr, columns=feature_cols)
        for c in cat_cols:
            X[c] = [cat_maps[c][int(round(v))] if not np.isnan(v) else np.nan
                    for v in X[c].to_numpy(dtype=float)]
        return X

    x_ctx = train_t.drop_columns(target)
    y_ctx = train_t[:, target]

    def predict_fn(arr):
        X = decode(arr)
        xq = sdm.TableTensor.from_pandas(X, stypes=feat_stypes, device=device)
        out = model(x_context=x_ctx, y_context=y_ctx, x_query=xq).to_pandas()
        return out["q500"].to_numpy()
        #return out.to_numpy()

    X_train = encode(train_df)
    X_test = encode(test_df)

    #background = X_train.sample(min(30, len(X_train)), random_state=0)
    background = X_train.sample(min(10, len(X_train)), random_state=0)
    to_explain = X_test.iloc[:10]

    #masker = shap.maskers.Independent(background, max_samples=30)
    masker = shap.maskers.Independent(background, max_samples=10)
    explainer = shap.PermutationExplainer(predict_fn, masker)
    sv = explainer(to_explain, max_evals=2 * len(feature_cols) + 1)

    model_name = "tabFM" # TODO temp
    beeswarm(sv, model_name)
    waterfall(sv[0], model_name, 0, "unknown", "unknown", mode="regression")
    bar(sv, model_name)

    # Numeric ranking of features by mean |SHAP|
    importance = pd.Series(np.abs(sv.values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    print(importance.head(15))

def lime_exp(dataset, target):
    df = pd.read_csv(dataset)
    X = df.drop(columns=[target, "epd", "solution"])
    y = df[target]

    from xgboost import XGBRegressor
    xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, eval_metric="mae", random_state=42)

    X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_reg.fit(X_train, y_train)
    expl = LimeExplainer(xgb_reg)

    expl.explain(X_train, y_train, X_test)
        
if __name__ == "__main__":
    dataset = "../datasets/dataset_main.csv"
    target = "rating"
    lime_exp(dataset, target)
