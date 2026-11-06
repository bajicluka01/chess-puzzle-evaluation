import numpy as np
import shap
import matplotlib.pyplot as plt
import sdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import lime
import lime.lime_tabular

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

    # Plots
    shap.plots.beeswarm(sv, show=False, max_display=20)
    plt.savefig("shap_tabfm_beeswarm.png", bbox_inches="tight", dpi=200)
    plt.close()

    shap.plots.bar(sv, show=False, max_display=20)
    plt.savefig("shap_tabfm_bar.png", bbox_inches="tight", dpi=200)
    plt.close()

    shap.plots.waterfall(sv[0], show=False, max_display=20)
    plt.savefig("shap_tabfm_waterfall_row0.png", bbox_inches="tight", dpi=200)
    plt.close()

    # Numeric ranking of features by mean |SHAP|
    importance = pd.Series(np.abs(sv.values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    print(importance.head(15))

def lime_explanation(dataset, model, description_cols, target, device="cpu"):
    df = pd.read_csv(dataset)
    df = df[description_cols]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    stypes = sdm.infer_stypes(df, overrides={target: "numerical"})

    train_t = sdm.TableTensor.from_pandas(train_df, stypes=stypes, device=device)
    test_t  = sdm.TableTensor.from_pandas(test_df,  stypes=stypes, device=device)

    #trainy = train_df[target]
    #trainX = train_df.drop(target, axis=1)

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

    def predict_fn(arr):
        X = arr
        x_ctx = train_t.drop_columns(target)
        y_ctx = train_t[:, target]
        all_cols = [c for c in df.columns]
        feature_cols = [c for c in df.columns if c != target]
        feat_stypes = {k: v for k, v in stypes.items() if k in feature_cols}
        xq = sdm.TableTensor.from_pandas(pd.DataFrame(X, columns=all_cols), stypes=feat_stypes, device=device)
        out = model(x_context=x_ctx, y_context=y_ctx, x_query=xq).to_pandas()
        return out["q500"].to_numpy()


    explainer = lime.lime_tabular.LimeTabularExplainer(train_df.values, feature_names=train_df.columns.values.tolist(),
                                                  class_names=[target], verbose=True, mode='regression')
    j = 5
    exp = explainer.explain_instance(test_df.values[j], predict_fn, num_features=20)
    exp_list = exp.as_list()
    exp.save_to_file("lime.html")
    print(exp_list)
    #exp.show_in_notebook(show_table=True)
    #plot = exp.as_pyplot_figure(label = 1, figsize=(1000, 480))
    plot = exp.as_pyplot_figure()
    print(plot)
    plt.figure(plot)
    plt.savefig('lime_explanation.png')
    

if __name__ == "__main__":
    dataset = "datasets/dataset_test.csv"

    description_cols = [
                'rating', 'white_material', 'black_material', 'material_balance', 'total_material', 'white_P', 'white_N', 'white_B',
                'white_R', 'white_Q', 'white_K', 'black_p', 'black_n', 'black_b', 'black_r', 'black_q', 'black_k', 'current_forks_L1',
                'threatening_forks_L1', 'all_pins_L1', 'meaningful_L1', 'winning_no_mate', 'pieces_L1', 'meaningful_L2', 'possible_L2',
                'pieces_L2', 'current_forks_L2', 'threatening_forks_L2', 'all_pins_L2', 'meaningful_L3', 'possible_L3', 'pieces_L3',
                'current_forks_L3', 'threatening_forks_L3', 'all_pins_L3', 'meaningful_L4', 'possible_L4', 'pieces_L4', 'current_forks_L4',
                'threatening_forks_L4', 'all_pins_L4', 'meaningful_L5', 'possible_L5', 'pieces_L5',
                'current_forks_L5', 'threatening_forks_L5', 'all_pins_L5', 'all_pieces_involved',
                'possible_L1', 'all_possible_moves', 'discoveredCheck', 'deflection', 'bishopEndgame', 'quietMove', 'oneMove',
                'interference', 'trappedPiece', 'endgame', 'fork', 'queenRookEndgame', 'knightEndgame', 'advancedPawn', 'sacrifice',
                'enPassant', 'castling', 'defensiveMove', 'middlegame', 'kingsideAttack', 'promotion', 'rookEndgame', 'doubleCheck',
                'attackingF2F7', 'exposedKing', 'opening', 'intermezzo', 'mate', 'hangingPiece', 'skewer', 'attraction', 'pawnEndgame',
                'underPromotion', 'zugzwang', 'queenEndgame', 'xRayAttack', 'capturingDefender', 'queensideAttack', 'discoveredAttack']
    target = "rating"

    model = sdm.models.KumoTabular(task="regression", device="cpu")
    #model = sdm.models.TabFM(task="regression", device="cpu")
    
    #shap_explanation(dataset, model, description_cols, target)
    lime_explanation(dataset, model, description_cols, target)
