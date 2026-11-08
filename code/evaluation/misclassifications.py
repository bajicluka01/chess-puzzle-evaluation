import numpy as np
import shap
import matplotlib.pyplot as plt
import sdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import chess
#import cairosvg
from io import BytesIO
import chess.svg

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.KumoTabular import KumoClassifier
from evaluation.evaluation import process_data

def plot_boards(epd):
    """
        Visualizes puzzle from given epd
    """

    board = chess.Board()
    board.set_epd(epd)
    #print(board)
    epd = epd.replace("/", "_")

    svg = chess.svg.board(board=board, size=400)


def plot_group(group, group_name, original_df, n=3,  start=0):
    """
    Visualizes n random puzzles from a difficulty group.
    """

    samples = group.iloc[start:start + n]

    if samples.empty:
        print(f"No puzzles found in {group_name}")
        return

    fig, axes = plt.subplots(1, len(samples), figsize=(5 * len(samples), 5))

    axes = np.atleast_1d(axes)

    for ax, (idx, row) in zip(axes, samples.iterrows()):

        epd = original_df.loc[idx, "epd"]

        board = chess.Board()
        board.set_epd(epd)

        svg = chess.svg.board(board=board, size=400)

    fig.suptitle(group_name, fontsize=17, fontweight="bold")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    #epd = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w - -"
    #plot_boards(epd)
    original_df = pd.read_csv("../../datasets/dataset_test_engines.csv")
    X = process_data("../../datasets/dataset_test_engines.csv", remove=True, class_split="balanced")

    y = X["difficulty"]
    X = X.drop(columns=["rating", "difficulty"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_epd = X_train["epd"]
    X_test_epd = X_test["epd"]
    X_train = X_train.drop(columns="epd")
    X_test = X_test.drop(columns="epd")


    model = KumoClassifier(device="cuda", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)



    classes = np.unique(y_train)
    print("train")
    print(f"Classes: {classes}")
    counts = [np.sum(y_train == y_i) for y_i in classes]
    print(f"Count: {counts}")

    classes = np.unique(y_test)
    print("test")
    print(f"Classes: {classes}")
    counts = [np.sum(y_test == y_i) for y_i in classes]
    print(f"Count: {counts}")


    #X_train = X_train.reset_index(drop=True)
    #y_train = y_train.reset_index(drop=True)


    # easy -> predicted hard
    easy_as_hard_idx = np.where((y_test == 0) & (y_pred == 2))[0]

    # hard -> predicted easy
    hard_as_easy_idx = np.where((y_test == 2) & (y_pred == 0))[0]

    print("Easy classified as hard:", easy_as_hard_idx)
    print("Hard classified as easy:", hard_as_easy_idx)

    for eas in easy_as_hard_idx:
        print(f"easy classified as hard idx={eas}")
        #epd = X_test_epd.iloc[eas]["epd"]
        epd = original_df.loc[X_test.index[eas], "epd"]

        plot_boards(epd)

    for eas in hard_as_easy_idx:
        print(f"hard classified as easy idx={eas}")
        #epd = X_test_epd.iloc[eas]["epd"]
        epd = original_df.loc[X_test.index[eas], "epd"]
        plot_boards(epd)


    """
    easy_ones = [3]

    for idx,x in enumerate(hard_as_hard_idx):

        if idx > 0:
            break
        if idx not in easy_ones:
            continue
        starting_idx = 600

        print()
        print()
        print(f"IDX = {idx}")
        print("EPD:", df.iloc[starting_idx + x]["epd"])
        print(df.iloc[starting_idx + x])
        print(y_pred[x])
        print(y_test[starting_idx + x])
    """
    """
    sample = X_test.iloc[[x]]
    shap_values = explainer(sample)
    pred_class = int(y_pred[x])
    sv = shap_values[0][:, pred_class]   # pick sample + class properly
    shap.plots.waterfall(sv, max_display=15, show=False)
    plt.savefig(f"hard_puzzle_{idx+1}_shap.png", bbox_inches="tight")
    plt.close()
    """

    """
    import shap
    explainer = shap.TreeExplainer(model)

    sample = X_test.iloc[[84]]
    shap_values = explainer(sample)
    pred_class = int(y_pred[84])
    sv = shap_values[0][:, pred_class]   # pick sample + class properly
    shap.plots.waterfall(sv, show=False)
    plt.savefig(f"easy_puzzle_miss_1_shap.png", bbox_inches="tight")
    plt.close()

    sample = X_test.iloc[[86]]
    shap_values = explainer(sample)
    pred_class = int(y_pred[86])
    sv = shap_values[0][:, pred_class]   # pick sample + class properly
    shap.plots.waterfall(sv, show=False)
    plt.savefig(f"easy_puzzle_miss_2_shap.png", bbox_inches="tight")
    plt.close()

    sample = X_test.iloc[[104]]
    shap_values = explainer(sample)
    pred_class = int(y_pred[104])
    sv = shap_values[0][:, pred_class]   # pick sample + class properly
    shap.plots.waterfall(sv, show=False)
    plt.savefig(f"easy_puzzle_miss_3_shap.png", bbox_inches="tight")
    plt.close()
    """



    """
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

    original_df = pd.read_csv("../datasets/dataset_test.csv")
    df = original_df[description_cols].copy()
    #print(df)
    #df = df.drop(columns=["epd", "solution", "rating_dev"])


    target = "rating"

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    stypes = sdm.infer_stypes(df, overrides={target: "numerical"})
    #print(stypes)  # sanity check

    train_t = sdm.TableTensor.from_pandas(train_df, stypes=stypes, device="cuda")
    test_t  = sdm.TableTensor.from_pandas(test_df,  stypes=stypes, device="cuda")

    #model = sdm.models.KumoTabular(task="regression", device="cuda")
    model = sdm.models.TabFM(task="regression", device="cuda")
    pred = model(
        x_context=train_t.drop_columns(target),
        y_context=train_t[:, target],
        x_query=test_t.drop_columns(target),
    )

    pred_df = pred.to_pandas()
    y_true = test_df[target].to_numpy()
    y_med = pred_df.to_numpy()
    #y_med = pred_df["q500"].to_numpy()


    y_pred = np.asarray(y_med).ravel()

    actual_classes = np.array([elo_to_difficulty(y) for y in y_true])
    predicted_classes = np.array([elo_to_difficulty(y) for y in y_pred])

    results = test_df.copy()
    results["predicted_rating"] = y_pred
    results["actual_class"] = actual_classes
    results["predicted_class"] = predicted_classes

    easy_easy = results[
        (results["actual_class"] == 0) &
        (results["predicted_class"] == 0)
    ]

    hard_hard = results[
        (results["actual_class"] == 2) &
        (results["predicted_class"] == 2)
    ]

    easy_hard = results[
        (results["actual_class"] == 0) &
        (results["predicted_class"] == 2)
    ]

    hard_easy = results[
        (results["actual_class"] == 2) &
        (results["predicted_class"] == 0)
    ]

    # if you want access epd 
    easy_hard_puzzles = original_df.loc[easy_hard.index]
    hard_easy_puzzles = original_df.loc[hard_easy.index]



    plot_group(easy_easy, "Easy → Easy", original_df, n=3, start=0)
    plot_group(hard_hard, "Hard → Hard", original_df, n=3, start=0)
    plot_group(easy_hard, "Easy → Hard", original_df, n=3, start=0)
    plot_group(hard_easy, "Hard → Easy", original_df, n=3, start=0)



    print("MAE :", mean_absolute_error(y_true, y_med))
    print("RMSE:", root_mean_squared_error(y_true, y_med))
    print("R2  :", r2_score(y_true, y_med))
    """