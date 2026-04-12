import pandas as pd
import ast

import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    data = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(ast.literal_eval(line))

    df = pd.DataFrame(data)
    return df


PIECE_VALUES = {
    "P": 1, "p": 1,
    "N": 3, "n": 3,
    "B": 3, "b": 3,
    "R": 5, "r": 5,
    "Q": 9, "q": 9,     # 10 ? 
    "K": 0, "k": 0
}

def extract_piece_features(epd):
    board = epd.split(" ")[0]  # only board part
    rows = board.split("/")

    # piece counters
    white = {
        "P": 0, "N": 0, "B": 0, "R": 0, "Q": 0, "K": 0
    }
    black = {
        "p": 0, "n": 0, "b": 0, "r": 0, "q": 0, "k": 0
    }

    for row in rows:
        for char in row:
            if char.isdigit():
                continue
            if char.isupper():
                white[char] += 1
            else:
                black[char] += 1

    return white, black

def add_features_from_epd(df):
    # add features:
    #   number of pieces altogether
    #   pieces count, respectfully 
    #   material imbalance

    feature_rows = []

    for epd in df["epd"]:
        white, black = extract_piece_features(epd)

        # material
        white_material = sum(white[p] * PIECE_VALUES[p] for p in white)
        black_material = sum(black[p] * PIECE_VALUES[p] for p in black)

        material_balance = white_material - black_material
        total_material = white_material + black_material    # could add number of pieces on board instead of this 


        features = {
            "white_material": white_material,
            "black_material": black_material,
            "material_balance": material_balance,
            "total_material": total_material,
        }

        feature_rows.append(features)

        # add individual piece counts
        for p in white:
            features[f"white_{p}"] = white[p]
        for p in black:
            features[f"black_{p}"] = black[p]


    feature_df = pd.DataFrame(feature_rows)

    # merge with original data
    df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)


    #print(df)


    return df


def clean_data(df):


    # check types of columns 
    #for col in df.columns:
    #    print(f"{col}: {df[col].dtype}")

    # turn numeric columns from string to int
    numeric_cols = [
        'rating', 'rating_dev',
        'move1w', 'move1d', 'move1l',
        'move2w', 'move2d', 'move2l',
        'move3w', 'move3d', 'move3l'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')


    # turn categorical ones to one hot encoded
    # for now only to_move
    df["to_move"] = df["to_move"].map({"w": 1, "b": -1})

    # get rid of NaNs
    df = df.dropna().copy()


    # check types of columns again
    #for col in df.columns:
    #    print(f"{col}: {df[col].dtype}")

       
    return df


def data_analyse(df):

    min_rating = df["rating"].min()
    max_rating = df["rating"].max()

    bins = np.arange(0, max_rating + 100, 100)

    plt.figure()
    plt.hist(df["rating"], bins=bins)

    plt.xlabel("Rating range")
    plt.ylabel("Number of puzzles")
    plt.title("Rating distribution (100-point bins)")

    plt.show()
 

def save_data(df, output_path):
    df.to_csv(output_path, index=False)    

if __name__ == '__main__':

    # read txt file
    dataset = load_data("dataset.txt")
    #print(dataset.columns)


    # add some more features
    dataset = add_features_from_epd(dataset)


    # clean dataset of NaNs 
    print(f"size of data before cleaning: {len(dataset)}")
    dataset = clean_data(dataset)
    print(f"size of data after cleaning: {len(dataset)}")

    #print(dataset)

    # save to csv
    save_data(dataset, "dataset_upgraded.csv")


    #data_analyse(dataset)
