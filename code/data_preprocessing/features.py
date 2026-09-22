import os
import os.path as path
import pandas
from stockfish import Stockfish
from chess_functions.functions import *
from attributes_from_study import get_level_attributes

#import warnings
#warnings.filterwarnings("ignore")


# TODO: check if this is okay
EVAL_BOUND = 1500
PIECE_VALUES = {
    "P": 1, "p": 1,
    "N": 3, "n": 3,
    "B": 3, "b": 3,
    "R": 5, "r": 5,
    "Q": 9, "q": 9,
    "K": 0, "k": 0
}
LEVELS = 3
# These themes are from lichess dataset (6.100.952 puzzles, last updated 2026-09-10, computed with get_all_themes function)
ALL_THEMES = ['balestraMate', 'discoveredCheck', 'deflection', 'bishopEndgame', 'quietMove', 'epauletteMate', 'oneMove',
              'arabianMate', 'interference', 'trappedPiece', 'cornerMate', 'endgame', 'fork', 'crushing',
              'collinearMove', 'queenRookEndgame', 'bodenMate', 'hookMate', 'triangleMate', 'knightEndgame',
              'vukovicMate', 'pin', 'mateIn1', 'advancedPawn', 'short', 'veryLong', 'sacrifice', 'superGM',
              'dovetailMate', 'blindSwineMate', 'smotheredMate', 'mateIn3', 'enPassant', 'morphysMate',
              'pillsburysMate', 'mateIn4', 'castling', 'defensiveMove', 'middlegame', 'long', 'backRankMate',
              'kingsideAttack', 'promotion', 'equality', 'rookEndgame', 'doubleCheck', 'attackingF2F7', 'exposedKing',
              'mateIn5', 'doubleBishopMate', 'opening', 'intermezzo', 'mate', 'hangingPiece', 'advantage', 'mateIn2',
              'master', 'masterVsMaster', 'skewer', 'attraction', 'pawnEndgame', 'underPromotion', 'zugzwang',
              'queenEndgame', 'operaMate', 'killBoxMate', 'anastasiaMate', 'clearance', 'xRayAttack',
              'capturingDefender', 'queensideAttack', 'discoveredAttack', 'swallowstailMate']


def parse_line(line):
    _, fen, moves, rating, ratingdev, _, _, themes, _, _, _ = line.split(",")
    board = get_board_from_fen(fen)
    board.push(get_first_move(moves))
    return fen, board.epd(), moves, rating, ratingdev, themes

def get_existing_positions(out_filename):
    with open(out_filename, "r") as file:
        # Skip header
        file.readline()
        # Remember EPD's, Note that EPD's already exist
        existing_positions = {line.split(",")[1] for line in file}
    return existing_positions

'''
    Gets all possible themes from the dataset. 
'''
def get_all_themes(in_filename):
    with open(in_filename, "r") as file:
        # Skip header
        file.readline()
        all_themes = {theme for line in file for theme in parse_line(line)[5].split(" ")}
    return all_themes

def append_features(append_to, append_from):
    for k, v in append_from.items():
        append_to[k] = v

def compute_features(in_filename, out_filename, n, stockfish_path, stockfish_time_ms, skip_header=True, override=False):
    file = open(in_filename)

    if path.isfile(out_filename) and not override:
        existing_positions = get_existing_positions(out_filename)
    else:
        existing_positions = set()

    if skip_header:
        header = file.readline()

    if override or not path.isfile(out_filename):
        out_file = open(out_filename, "w")
        write_header = True
    else:
        out_file = open(out_filename, "a")
        write_header = False

    # Initialize Stockfish engine only once
    stockfish = Stockfish(path=stockfish_path)

    i = 1
    while i <= n:
        line = file.readline()
        fen, epd, moves, rating, ratingdev, themes = parse_line(line)

        if epd in existing_positions:
            continue

        print(f"{i}: {epd}")

        # Basic
        attributes = {}
        attributes["rating"] = rating
        attributes["epd"] = epd
        attributes["solution"] = moves
        attributes["rating_dev"] = ratingdev
        attributes["to_move"] = 1 if epd.split(" ")[-3] == "w" else -1

        # Stockfish attributes
        stockfish_attributes = get_stockfish_attributes(stockfish, fen, attributes["to_move"])
        if stockfish_attributes is None:
            continue

        append_features(attributes, stockfish_attributes)


        # Basic piece counts
        append_features(attributes, basic_piece_features(epd))


        # Get attributes for multiple levels
        append_features(attributes, get_level_attributes(stockfish, fen, LEVELS, stockfish_time_ms=stockfish_time_ms))


        # One hot encode themes
        # Add themes as last column
        one_hot = themes_one_hot_encoded(themes)
        append_features(attributes, one_hot)

        i += 1

        # Header fix
        if write_header:
            out_file.write(",".join([str(x) for x in attributes.keys()]) + "\n")
            write_header = False

        # Write each line to file separately
        out_file.write(",".join([str(x) for x in attributes.values()]) + "\n")

    file.close()
    out_file.close()

def themes_one_hot_encoded(themes):
    themes_one_hot = {}
    for theme in ALL_THEMES:
        themes_one_hot[theme] = 0

    for theme in themes.split(" "):
        themes_one_hot[theme] = 1

    return themes_one_hot


'''
    Attributes given by stockfish.
    Such as centipawn evaluations for top moves. 
'''
def get_stockfish_attributes(stockfish, fen, to_move):
    out = {}
    stockfish.set_fen_position(fen)
    eval = stockfish.get_evaluation()
    if eval["type"] == "cp":
        out["cp_eval"] = stockfish.get_evaluation()["value"]
    else:
        out["cp_eval"] = to_move * EVAL_BOUND

    top_moves = stockfish.get_top_moves(3, verbose=True)
    if len(top_moves) < 3:
        return None

    for i, top in enumerate(top_moves, start=1):
        if top["Mate"]:
            out[f"move{i}cp"] = to_move * EVAL_BOUND
        else:
            out[f"move{i}cp"] = top["Centipawn"]
        out["nodes"] = top["Nodes"]
        out[f"move{i}multiPV"] = top["MultiPVNumber"]
        out[f"move{i}sel_depth"] = top["SelectiveDepth"]
        w, d, l = top["WDL"].split(" ")
        out[f"move{i}w"] = w
        out[f"move{i}d"] = d
        out[f"move{i}l"] = l

    w, d, l = stockfish.get_wdl_stats()
    out["orig_w"] = w
    out["orig_d"] = d
    out["orig_l"] = l
    return out

'''
    Counts how many white and black pieces are on the board.
'''
def individual_piece_counts(epd):
    # Board part is at index 0
    board = epd.split(" ")[0]
    rows = board.split("/")

    # If character is uppercase, then it belongs to white. Otherwise its lowercase and it belongs to black.
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

'''
    Adds number of pieces on the board.
    Piece counts for white and black. 
    Material imbalance.
'''
def basic_piece_features(epd):
    feature_rows = []

    white, black = individual_piece_counts(epd)

    # Material
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

    # Individual piece count
    for p in white:
        features[f"white_{p}"] = white[p]
    for p in black:
        features[f"black_{p}"] = black[p]

    return features

if __name__ == "__main__":
    DIRECTORY = "../../datasets/"
    STOCKFISH = "D:/stockfish/stockfish-windows-x86-64-universal.exe"

    stockfish_time_ms=10
    n = 10
    compute_features(DIRECTORY + "dataset_lichess.csv", DIRECTORY + "dataset_idk.csv", n, STOCKFISH, override=False, stockfish_time_ms=stockfish_time_ms)



