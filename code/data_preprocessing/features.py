import os
import os.path as path
import pandas
import chess
from stockfish import Stockfish
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
    "K": 0, "k": 0      #TODO: why is king 0?
}
PIECE_VALUES_PINS = {
    "P": 1, "p": 1,
    "N": 3, "n": 3,
    "B": 3, "b": 3,
    "R": 5, "r": 5,
    "Q": 9, "q": 9,
    "K": 100, "k": 100
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

USEFUL_THEMES = ['discoveredCheck', 'deflection', 'bishopEndgame', 'quietMove', 'oneMove',
              'interference', 'trappedPiece', 'endgame', 'fork', 
              'collinearMove', 'queenRookEndgame', 'knightEndgame',
              'pin', 'advancedPawn', 'sacrifice', 
              'enPassant', 
              'castling', 'defensiveMove', 'middlegame', 'long',
              'kingsideAttack', 'promotion', 'equality', 'rookEndgame', 'doubleCheck', 'attackingF2F7', 'exposedKing',
              'opening', 'intermezzo', 'mate', 'hangingPiece', 'advantage', 
              'skewer', 'attraction', 'pawnEndgame', 'underPromotion', 'zugzwang',
              'queenEndgame', 'clearance', 'xRayAttack',
              'capturingDefender', 'queensideAttack', 'discoveredAttack']

def get_first_move(moves):
    return chess.Move.from_uci(moves.split(" ")[0])

def get_board_from_fen(fen):
    return chess.Board(fen)

def parse_line(line):
    _, fen, moves, rating, ratingdev, _, _, themes, _, _, _ = line.split(",")
    board = chess.Board(fen)
    board.push(chess.Move.from_uci(moves.split(" ")[0]))
    return fen, board.epd(), moves, rating, ratingdev, themes

def get_existing_positions(out_filename):
    with open(out_filename, "r") as file:
        # Skip header
        file.readline()
        # Remember EPD's, Note that EPD's already exist
        existing_positions = {line.split(",")[1] for line in file}
    return existing_positions

def find_specific_pin(epd, pinning_piece):

    """
        Finds and returns all pins of a specific piece (queen, rook or bishop) on the current board 
    """

    board = chess.Board()
    board.set_epd(epd)

    pins = []
    for piece_square, piece in board.piece_map().items():

        # among all pieces find the pinning piece
        if piece.piece_type != pinning_piece:
            continue

        # go through the squares this piece is attacking
        for attacked_square in board.attacks(piece_square):

            # find a square that has opponents piece (not pawn) on it
            attacked_piece = board.piece_at(attacked_square)
            if attacked_piece is None or attacked_piece.color == piece.color or attacked_piece.piece_type == chess.PAWN:
                continue

            # now we need to check if there is any piece behind it 

            # we find in which direction we will be increasing or decreasing file and row
            file_direction = chess.square_file(attacked_square) - chess.square_file(piece_square)
            if file_direction != 0:     # if it's on the same file
                file_direction = int(file_direction / abs(file_direction))
            row_direction = chess.square_rank(attacked_square) - chess.square_rank(piece_square)
            if row_direction != 0:      # if it's on the same row 
                row_direction = int(row_direction / abs(row_direction))

            direction = (
                file_direction,
                row_direction
            )

            file, row = (
                chess.square_file(attacked_square) + direction[0],
                chess.square_rank(attacked_square) + direction[1],
            )

            # now we check until we find a piece or until we come to the end of the board
            while 0 <= file < 8 and 0 <= row < 8:

                current_square = chess.square(file, row)
                current_piece = board.piece_at(current_square)

                # until we find a piece
                if current_piece is not None:

                    # check if it's opponent piece or a pawn
                    if current_piece.color == piece.color or current_piece.piece_type == chess.PAWN:
                        break


                    # now the current piece either has higher value than the pinned one or is undefendend 
                    if (
                        PIECE_VALUES_PINS[current_piece.symbol()] > PIECE_VALUES_PINS[piece.symbol()]
                        or not 
                        board.attackers(current_piece.color, current_square)
                    ):
                        pins.append({
                            "pinning_piece": piece.symbol(),
                            "pinning_piece_square": chess.square_name(piece_square),
                            "pinned_piece": attacked_piece.symbol(),
                            "pinned_piece_square": chess.square_name(attacked_square),
                            "piece_behind": current_piece.symbol(),
                            "piece_behind_square": chess.square_name(current_square),
                        })

                    break

                file += direction[0]
                row += direction[1]

    return pins



def find_all_pins(epd):

    """
        Function that returns all pins (created by queen, rook and bishop) from given epd.

        Returns it as a list corresponding to dataset:
            [bishop_pin_1, ... , bishop_pin_p, rook_pin_1, ... , rook_pin_q, queen_pin_1, ... , queen_pin_r,]
                    
        Each pin is a dictionary describing a piece that is attacking (queen, rook or bishop),
        which piece it's attacking, and which piece is behind it. For example:
            {
                "pinning_piece": R,
                "pinning_piece_square": "a1",
                "pinned_piece": "n",
                "pinned_piece_square": "a5",
                "piece_behind": "q",
                "piece_behind_square": "a8",
            }
    """

    """
        General definition of a pin:
            a piece attacks a piece of the opponent 
            -> if the attacked piece has a higher valued piece of itself in the back, it is considered pinned
            -> if the attacked piece has equal or lower valued piece of itself in the back, this piece needs to 
            be undefeneded for attacked piece to be considered pinned
    """


    pins = []

    pins.extend(find_specific_pin(epd, chess.BISHOP))
    pins.extend(find_specific_pin(epd, chess.ROOK))
    pins.extend(find_specific_pin(epd, chess.QUEEN))

    return pins


def forks_from_current_square(board, knight_square, knight):
    """
        Auxiliary function that finds current forks from a given square 
    """

    # find the opposite pieces that can be attacked by a knight from this square
    forked_pieces = []
    for attacked_square in chess.SquareSet(chess.BB_KNIGHT_ATTACKS[knight_square]):

        potentially_attacked_piece = board.piece_at(attacked_square)

        # if there is a piece and it's opponents piece, store it 
        if potentially_attacked_piece is not None and potentially_attacked_piece.color != knight.color:
            forked_pieces.append({
                "piece": potentially_attacked_piece.symbol(),
                "square": chess.square_name(attacked_square),
            })

    return forked_pieces


def find_all_forks_threatening(epd):
    """
        Function that returns all potential forks (created by a knight) from given epd.

        Returns it as a list of forks:
            [fork_1, fork_2, ... , fork_n]

        Each fork is a dictionary describing a knight move and the opponent pieces
        that would be attacked by the knight after making that move. For example:
            {
                "knight_color": "white",
                "knight_from": "c3",
                "knight_to": "d5",
                "forked_pieces": [
                    {
                        "piece": "q",
                        "square": "e7",
                    },
                    {
                        "piece": "r",
                        "square": "f6",
                    }
                ],
            }
    """

    """
        A potential fork is defined as a knight move to a square from which the
        knight would attack at least two opponent pieces. It does not matter whether 
        the piece is defended or not, it also takes into account pawns. It returns 
        possible forks from both sides. 
    """

    board = chess.Board()
    board.set_epd(epd)

    forks = []


    for knight_square, knight in board.piece_map().items():

        # if the piece is not a knight, continue
        if knight.piece_type != chess.KNIGHT:
            continue

        # go through all possible jumps of this knight
        for jump_square in board.attacks(knight_square):

            # check whether this square is occupied by their own piece
            jump_square_occupation = board.piece_at(jump_square)
            if jump_square_occupation is not None and jump_square_occupation.color == knight.color:
                continue

            # find the opposite pieces that can be attacked by a knight from this square
            forked_pieces = forks_from_current_square(board, jump_square, knight)

            # if there is more than one piece that can be forked, we store this as fork
            if len(forked_pieces) >= 2:
                forks.append({
                    "knight_color": (
                        "white" if knight.color == chess.WHITE else "black"
                    ),
                    "knight_from": chess.square_name(knight_square),
                    "knight_to": chess.square_name(jump_square),
                    "forked_pieces": forked_pieces,
                })

    return forks



def find_all_forks_current(epd):
    """
        Function that returns all forks (created by a knight) from given epd.
        
        Returns it as a list of forks:
            [fork_1, fork_2, ... , fork_n]

        Each fork is a dictionary describing the knight and the opponent pieces
        currently attacked by that knight. For example:
            {
                "knight_color": "white",
                "knight_square": "d5",
                "forked_pieces": [
                    {
                        "piece": "q",
                        "square": "e7",
                    },
                    {
                        "piece": "r",
                        "square": "f6",
                    }
                ],
            }
    """

    """
        A current fork is defined as a knight that is currently attacking at least
        two opponent pieces from its current square. It does not matter whether 
        the piece is defended or not, it also takes into account pawns. It returns 
        forks from both sides. 
    """


    board = chess.Board()
    board.set_epd(epd)

    forks = []


    for knight_square, knight in board.piece_map().items():

        # if the piece is not a knight, continue
        if knight.piece_type != chess.KNIGHT:
            continue

        # find the opposite pieces that can be attacked by a knight from this square
        forked_pieces = forks_from_current_square(board, knight_square, knight)

        # if there is more than one piece that can be forked, we store this as fork
        if len(forked_pieces) >= 2:
            forks.append({
                "knight_color": (
                    "white" if knight.color == chess.WHITE else "black"
                ),
                "knight_square": chess.square_name(knight_square),
                "forked_pieces": forked_pieces,
            })

    return forks


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

        # Get attributes for pins and forks
        appendix = {}
        appendix["current_forks"] = len(find_all_forks_current(epd))
        appendix["threatening_forks"] = len(find_all_forks_threatening(epd))
        appendix["all_pins"] = len(find_all_pins(epd))
        print(appendix)
        append_features(attributes, appendix)



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
    for theme in USEFUL_THEMES:
        themes_one_hot[theme] = 0

    for theme in themes.split(" "):
        if theme in USEFUL_THEMES:
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
    DIRECTORY = "./datasets/"
    STOCKFISH = "C:/engines/stockfish/stockfish-windows-x86-64-universal"

    stockfish_time_ms=10
    n = 10
    compute_features(DIRECTORY + "dataset_lichess.csv", DIRECTORY + "dataset_idk.csv", n, STOCKFISH, override=False, stockfish_time_ms=stockfish_time_ms)
