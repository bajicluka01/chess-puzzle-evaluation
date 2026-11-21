"""
import the following functions:
Functions available in this module:

    find_bishop_pins(epd), find_rook_pins(epd), find_queen_pins(epd)
        Detect pins involving bishops, rooks, or queens.
        Input:  EPD string representing a chess position.
        Output: List of detected pins.

    find_all_pins(dataset_path)
        Detect all pins in every position in a dataset.
        Input:  Path to a CSV dataset containing an "epd" column.
        Output: List of pin lists, one for each position.

    find_all_forks_threatening(dataset_path)
        Detect potential knight forks by checking possible knight moves.
        Input:  Path to a CSV dataset containing an "epd" column.
        Output: List of fork lists, one for each position.

    find_all_forks_current(dataset_path)
        Detect current knight forks in each position.
        Input:  Path to a CSV dataset containing an "epd" column.
        Output: List of fork lists, one for each position.
"""

import pandas as pd
import chess
from plots import plot_boards

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

def absolute_pins(pinned_piece, epd):
    """
        Prints absolute pins (on the king)
    """

    board = chess.Board()
    board.set_epd(epd)

    for square, piece in board.piece_map().items():
        if piece.piece_type == pinned_piece:
            if board.is_pinned(piece.color, square):
                print(f"{pinned_piece} pinned on {chess.square_name(square)}")

def find_bishop_pins(epd):
    """
        Finds and returns all the bishop pins on the current board 
    """

    board = chess.Board()
    board.set_epd(epd)

    pins = []
    number_of_bishops = 0
    for _, (bishop_square, bishop) in enumerate(board.piece_map().items()):
        # if it's not a bishop, pass
        if bishop.piece_type != chess.BISHOP:
            continue
        number_of_bishops += 1

        # go through the squares this bishop is attacking
        for attacked_square in board.attacks(bishop_square):
            # check what is on the square
            attacked_piece = board.piece_at(attacked_square)

            # None ... empty square
            # also must be of opposite color
            if attacked_piece is None or attacked_piece.color == bishop.color:
                continue

            # also exclude pawns
            if attacked_piece.piece_type == chess.PAWN:
                continue

            # now we need to check if there is any piece behind it 

            # we find in which direction we will be increasing or decreasing file and row
            file_direction = chess.square_file(attacked_square) - chess.square_file(bishop_square)
            file_direction = int(file_direction / abs(file_direction))
            row_direction = chess.square_rank(attacked_square) - chess.square_rank(bishop_square)
            row_direction = int(row_direction / abs(row_direction))

            direction = (file_direction, row_direction)

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
                    # check if it's opponent piece
                    if current_piece.color == bishop.color:
                        break

                    # check if it's a pawn, we exclude them
                    if current_piece.piece_type == chess.PAWN:
                        break

                    # now the current piece either has higher value than the attacking one
                    # (consider also for equal, depending how we define a pin)
                    # or is undefendend 

                    """
                    if PIECE_VALUES[current_piece.piece_type] > PIECE_VALUES[bishop.piece_type]:
                        print(f"        piece behind it {current_piece} has higher than the attacking piece {bishop}")
                    print(f"        piece behind the pinned one has these defenders:\n{board.attackers(current_piece.color, current_square)}")
                    """

                    if (
                        PIECE_VALUES[current_piece.piece_type] > PIECE_VALUES[bishop.piece_type]
                        or not 
                        board.attackers(current_piece.color, current_square)
                    ):
                        pins.append({
                            "bishop_square": chess.square_name(bishop_square),
                            "pinned_piece_square": chess.square_name(attacked_square),
                            "piece_behind_square": chess.square_name(current_square),
                            "pinned_piece": attacked_piece.symbol(),
                            "piece_behind": current_piece.symbol(),
                        })

                    break

                file += direction[0]
                row += direction[1]
    return pins

def find_rook_pins(epd):
    """
        Finds and returns all the rook pins on the current board 
    """

    board = chess.Board()
    board.set_epd(epd)

    pins = []
    number_of_rooks = 0
    for _, (rook_square, rook) in enumerate(board.piece_map().items()):
        # if it's not a rook, pass
        if rook.piece_type != chess.ROOK:
            continue
        number_of_rooks += 1

        # go through the squares this rook is attacking
        for attacked_square in board.attacks(rook_square):
            # check what is on the square
            attacked_piece = board.piece_at(attacked_square)

            # None ... empty square
            # also must be of opposite color
            if attacked_piece is None or attacked_piece.color == rook.color:
                continue

            # also exclude pawns
            if attacked_piece.piece_type == chess.PAWN:
                continue

            # now we need to check if there is any piece behind it 

            # we find in which direction we will be increasing or decreasing file and row
            file_direction = chess.square_file(attacked_square) - chess.square_file(rook_square)
            if file_direction != 0:     # if it's on the same file
                file_direction = int(file_direction / abs(file_direction))
            row_direction = chess.square_rank(attacked_square) - chess.square_rank(rook_square)
            if row_direction != 0:      # if it's on the same row 
                row_direction = int(row_direction / abs(row_direction))

            direction = (file_direction, row_direction)

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
                    # check if it's opponent piece
                    if current_piece.color == rook.color:
                        break

                    # check if it's a pawn, we exclude them
                    if current_piece.piece_type == chess.PAWN:
                        break

                    # now the current piece either has higher value than the attacking one
                    # (consider also for equal, depending how we define a pin)
                    # or is undefendend 

                    """
                    if PIECE_VALUES[current_piece.piece_type] > PIECE_VALUES[rook.piece_type]:
                        print(f"        piece behind it {current_piece} has higher than the attacking piece {rook}")
                    print(f"        piece behind the pinned one has these defenders:\n{board.attackers(current_piece.color, current_square)}")
                    """

                    if (
                        PIECE_VALUES[current_piece.piece_type] > PIECE_VALUES[rook.piece_type]
                        or not 
                        board.attackers(current_piece.color, current_square)
                    ):
                        pins.append({
                            "rook_square": chess.square_name(rook_square),
                            "pinned_piece_square": chess.square_name(attacked_square),
                            "piece_behind_square": chess.square_name(current_square),
                            "pinned_piece": attacked_piece.symbol(),
                            "piece_behind": current_piece.symbol(),
                        })

                    break

                file += direction[0]
                row += direction[1]
    return pins

def find_queen_pins(epd):
    """
        Finds and returns all the queen pins on the current board 
    """

    board = chess.Board()
    board.set_epd(epd)

    pins = []
    number_of_queens = 0
    for _, (queen_square, queen) in enumerate(board.piece_map().items()):
        # if it's not a queen, pass
        if queen.piece_type != chess.QUEEN:
            continue
        number_of_queens += 1

        # go through the squares this queen is attacking
        for attacked_square in board.attacks(queen_square):
            # check what is on the square
            attacked_piece = board.piece_at(attacked_square)

            # None ... empty square
            # also must be of opposite color
            if attacked_piece is None or attacked_piece.color == queen.color:
                continue

            # also exclude pawns
            if attacked_piece.piece_type == chess.PAWN:
                continue

            # now we need to check if there is any piece behind it 

            # we find in which direction we will be increasing or decreasing file and row
            file_direction = chess.square_file(attacked_square) - chess.square_file(queen_square)
            if file_direction != 0:     # if it's on the same file
                file_direction = int(file_direction / abs(file_direction))
            row_direction = chess.square_rank(attacked_square) - chess.square_rank(queen_square)
            if row_direction != 0:      # if it's on the same row 
                row_direction = int(row_direction / abs(row_direction))

            direction = (file_direction, row_direction)

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
                    # check if it's opponent piece
                    if current_piece.color == queen.color:
                        break

                    # check if it's a pawn, we exclude them
                    if current_piece.piece_type == chess.PAWN:
                        break

                    # now the current piece either has higher value than the pinned one
                    # (consider also for equal, depending how we define a pin)
                    # or is undefendend 

                    """
                    if PIECE_VALUES[current_piece.piece_type] > PIECE_VALUES[queen.piece_type]:
                        print(f"        piece behind it {current_piece} has higher than the attacking piece {queen}")
                    print(f"        piece behind the pinned one has these defenders:\n{board.attackers(current_piece.color, current_square)}")
                    """

                    if (
                        PIECE_VALUES[current_piece.piece_type] > PIECE_VALUES[queen.piece_type]
                        or not 
                        board.attackers(current_piece.color, current_square)
                    ):
                        pins.append({
                            "queen_square": chess.square_name(queen_square),
                            "pinned_piece_square": chess.square_name(attacked_square),
                            "piece_behind_square": chess.square_name(current_square),
                            "pinned_piece": attacked_piece.symbol(),
                            "piece_behind": current_piece.symbol(),
                        })

                    break

                file += direction[0]
                row += direction[1]
    return pins

def find_all_pins(df_path):
    """
        Function that returns all pins (created by queen, rook and bishop) from given dataset.
        Dataset needs to have column "epd".
        Returns it as a list corresponding to dataset:
            [
                [bishop_pin1, rook_pin1, queen_pin1],       # EPD 1
                [bishop_pin1, bishop_pin2, queen_pin1],     # EPD 2
                [],                                         # EPD 3: no pins
                ...
            ]
        Each pin is a dictionary describing a piece that is attacking (queen, rook or bishop),
        which piece it's attacking, and which piece is behind it. For example:
            {
                "rook_square": "a1",
                "pinned_piece_square": "a5",
                "piece_behind_square": "a8",
                "pinned_piece": "n",
                "piece_behind": "q",
            }
    """

    """
        General definition of a pin:
            a piece attacks a piece of the opponent 
            -> if the attacked piece has a higher valued piece of itself in the back, it is considered pinned
            -> if the attacked piece has equal or lower valued piece of itself in the back, this piece needs to 
            be undefeneded for attacked piece to be considered pinned
    """

    df = pd.read_csv(df_path)

    all_pins = []
    for epd in df["epd"]:
        pins = []

        pins.extend(find_bishop_pins(epd))
        pins.extend(find_rook_pins(epd))
        pins.extend(find_queen_pins(epd))

        all_pins.append(pins)

    return all_pins

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

def find_all_forks_threatening(df_path):
    """
        Function that returns all potential forks (created by a knight) from given dataset.
        Dataset needs to have column "epd".
        Returns it as a list corresponding to dataset:
            [
                [fork1, fork2],         # EPD 1
                [fork1],                # EPD 2
                [],                     # EPD 3: no potential forks
                ...
            ]
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

    df = pd.read_csv(df_path)

    all_forks = []
    for epd in df["epd"]:
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

                # simulate the knight move (because if we capture a piece with this knight)

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
        
        all_forks.append(forks)
    return all_forks

def find_all_forks_current(df_path):
    """
        Function that returns all forks (created by a knight) from given dataset.
        Dataset needs to have column "epd".
        Returns it as a list corresponding to dataset:
            [
                [fork1, fork2],         # EPD 1
                [fork1],                # EPD 2
                [],                     # EPD 3: no forks
                ...
            ]
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

    df = pd.read_csv(df_path)

    all_forks = []
    for epd in df["epd"]:
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
    
        all_forks.append(forks)
    return all_forks

if __name__ == "__main__":
    dataset_path = "./datasets/artificle_tests.csv"

    print("POTENTIAL FORKS")
    forks = find_all_forks_threatening(dataset_path)
    for fork in forks:
        print(fork)
        print(f"total number of forks in this position: {len(fork)}")
        print()

    print("CURRENT FORKS")
    forks = find_all_forks_current(dataset_path)
    print(forks)
    for fork in forks:
        print(fork)
        print(f"total number of forks in this position: {len(fork)}")
        print()
        