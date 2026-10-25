import chess
from engines import otter_engine_func

EVAL_BOUND = 1500
POSSIBLE_MOVES_LIMIT = 3

PIECE_VALUES_PINS = {
    "P": 1, "p": 1,
    "N": 3, "n": 3,
    "B": 3, "b": 3,
    "R": 5, "r": 5,
    "Q": 9, "q": 9,
    "K": 100, "k": 100
}


def chebyshev_distance(move):
    from_sq = move.from_square
    to_sq = move.to_square

    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)

    to_file = chess.square_file(to_sq)
    to_rank = chess.square_rank(to_sq)

    return max(abs(from_file - to_file), abs(from_rank - to_rank))


def dfs_possible_moves(position, total_num_of_levels, current_level, counts):
    if position.is_game_over():
        return

    # Uncomment if level is more than 3 then i guess this is not so stupid to add
    if counts[current_level] > POSSIBLE_MOVES_LIMIT:
       return

    moves = list(position.legal_moves)

    # Count total number of moves
    counts[current_level] += len(moves)

    # Base case
    if current_level >= total_num_of_levels:
        return

    # Go through moves and execute them
    current_level += 1
    for move in moves:
        new_pos = position.copy()
        new_pos.push(move)

        dfs_possible_moves(new_pos, total_num_of_levels, current_level, counts)


def check_pawn_attack(board, pawn, pawn_square, attacker_piece, attacker_square):
    """
        Function that returns 1 the given epd is a fork/pin or not for a special cases.

        In case of fork:
            If two pieces are attacked at the same time and one of them is pawn, we consider this as a special case. 
            If the pawn is not defended, we do consider this as a fork. 
            If the pawn is defended, in general we do not consider this as a fork, except if these two conditions hold
                - when a pawn is defended only by a king 
                - this pawn is also attacked by some other piece than the one that did the fork

        Similar logic applies to a case of a pin where we have something like piece -> opponent_piece -> opponent_pawn 
        (However, not piece -> opponent_pawn -> opponent_piece. This is automatically considered a pin)
    """

    
    print(f"pawn {pawn} on square {chess.square_name(pawn_square)}")
    print(f"attacker {attacker_piece} on square {chess.square_name(attacker_square)}")


    # 1. if the pawn is not defended, return 1
    defenders_of_pawn_board = board.attackers(pawn.color, pawn_square)
    defenders_of_pawn = [chess.square_name(sq) for sq in defenders_of_pawn_board]
    print("defenders of the pawn:", defenders_of_pawn)

    if len(defenders_of_pawn) == 0:
        print("pawn has no defenders!")
        return 1

    # 2. if it is defended by 2 pieces or more, return 0
    if len(defenders_of_pawn) > 1:
        print("pawn has more than one defender!")
        return 0

    # 3a. now we check if its only defending is a king
    if board.piece_at(chess.parse_square(defenders_of_pawn[0])).piece_type == chess.KING:
        print("pawn has only one defender and that is a king!")

        # 3b. now we check whether this pawn is attacked by one more piece (that is not attacker piece)
        other_attackers_board = board.attackers(not pawn.color, pawn_square)
        other_attackers = [chess.square_name(sq) for sq in other_attackers_board]
        print(f"other attackers of a pawn: {other_attackers}")
        if any(sq != attacker_square for sq in other_attackers):
            print("Pawn is also attacked by another piece!")
            return 1
        else:
            print("Pawn is not attacked by any other piece.")
            return 0

    # 4. if the only defending piece is not king, return 0
    return 0



def find_specific_pin(epd, pinning_piece):

    """
        Finds and returns all pins of a specific piece (queen, rook or bishop) on the current board
    """

    board = chess.Board()
    board.set_epd(epd)
    #print(board)

    pins = []
    for piece_square, piece in board.piece_map().items():

        # among all pieces find the pinning piece
        if piece.piece_type != pinning_piece:
            continue

        # go through the squares this piece is attacking
        for attacked_square in board.attacks(piece_square):

            # find a square that has opponents piece on it
            attacked_piece = board.piece_at(attacked_square)
            if attacked_piece is None or attacked_piece.color == piece.color:
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
                    if current_piece.color == piece.color:
                        break

                    # if it's a pawn it's a special case 
                    if current_piece.piece_type == chess.PAWN:

                        its_a_pin = check_pawn_attack(board, pawn=current_piece, pawn_square=current_square, attacker_piece=piece, attacker_square=piece_square)
                        if its_a_pin:
                            pins.append({
                                "pinning_piece": piece.symbol(),
                                "pinning_piece_square": chess.square_name(piece_square),
                                "pinned_piece": attacked_piece.symbol(),
                                "pinned_piece_square": chess.square_name(attacked_square),
                                "piece_behind": current_piece.symbol(),
                                "piece_behind_square": chess.square_name(current_square),
                            })

                        break

                    # we check whether the piece behind is defended or not 
                    # also, it doesn't count if it's defended by the pinned piece
                    defenders_board = board.attackers(current_piece.color, current_square)
                    defenders = [chess.square_name(sq) for sq in defenders_board]
                    #print(f"attacking piece: {piece} on square: {chess.square_name(piece_square)}")
                    #print(f"pinned piece {attacked_piece} on square {chess.square_name(attacked_square)}")
                    #print(f"defenders: {defenders}")

                    # exclude pinned piece from defenders
                    defenders = [sq for sq in defenders if sq != chess.square_name(attacked_square)]

                    # if there's no other defender, this is for sure a pin
                    is_defended = len(defenders) > 0

                    #print(f"filtered defenders: {defenders}")
 

                    # now the current piece either has higher value than the pinned one or is undefendend
                    if (
                        PIECE_VALUES_PINS[current_piece.symbol()] > PIECE_VALUES_PINS[piece.symbol()]
                        or not
                        is_defended
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
    1 Meaningful(L) -> for 3 levels
    2 PossibleMoves(L)
    3 AllPossibleMoves
    4 Branching(L) -> for 3 levels (but 1 excluded)
    5 AverageBranching -> maybe not even useful since we then only have two levels lol
    6 NarrowSolution(L)
    7 AllNarrowSolutions
    8 TreeSize
    9 MoveRatio
    11 Distance(L)
    12 SumDistance
    13 AverageDistance
    14 Pieces(L)
    15 AllPiecesInvolved
    17 WinningNoCheckmate
'''
def get_level_attributes(stockfish, fen, levels, otter_engine, stockfish_time_ms):
    data = {}

    #print("LEVEL 1")

    starting_position = chess.Board(fen)
    winning_side = starting_position.turn

    # pins and forks at level 1
    starting_epd = starting_position.epd()
    current_forks = find_all_forks_current(starting_epd)
    threatening_forks = find_all_forks_threatening(starting_epd)
    all_pins = find_all_pins(starting_epd)

    #print(current_forks)
    #print(threatening_forks)
    #print(all_pins)

    data["current_forks_L1"] = len(current_forks)
    data["threatening_forks_L1"] = len(threatening_forks)
    data["all_pins_L1"] = len(all_pins)

    #meaningful_moves, winning_but_not_mating = find_meaningful_moves(stockfish, starting_position, winning_side, level_one=True, stockfish_time_ms=stockfish_time_ms)
    #print(f"meaningful moves: {meaningful_moves}, winning without mate: {winning_but_not_mating}")
    meaningful_moves, winning_but_not_mating = find_otter_moves(stockfish, otter_engine, starting_position)
    #print(f"otter moves: {meaningful_moves}, winning without mate: {winning_but_not_mating}")
    #print(f"number of combs: {len(meaningful_moves)}")
    #print(f"combs:\n{meaningful_moves}")

    data["meaningful_L1"] = len(meaningful_moves)
    data["winning_no_mate"] = 1 if winning_but_not_mating else 0
    data["narrow_L1"] = 1 if len(meaningful_moves) == 1 else 0
    data["distance_L1"] = sum(chebyshev_distance(m) for m in meaningful_moves)

    data["all_narrow_solutions"] = data["narrow_L1"]
    data["tree_size"] = data["meaningful_L1"]
    data["sum_distance"] = data["distance_L1"]

    all_piece_types = set()
    piece_types = set()
    for move in meaningful_moves:
        piece = starting_position.piece_at(move.from_square)
        if piece is not None:
            piece_types.add(piece.piece_type)
            all_piece_types.add(piece.piece_type)
    data["pieces_L1"] = len(piece_types)

    meaningful_moves_combinations = [[move] for move in meaningful_moves]

    for level in range(2, levels+1):
        #print(f"LEVEL {level}")
        data[f"meaningful_L{level}"] = 0
        data[f"possible_L{level}"] = 0
        data[f"narrow_L{level}"] = 0
        data[f"distance_L{level}"] = 0
        data[f"pieces_L{level}"] = 0

        new_combinations = []
        narrow_count = 0
        distance_count = 0
        pieces_count = set()
        current_forks_count = 0
        threatening_forks_count = 0
        all_pins_count = 0

        next_position = starting_position.copy()

        #print("meaningful moves combinations:")
        #print(meaningful_moves_combinations)
        # Go through all combinations
        for meaningful_move_list in meaningful_moves_combinations:
            new_position = next_position.copy()
            for meaningful_move in meaningful_move_list:
                new_position.push(meaningful_move)

            position_epd = new_position.epd()
            #print(f"za meaningful potezo: {meaningful_move_list}")
            current_forks = find_all_forks_current(position_epd)
            threatening_forks = find_all_forks_threatening(position_epd)
            all_pins = find_all_pins(position_epd)

            #print(current_forks)
            #print(threatening_forks)
            #print(all_pins)

            current_forks_count += len(find_all_forks_current(position_epd))
            threatening_forks_count += len(find_all_forks_threatening(position_epd))
            all_pins_count += len(find_all_pins(position_epd))

            #print(f"new position:\n{new_position}")
            #new_moves, _ = find_meaningful_moves(stockfish, new_position, winning_side, level_one=False, stockfish_time_ms=stockfish_time_ms)
            #print(f"meaningful moves: {new_moves}")
            #print(f"new position:\n{new_position}")
            new_moves, _ = find_otter_moves(stockfish, otter_engine, new_position)
            #print(f"otter moves: {new_moves}")
            distance_count += sum(chebyshev_distance(m) for m in new_moves)
            narrow_count += 1 if len(new_moves) == 1 else 0

            for move in new_moves:
                new_combinations.append(meaningful_move_list + [move])

                # Also check the pieces in the meantime
                piece = new_position.piece_at(move.from_square)
                if piece is not None:
                    pieces_count.add(piece.piece_type)
                    all_piece_types.add(piece.piece_type)
        
        meaningful_moves_combinations = new_combinations
        #print(f"number of combs: {len(new_combinations)}")
        #print(f"combs:\n{new_combinations}")
        data[f"meaningful_L{level}"] = len(new_combinations)
        data[f"narrow_L{level}"] = narrow_count
        data[f"distance_L{level}"] = distance_count
        data[f"pieces_L{level}"] = len(pieces_count)

        data["all_narrow_solutions"] += data[f"narrow_L{level}"]
        data["tree_size"] += data[f"meaningful_L{level}"]
        data["sum_distance"] += data[f"distance_L{level}"]

        data[f"current_forks_L{level}"] = current_forks_count
        data[f"threatening_forks_L{level}"] = threatening_forks_count
        data[f"all_pins_L{level}"] = all_pins_count


    data["all_pieces_involved"] = len(all_piece_types)

    branchings = []
    for level in range(2, levels + 1):
        prev_val = data[f"meaningful_L{level-1}"]
        curr_val = data[f"meaningful_L{level}"]

        branching = curr_val / prev_val if prev_val > 0 else 0.0
        data[f"branching_L{level}"] = branching
        branchings.append(branching)

    data["avg_branching"] = sum(branchings) / len(branchings) if branchings else 0.0

    # 2 PossibleMoves(L)
    counts = {l: 0 for l in range(1, levels + 1)}

    dfs_possible_moves(starting_position, levels, 1, counts)

    for level in range(1, levels + 1):
        data[f"possible_L{level}"] = counts[level]

    # 3 AllPossibleMoves
    data["all_possible_moves"] = sum(counts.values())

    # 9 MoveRatio
    for level in range(1, levels + 1):
        meaningful = f"meaningful_L{level}"
        possible = f"possible_L{level}"
        ratio = f"move_ratio_L{level}"

        data[ratio] = data[meaningful] / data[possible] if data[possible] != 0 else 0.0

    # 13 AverageDistance
    data["avg_distance"] = data["sum_distance"] / data["tree_size"] if data["tree_size"] else 0.0

    return data


def find_otter_moves(stockfish, otter_engine, position):

    meaningful_moves = []
    winning_but_not_mating = 1

    if position.is_game_over():
        return [], 1

    # 1. find whether there is a mate
    stockfish.set_fen_position(position.fen())
    top_eval_stockfish = stockfish.get_top_moves(1)[0]
    #print(f"stockfish top eval: {top_eval_stockfish}")

    # if yes store winning and mating 
    if top_eval_stockfish["Mate"] is not None:
        winning_but_not_mating = 0

    # 2. extract otter moves and filter them with some threshold
    otter_data = otter_engine_func(position.epd(), otter_engine)
    #print(otter_data)

    for item in otter_data["moves"]:

        move = item["move"]
        probability = item["probability"]
        #print(f"move={move}, p={probability}")

        # threshold
        if probability < 0.1:
            break

        meaningful_moves.append(chess.Move.from_uci(move))

    stockfish_move = chess.Move.from_uci(top_eval_stockfish["Move"])
    if stockfish_move not in meaningful_moves:
        meaningful_moves.append(stockfish_move)

    #print(f"Final meaningful moves: {meaningful_moves}")

    return meaningful_moves, winning_but_not_mating
    





def find_meaningful_moves(stockfish, position, winning_side, level_one=False, stockfish_time_ms=100):

    
    if position.is_game_over():
        return [], []

    # pos.turn = True if white or False if black

    moves = list(position.legal_moves)
    #print(f"legal moves:\n{moves}")
    #print(f"position: {position}")
    evaluations = get_stockfish_evaluation(stockfish, position, moves, stockfish_time_ms=stockfish_time_ms)
    #print(evaluations)

    f = max if position.turn else min
    best_eval = f(e for _, e, _, _ in evaluations)

    meaningful_moves = []
    winning_but_not_mating = 0

    mate_moves = []

    x = 1 if winning_side == position.turn else -1
    y = 1 if winning_side else -1

    best_mate = None
    for move, eval_cp, is_mate, mate_in in evaluations:
        # If y is  1, then its white => mate_in has to be > 0
        # If y is -1, then its black => mate_in has to be < 0
        #print(f"move: {move}, eval_cp: {eval_cp}, is_mate: {is_mate}, mate_in: {mate_in}")
        #print(f"y = {y}")
        if is_mate and (y * mate_in) >= 0:
            # Same logic as above, but you have to check if its players turn or not
            if best_mate is None or (y * x * mate_in) < (y * x * best_mate):
                best_mate = mate_in
                mate_moves = [move]

            elif mate_in == best_mate:
                mate_moves.append(move)

    # If some mate moves are found, finish
    #print(f"mate moves: {mate_moves}")
    if mate_moves:
        #print("!")
        return mate_moves, winning_but_not_mating

    # If no mate moves are found, continue to search for good enough moves
    # Good enough moves are dictated by constants W and M
    W = 200
    M = 50
    for move, eval_cp, is_mate, mate_in in evaluations:
        if winning_side == position.turn:
            if y * eval_cp >= W:
                meaningful_moves.append(move)
                # Win and mate?
                if level_one:
                    winning_but_not_mating += 0 if abs(eval_cp) == EVAL_BOUND else 1
        elif abs(best_eval - eval_cp) <= M:
            # Move is close enough to best move
            meaningful_moves.append(move)

    return meaningful_moves, winning_but_not_mating



def get_stockfish_evaluation(stockfish, position, moves, stockfish_time_ms=100):
    turn_next_move = -1 if position.turn else 1
    evaluations = []
    mate_in = 0

    for move in moves:
        new_position = position.copy()
        new_position.push(move)

        stockfish.set_fen_position(new_position.fen())
        raw_eval = stockfish.get_evaluation(searchtime=stockfish_time_ms)
        wdl = stockfish.get_wdl_stats()

        if raw_eval["type"] == "cp":
            # This is centipawn evaluation from stockfish
            is_mate = 0
            eval_cp = turn_next_move * raw_eval["value"]
        else:
            # This is mate in n moves
            is_mate = 1

            if raw_eval["value"] == 0:
                # Mate on the board
                mate_in = 0
                eval_cp = turn_next_move * EVAL_BOUND
            else:
                # Mate in n moves
                mate_in = turn_next_move * abs(raw_eval["value"])
                mate_in = mate_in if wdl[0] > wdl[2] else -mate_in

                # Check who is winning (white or black)
                # Example of wdl [W, D, L]: [402, 0, 598] -> in this case black is winning
                eval_cp = turn_next_move * EVAL_BOUND
                eval_cp = eval_cp if wdl[0] > wdl[2] else -eval_cp

        evaluations.append((move, eval_cp, is_mate, mate_in))

    return evaluations


if __name__ == "__main__":

    """
    import time
    from stockfish import Stockfish
    from otter_chess import OtterModel

    STOCKFISH = "/usr/games/stockfish"
    stockfish = Stockfish(path=STOCKFISH)
    otter_engine = OtterModel(device="cpu")

    # 2, 3
    epds = ["5rk1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 b - -", "r4rk1/pp3ppp/8/2bqp2Q/3N4/P2Bn2P/1PPN1P2/R1B2RK1 b - -", "r2qr1k1/b1p2ppp/p5n1/P1p1p3/4P1n1/B2P2Pb/3NBP1P/RN1QR1K1 w - -"]
    levels = 5

    
    for epd in epds:
        epd_start = time.perf_counter()
        starting_position = chess.Board(epd)
        winning_side = starting_position.turn
        fen = starting_position.fen()

        print(epd)
        if winning_side:
            print(f"white is winning")
        else:
            print(f"black is winning")


        #find_otter_moves(stockfish, position=starting_position, otter_engine=otter_engine)
        get_level_attributes(stockfish, fen, levels, otter_engine, stockfish_time_ms=10)
        print()

        epd_elapsed = time.perf_counter() - epd_start
        print(f"Time for this EPD: {epd_elapsed:.2f} seconds for {levels} levels\n")
    """
    epd_pins = [
        "7k/6p1/r7/4n3/8/2B5/6Q1/6K1 w - -", 
        "7k/r5p1/8/4n3/8/2B3Q1/8/6K1 w - -", 
        "7k/r7/5p2/4n3/8/2B3Q1/8/6K1 w - -", 
        "7k/6p1/r7/4n3/8/2B5/5Q2/6K1 b - -", 
        "3r2k1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 w - -",
        "3r2k1/1p3ppp/pq1Q1b2/8/8/1P3N2/P2R1PPP/6K1 b - -",
        "r4rk1/pp3ppp/8/2bqp2Q/3N4/P2Bn2P/1PPN1P2/R1B2RK1 b - -",
        "r4r1k/pp3ppp/8/2bqp2Q/P1nN4/3B3P/1PPN1P2/R1B2RK1 w - -"
    ]

    for epd in epd_pins:
        bishop_pins = find_specific_pin(epd, chess.BISHOP)
        rook_pins = find_specific_pin(epd, chess.ROOK)
        queen_pins = find_specific_pin(epd, chess.QUEEN)
        print("----------------------")
        print(f"final bishop pins: {bishop_pins}")
        print(f"final rook pins: {rook_pins}")
        print(f"final queen pins: {queen_pins}")
        print("----------------------")
