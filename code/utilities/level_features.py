import chess
from engines import otter_engine_func

from additional_topics_features import find_all_forks_current, find_all_forks_threatening, find_all_pins

EVAL_BOUND = 10000 # TODO discuss
POSSIBLE_MOVES_LIMIT = 3


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

    #TODO memoization instad of this limit
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

    starting_position = chess.Board(fen)
    winning_side = starting_position.turn

    # pins and forks at level 1
    starting_epd = starting_position.epd()
    current_forks = find_all_forks_current(starting_epd)
    threatening_forks = find_all_forks_threatening(starting_epd)
    all_pins = find_all_pins(starting_epd)

    data["current_forks_L1"] = len(current_forks)
    data["threatening_forks_L1"] = len(threatening_forks)
    data["all_pins_L1"] = len(all_pins)

    #meaningful_moves, winning_but_not_mating = find_meaningful_moves(stockfish, starting_position, winning_side, level_one=True, stockfish_time_ms=stockfish_time_ms)
    meaningful_moves, winning_but_not_mating = find_otter_moves(stockfish, otter_engine, starting_position)

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

        # Go through all combinations
        for meaningful_move_list in meaningful_moves_combinations:
            new_position = next_position.copy()
            for meaningful_move in meaningful_move_list:
                new_position.push(meaningful_move)

            position_epd = new_position.epd()
            current_forks = find_all_forks_current(position_epd)
            threatening_forks = find_all_forks_threatening(position_epd)
            all_pins = find_all_pins(position_epd)

            current_forks_count += len(find_all_forks_current(position_epd))
            threatening_forks_count += len(find_all_forks_threatening(position_epd))
            all_pins_count += len(find_all_pins(position_epd))

            #new_moves, _ = find_meaningful_moves(stockfish, new_position, winning_side, level_one=False, stockfish_time_ms=stockfish_time_ms)
            new_moves, _ = find_otter_moves(stockfish, otter_engine, new_position)
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

    # if yes store winning and mating 
    if top_eval_stockfish["Mate"] is not None:
        winning_but_not_mating = 0

    # 2. extract otter moves and filter them with some threshold
    otter_data = otter_engine_func(position.epd(), otter_engine)

    for item in otter_data["moves"]:
        move = item["move"]
        probability = item["probability"]

        # threshold
        if probability < 0.05:
            break

        meaningful_moves.append(chess.Move.from_uci(move))

    stockfish_move = chess.Move.from_uci(top_eval_stockfish["Move"])
    if stockfish_move not in meaningful_moves:
        meaningful_moves.append(stockfish_move)

    return meaningful_moves, winning_but_not_mating
    
def find_meaningful_moves(stockfish, position, winning_side, level_one=False, stockfish_time_ms=100):
    if position.is_game_over():
        return [], []

    # pos.turn = True if white or False if black

    moves = list(position.legal_moves)
    evaluations = get_stockfish_evaluation(stockfish, position, moves, stockfish_time_ms=stockfish_time_ms)

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
        if is_mate and (y * mate_in) >= 0:
            # Same logic as above, but you have to check if its players turn or not
            if best_mate is None or (y * x * mate_in) < (y * x * best_mate):
                best_mate = mate_in
                mate_moves = [move]

            elif mate_in == best_mate:
                mate_moves.append(move)

    # If some mate moves are found, finish
    if mate_moves:
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
                eval_cp = turn_next_move * EVAL_BOUND
                eval_cp = eval_cp if wdl[0] > wdl[2] else -eval_cp

        evaluations.append((move, eval_cp, is_mate, mate_in))
    return evaluations
