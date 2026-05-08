import pandas as pd
import chess

import warnings
warnings.filterwarnings('ignore')

from stockfish import Stockfish
import time

# idk, they set it like this in the paper
w = 200
m = 50

stockfish = Stockfish(path="stockfish.exe")


def tic():
    return time.time()

def toc(start, name="block"):
    print(f"{name}: {time.time() - start:.4f}s")



def chebyshev_distance(move):
    from_sq = move.from_square
    to_sq = move.to_square

    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)

    to_file = chess.square_file(to_sq)
    to_rank = chess.square_rank(to_sq)

    return max(abs(from_file - to_file), abs(from_rank - to_rank))



def evaluation_cp(position, moves, turn, turn_next_move, stockfish_time_ms):
    

    # go through all the possible moves and get evaluations
    evaluations = []
    for move in moves:
        new_pos = position.copy()
        new_pos.push(move)

        stockfish.set_fen_position(new_pos.fen())
        #start = tic()
        raw_eval = stockfish.get_evaluation(searchtime=stockfish_time_ms)
        #toc(start, "get_evaluation")
        #print(f"RAW EVAL: {raw_eval} for move {move}")

        wdl = stockfish.get_wdl_stats()
        #print(f"WDL: {wdl}")

        # convert to something normal iguess
        if turn_next_move: 
            # if white is to move
            if raw_eval["type"] == "cp":
                eval_cp = raw_eval["value"]
            else:

                # to se delam da mat stoji na sahovnci idk idc
                if raw_eval["value"] == 0:
                    eval_cp = 15 * 100
                else:
                    # if wdl is [1000, 0, 0] then white is winning
                    if wdl[0] > 900:
                        eval_cp = 15 * 100
                    else:
                        eval_cp = -15 * 100

        else: 
            # if black is to move 
            if raw_eval["type"] == "cp":
                eval_cp = -raw_eval["value"]
            else:
                
                if raw_eval["value"] == 0:
                    eval_cp = -15 * 100

                else:

                    # if wdl is [1000, 0, 0] then black is winning
                    if wdl[0] > 900:
                        eval_cp = -15 * 100
                    else:
                        eval_cp = 15 * 100

        #print(f"final eval for this position: {eval_cp}")

        evaluations.append((move, eval_cp))

    return evaluations




def evaluation_static(position, moves, turn, turn_next_move):

    #print(f"position: {position}")
    """
    if turn:
        print(f"on move: white")
    else:
        print(f"on move: black")
    """

    evaluations = []
    for move in moves:
        new_pos = position.copy()
        new_pos.push(move)

        stockfish.set_fen_position(new_pos.fen())
        #raw_eval = stockfish.get_evaluation()
        raw_static_eval = stockfish.get_static_eval()

        """
        print(f"RAW EVAL: {raw_eval} for move {move}")
        print(f"STATIC EVAL: {static_eval} for move {move}")
        print()
        """

        #wdl = stockfish.get_wdl_stats()
        #print(f"WDL: {wdl}")

        eval_cp = 0

        if raw_static_eval is not None:
            if turn_next_move: 
                eval_cp = raw_static_eval * 100
            else: 
                eval_cp = raw_static_eval * (-100)

        else:
            
            raw_eval = stockfish.get_evaluation()
            wdl = stockfish.get_wdl_stats()

            # convert to something normal iguess
            if turn_next_move: 
                # if white is to move
                if raw_eval["type"] == "cp":
                    eval_cp = raw_eval["value"]
                else:

                    # to se delam da mat stoji na sahovnci idk idc
                    if raw_eval["value"] == 0:
                        eval_cp = 15 * 100
                    else:
                        # if wdl is [1000, 0, 0] then white is winning
                        if wdl[0] > 900:
                            eval_cp = 15 * 100
                        else:
                            eval_cp = -15 * 100

            else: 
                # if black is to move 
                if raw_eval["type"] == "cp":
                    eval_cp = -raw_eval["value"]
                else:
                    
                    if raw_eval["value"] == 0:
                        eval_cp = -15 * 100

                    else:

                        # if wdl is [1000, 0, 0] then black is winning
                        if wdl[0] > 900:
                            eval_cp = -15 * 100
                        else:
                            eval_cp = 15 * 100

        #print(f"final eval for this position: {eval_cp}")

        evaluations.append((move, eval_cp))

    return evaluations




def find_meaningful_moves(position, winning_side, level_one=False, stockfish_time_ms=100):
    """
    function that returns all the meaningful moves for the given position 
    """

    # first check if the position is mate (or stalemate or whatever)
    if position.is_game_over():
        return [], []
    
    
    # who's on the move
    turn = position.turn  # True = white, False = black
    #"""
    if turn:
        #print(f"white is on move")
        turn_next_move = False
    else:
        #print(f"black is on move")
        turn_next_move = True
    #"""
    #print(f"next move is gonna be by: {turn_next_move}")

    # first get all the possible moves in the position
    moves = list(position.legal_moves)
    #print(f"all legal moves: {moves}")

    #start = tic()

    #print(f"to search: {len(moves)} moves")
    evaluations = evaluation_cp(position, moves, turn, turn_next_move, stockfish_time_ms)
    #evaluations = evaluation_static(position, moves, turn, turn_next_move)

    #toc(start, "evaluations")

    # get the best score 
    if turn:
        # if white is on move, the best score is +
        best_eval = max(e for _, e in evaluations)
    else:
        # otherwise the best move score is as small as possible
        best_eval = min(e for _, e in evaluations)

    #print(f"evaluations\n{evaluations}")
    #print(f"BEST EVAL: {best_eval}")
    # now do whatever they do in pseudocode
    meaningful_moves = []
    winning_but_not_mating = 0

    for move, eval_cp in evaluations:

        # and now check based on who's supposed to win 
        # whether the move has higher score then w/m 
        #print(f"winning side: {winning_side}, side on turn: {turn}")
        if winning_side == turn:
            # checks if the correct side is still winning
            if winning_side == chess.WHITE:
                if eval_cp >= w:
                    meaningful_moves.append(move)

                    # winning and also mating ?
                    if level_one:
                        if abs(eval_cp) == 1500:            # ni najleps sam idc sue me
                            winning_but_not_mating += 0
                        else: 
                            winning_but_not_mating += 1
            else:
                if eval_cp <= -w:
                    meaningful_moves.append(move)

                    # winning and also mating ?
                    if level_one:
                        if abs(eval_cp) == 1500:
                            winning_but_not_mating += 0
                        else: 
                            winning_but_not_mating += 1

        else:
            # checks if a move is close enough to the best move 
            if abs(best_eval - eval_cp) <= m:
                meaningful_moves.append(move)


    #print(f"meaningful moves: {meaningful_moves}")
    #print(f"we have {len(meaningful_moves)} meaningful moves")

    return meaningful_moves, winning_but_not_mating



def number_of_meaningful_moves(data, max_levels, do_x_samples, testing, stockfish_time_ms):

    # (?) idk how much we should put 
    # level 1 means all the meaningful moves for the winning side 
    # level 2 all the meaningful moves for the opposite side after all the meaningful moves of the winning side 
    # level 3 all the meaningful moves for winning side after ...

    # also this is strictly for puzzles i think but ok 
    # because winningSideToMove parameter in algorithm

    levels = max_levels 

    # prepare columns
    for l in range(1, levels+1):
        data[f"meaningful_L{l}"] = 0

    for l in range(1, levels+1):
        data[f"branching_L{l}"] = 0.0

    data["avg_branching"] = 0.0

    for l in range(1, levels+1):
        data[f"narrow_L{l}"] = 0        

    for l in range(1, levels+1):
        data[f"distance_L{l}"] = 0

    for l in range(1, levels+1):
        data[f"pieces_L{l}"] = 0

    data["all_pieces_involved"] = 0

    data["winning_no_mate"] = 0

    for idx, row in data.iterrows(): 
        if testing:
            if idx >= do_x_samples:
                break
            print(f"row {idx} start")
        #print(row["epd"])

        starting_position = chess.Board(row["epd"])
        winning_side = starting_position.turn   # !!that's why it's only for puzzles
        #if winning_side:
        #    continue

        # for the level 1, that is a starting position
        # find all the meaningful moves and return them
        #print(f"LEVEL 1")
        meaningful_moves, winning_but_not_mating = find_meaningful_moves(starting_position, winning_side, True, stockfish_time_ms)
        #print(f"meaningful moves: {meaningful_moves}")

        meaningful_moves_combs = [[move] for move in meaningful_moves]
        data.loc[idx, "meaningful_L1"] = len(meaningful_moves)
        if winning_but_not_mating == False: 
            data.loc[idx, "winning_no_mate"] = 0
        else:
            data.loc[idx, "winning_no_mate"] = 1

        # narrow solutions
        if len(meaningful_moves) == 1:
            data.loc[idx, "narrow_L1"] = 1
        else:
            data.loc[idx, "narrow_L1"] = 0

        # distances 
        dist_L1 = sum(chebyshev_distance(m) for m in meaningful_moves)
        data.loc[idx, "distance_L1"] = dist_L1

        # different pieces involved
        all_piece_types = set()

        piece_types = set()
        for move in meaningful_moves:
            piece = starting_position.piece_at(move.from_square)
            if piece is not None:
                piece_types.add(piece.piece_type)
                all_piece_types.add(piece.piece_type)
        data.loc[idx, "pieces_L1"] = len(piece_types)

        # then for the other levels, play the found meaningful moves first
        # then find the meaningful moves in the final position
        # and also, moves should be like this 
        # [ move_combination_1[e4 e5 Sf3] move_combination_2[e4 e5 Sc3] move_combination_2[e4 e5 Lc4] move_combination_2[e4 c5 Sf3] ...]
        # so like level 1 is just
        # [ move_combination_1[e4] move_combination_2[d4] move_combination_2[c4] move_combination_2[Sf3]]
        # then level 2 like this:
        # [ move_combination_1[e4 c5] move_combination_2[e4 e5] move_combination_2[e4 c6] ... move_combination_2[Sf3 Sf6]]
        
        # break

        for level in range(0, levels-1):

            #print(f"LEVEL {level+2}")

            new_combinations = []

            # for narrow attribute 
            narrow_count = 0

            # for distance attribute
            distance_count = 0

            # for different pieces attribute
            pieces_count = set()

            # go through all the combinations of moves
            for meaningful_move_comb in meaningful_moves_combs:

                # play all the moves and get the final position 
                new_pos = starting_position.copy()
                for meaningful_move in meaningful_move_comb:
                    new_pos.push(meaningful_move)

                # get the next meaningful moves
                new_moves, _ = find_meaningful_moves(new_pos, winning_side, False, stockfish_time_ms)

                distance_count += sum(chebyshev_distance(m) for m in new_moves)

                if len(new_moves) == 1:
                    narrow_count += 1

                # and finally make combinations
                # and hope that it aint gonna explode <3

                for move in new_moves:
                    new_combinations.append(meaningful_move_comb + [move])

                    # also check the pieces in the meantime
                    piece = new_pos.piece_at(move.from_square)
                    if piece is not None:
                        pieces_count.add(piece.piece_type)
                        all_piece_types.add(piece.piece_type)


            meaningful_moves_combs = new_combinations

            data.loc[idx, f"meaningful_L{level+2}"] = len(meaningful_moves_combs)
            data.loc[idx, f"narrow_L{level+2}"] = narrow_count
            data.loc[idx, f"distance_L{level+2}"] = distance_count
            data.loc[idx, f"pieces_L{level+2}"] = len(pieces_count)

            data.loc[idx, "all_pieces_involved"] = len(all_piece_types)

        """
        print("Final combinations:")
        for comb in meaningful_moves_combs:
            print(comb)
        """


        # BRANCHING

        branchings = []

        # first Meaningful[L-1] / Meaningful[L] -> that's in paper
        # but we do Meaningful[L] / Meaningful[L-1] because that makes more sense? 
        # otherwise branching factor will never pass 1 
        # because Meaningful[L] > Meaningful[L-1] alwayys
        for l in range(2, levels+1):
            prev_val = data.loc[idx, f"meaningful_L{l-1}"]
            curr_val = data.loc[idx, f"meaningful_L{l}"]

            if prev_val > 0:
                branching = curr_val / prev_val
            else:
                branching = 0.0

            data.loc[idx, f"branching_L{l}"] = branching
            branchings.append(branching)


        # average branching
        if len(branchings) > 0:
            data.loc[idx, "avg_branching"] = sum(branchings) / len(branchings)
        else:
            data.loc[idx, "avg_branching"] = 0.0
     


    # remove branching L1
    data = data.drop(columns=["branching_L1"], errors="ignore")
        
    return data


def dfs_possible_moves(position, total_num_of_levels, current_level, counts):

    # first check if its final position
    if position.is_game_over():
        return

    # if level is more than 3 iguess this is not so stupid to add
    #if counts[current_level] > SOME_LIMIT:
    #    return

    moves = list(position.legal_moves)

    # count moves for this level
    counts[current_level] += len(moves)


    # breaking condition ig
    if current_level >= total_num_of_levels:
        return
    

    # otherwise go through moves and move them 
    current_level += 1
    for move in moves:
    
        new_pos = position.copy()
        new_pos.push(move)
        

        dfs_possible_moves(new_pos, total_num_of_levels, current_level, counts)


    



def number_of_possible_moves(data, max_levels, do_x_samples, testing):

    levels = max_levels
    for l in range(1, levels+1):
        data[f"possible_L{l}"] = 0


    for idx, row in data.iterrows():
        if testing:
            if idx >= do_x_samples:
                break
            print(f"row {idx} start")

        starting_position = chess.Board(row["epd"])
        counts = {l: 0 for l in range(1, levels+1)}

        # do dfs for the position for 3 levels
        dfs_possible_moves(starting_position, levels, 1, counts)

        # save results
        for l in range(1, levels+1):
            data.loc[idx, f"possible_L{l}"] = counts[l]



    return data




def miscellaneous(data, max_levels, do_x_samples, testing):

    levels = max_levels

    # AllPossibleMoves - sum over possible moves
    possible_cols = [f"possible_L{l}" for l in range(1, levels+1)]
    data["all_possible_moves"] = data[possible_cols].sum(axis=1)

    # AllNarrowSolutions - sum over narrow solutions
    narrow_cols = [f"narrow_L{l}" for l in range(1, levels+1)]
    data["all_narrow_solutions"] = data[narrow_cols].sum(axis=1)

    # TreeSize - sum over meaningful moves
    meaningful_cols = [f"meaningful_L{l}" for l in range(1, levels+1)]
    data["tree_size"] = data[meaningful_cols].sum(axis=1)

    # MoveRatio(L) - ratio between meaningful moves and all possible moves
    for l in range(1, levels+1):
        meaningful_col = f"meaningful_L{l}"
        possible_col = f"possible_L{l}"
        ratio_col = f"move_ratio_L{l}"

        data[ratio_col] = data[meaningful_col] / data[possible_col]
        data[ratio_col] = data[ratio_col].fillna(0.0)


    # SumDistance - sum over distances
    distance_cols = [f"distance_L{l}" for l in range(1, levels+1)]
    data["sum_distance"] = data[distance_cols].sum(axis=1)

    # AverageDistance
    data["avg_distance"] = data["sum_distance"] / data["tree_size"]
    data["avg_distance"] = data["avg_distance"].fillna(0.0)


    return data

def add_stockfish_encodings(data, do_x_samples, elo_ratings, testing):
    stockfish_evaluations = {f"solved{elo}": [] for elo in elo_ratings}

    for idx, row in data.iterrows():
        if testing:
            if idx >= do_x_samples:
                break
            print(f"row {idx} start")

        position = chess.Board(row["epd"])
        moves_gt = row["solution"].split(" ")[1:]

        for elo_rating in elo_ratings:
            stockfish.set_elo_rating(elo_rating)
            stockfish.set_fen_position(position.fen())

            did_solve = 1
            for i, move in enumerate(moves_gt):
                if i % 2 == 1:
                    stockfish_move = stockfish.get_best_move()
                    if stockfish_move != move:
                        did_solve = 0
                        break
                stockfish.make_moves_from_current_position([move])

            stockfish_evaluations[f"solved{elo_rating}"].append(did_solve)

        processed_data = data.head(do_x_samples)
        new_columns_df = pd.DataFrame(stockfish_evaluations, index=processed_data.index)
        return pd.concat([processed_data, new_columns_df], axis=1)

if __name__ == '__main__':

    # read the file
    dataset_100k = pd.read_csv("dataset_100k.csv")
    #print(dataset_100k)

    #dataset_100k = dataset_100k.loc[[6]]

    levels = 3          # depth, if we're brave enough, 5 would be nice (be aware to change some logic in that case tho!!)
    do_x_samples = 20    # debugging and time consuming reasons
    stockfish_time_ms = 10 # idk
    testing = True      # DO FALSE ONCE YOU RUN ON ALL DATA
    elo_ratings = [1320] + [x for x in range(1500, 3001, 250)]
    
    # 1 Meaningful(L) -> for 3 levels
    # 4 Branching(L) -> for 3 levels (but 1 excluded)
    # 5 AverageBranching -> maybe not even useful since we then only have two levels lol
    # 6 NarrowSolution(L) 
    # 11 Distance(L)
    # 14 Pieces(L)
    # 15 AllPiecesInvolved
    # 17 WinningNoCheckmate
    start = tic()
    dataset_100k = number_of_meaningful_moves(dataset_100k, levels, do_x_samples, testing, stockfish_time_ms)
    #toc(start, "meaningful moves 30 samples 10ms")
    #print(dataset_100k)

    
    # 2 PossibleMoves(L) 
    dataset_100k = number_of_possible_moves(dataset_100k, levels, do_x_samples, testing)
    #print(dataset_100k)


    # posebej funkcija, ker so sam sestevki po stolpcih ipd 
    # 3 AllPossibleMoves
    # 7 AllNarrowSolutions
    # 8 TreeSize
    # 9 MoveRatio
    # 12 SumDistance
    # 13 AverageDistance 
    dataset_100k = miscellaneous(dataset_100k, levels, do_x_samples, testing)

    # Add feature vector that encodes what stockfish solved puzzle
    dataset_100k = add_stockfish_encodings(dataset_100k, do_x_samples, elo_ratings, testing)

    print(dataset_100k.head(10))



    # 16 PieceValueRatio -> MAMO ZE SAMI
    # 18 BestMoveValue -> MAMO ZE SAMI
    # 19 AverageBestMove(5) -> we aint doing level 5 <3 

    # store new dataset
    #if testing == False:
    toc(start, name="all calculations for all samples")
    dataset_100k.head(do_x_samples).to_csv("dataset_100k_upgraded.csv", index=False)



