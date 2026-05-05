
import pandas as pd
import chess

from stockfish import Stockfish

# idk, they set it like this in the paper
w = 200
m = 50

stockfish = Stockfish(path="/usr/games/stockfish")


def find_meaninful_moves(position, winning_side):
    """
    function that returns all the meaningful moves for the given position 
    """

    # who's on the move
    turn = position.turn  # True = white, False = black
    #"""
    if turn:
        #print(f"white is on move: {turn}")
        turn_next_move = False
    else:
        #print(f"black is on move")
        turn_next_move = True
    #"""
    #print(f"next move is gonna be by: {turn_next_move}")

    # first get all the possible moves in the position
    moves = list(position.legal_moves)
    #print(f"all legal moves: {moves}")

    # go through all the possible moves and get evaluations
    evaluations = []
    for move in moves:
        new_pos = position.copy()
        new_pos.push(move)

        stockfish.set_fen_position(new_pos.fen())
        raw_eval = stockfish.get_evaluation()
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
    for move, eval_cp in evaluations:

        # and now check based on who's supposed to win 
        # whether the move has higher score then w/m 
        #print(f"winning side: {winning_side}, side on turn: {turn}")
        if winning_side == turn:
            # checks if the correct side is still winning
            if winning_side == chess.WHITE:
                if eval_cp >= w:
                    meaningful_moves.append(move)
            else:
                if eval_cp <= -w:
                    meaningful_moves.append(move)
        else:
            # checks if a move is close enough to the best move 
            if abs(best_eval - eval_cp) <= m:
                meaningful_moves.append(move)


    #print(f"meaningful moves: {meaningful_moves}")
    #print(f"we have {len(meaningful_moves)} meaningful moves")

    return meaningful_moves



def number_of_meaningful_moves(data):

    # (?) idk how much we should put 
    # level 1 means all the meaningful moves for the winning side 
    # level 2 all the meaningful moves for the opposite side after all the meaningful moves of the winning side 
    # level 3 all the meaningful moves for winning side after ...

    # also this is strictly for puzzles i think but ok 
    # because winningSideToMove parameter in algorithm

    levels = 3      # <- TODO: to se je se za odloct

    # prepare columns
    for l in range(1, levels+1):
        data[f"meaningful_L{l}"] = 0

    for idx, row in data.iterrows(): 


        #print(row["epd"])



        starting_position = chess.Board(row["epd"])
        winning_side = starting_position.turn   # !!that's why it's only for puzzles
        #if winning_side:
        #    continue

        # for the level 1, that is a starting position
        # find all the meaningful moves and return them
        #print(f"LEVEL 1")
        meaningful_moves = find_meaninful_moves(starting_position, winning_side)
        meaningful_moves_combs = [[move] for move in meaningful_moves]
        data.loc[idx, "meaningful_L1"] = len(meaningful_moves)

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

            # go through all the combinations of moves
            for meaningful_move_comb in meaningful_moves_combs:

                # play all the moves and get the final position 
                new_pos = starting_position.copy()
                for meaningful_move in meaningful_move_comb:
                    new_pos.push(meaningful_move)

                # get the next meaningful moves
                new_moves = find_meaninful_moves(new_pos, winning_side)

                # and finally make combinations
                # and hope that it aint gonna explode <3
                for move in new_moves:
                    new_combinations.append(meaningful_move_comb + [move])
            
            meaningful_moves_combs = new_combinations

            data.loc[idx, f"meaningful_L{level+2}"] = len(meaningful_moves_combs)

        """
        print("Final combinations:")
        for comb in meaningful_moves_combs:
            print(comb)
        """

        
        print(f"row {idx} done")
        #if idx > 10:
        #    break
        
    return data




if __name__ == '__main__':

    # read the file
    dataset_100k = pd.read_csv("dataset_100k.csv")
    print(dataset_100k)



    # add attributes
    updated_data = number_of_meaningful_moves(dataset_100k)
    print(updated_data)