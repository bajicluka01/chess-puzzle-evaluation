import chess

def get_first_move(moves):
    return chess.Move.from_uci(moves.split(" ")[0])

def get_board_from_fen(fen):
    return chess.Board(fen)


