import chess
import chess.engine
import pandas as pd
import time
import sys
from pathlib import Path
import os
from plots import plot_boards, plot_comparison

# PYTUROCHAMP
# https://github.com/stevexyz/PyTuroChamp
import subprocess
PYTUROCHAMP_PATH = Path(__file__).resolve().parent.parent / "engines" / "PyTuroChamp"




def _classical_chess_engine_func(epd, move, engine, depth=3):
    board = chess.Board()
    board.set_epd(epd)

    move_info = engine.play(board, chess.engine.Limit(depth=depth))
    engine_move = move_info.move.uci() if move_info.move else None

    return int(engine_move == move)

def _sunfish_engine_func(epd, move, engine, rating=-1, depth=3):
    board = chess.Board()
    board.set_epd(epd)

    move_info = engine.play(board, chess.engine.Limit(depth=depth))
    engine_move = move_info.move.uci() if move_info.move else None

    return int(engine_move == move)

def _otter_engine_func(epd, move, engine):
    board = chess.Board()
    board.set_epd(epd)

    moves = list(board.legal_moves)
    
    result = engine.predict(
        fen=board.fen(),
        player_elo=3000,
        opponent_elo=3000,
        history_moves=[],
        time_control="1000+0",
        clock_fraction=0.9,
        top_k=len(moves)
    )

    engine_move = (
        max(result["moves"], key=lambda x: x["probability"])["move"]
        if result["moves"]
        else None
    )

    return int(engine_move == move)


def _stockfish_engine_func(epd, move, engine, elo_levels=[1320, 2000]):

    out = {}

    board = chess.Board()
    board.set_epd(epd)

    for elo in elo_levels:
        engine.set_elo_rating(elo)
        engine.set_fen_position(board.fen())
        engine_move = engine.get_best_move()

        out[f"stockfish_{elo}_solved"] = int(engine_move == move)

    return out

def _maia3_engine_func(epd, move, engine, elo_levels=[1320, 2000]):

    out = {}

    board = chess.Board()
    board.set_epd(epd)

    for elo in elo_levels:
        # Set the human skill level being modeled
        engine.configure({"Elo": elo, "Temperature": "0"})
        move_info = engine.play(board, chess.engine.Limit(nodes=1))
        engine_move = (move_info.move.uci() if move_info.move else None)

        out[f"maia3_{elo}_solved"] = int(engine_move == move)

    return out

def _get_pyturochamp_move(epd, mode="ptc"):
    fen = epd.strip()
    fields = fen.split()
    if len(fields) == 4:
        fen += " 0 1"

    process = subprocess.Popen(
        [sys.executable, str(PYTUROCHAMP_PATH / "ptc_xboard.py"), mode],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        process.stdin.write("uci\n")
        process.stdin.flush()

        while True:
            line = process.stdout.readline()
            
            if not line:
                raise RuntimeError("Engine exited before sending uciok.")

            if line.strip() == "uciok":
                break

        process.stdin.write("isready\n")
        process.stdin.flush()

        while True:
            line = process.stdout.readline()

            if not line:
                raise RuntimeError("Engine exited before sending readyok.")

            if line.strip() == "readyok":
                break

        process.stdin.write(f"position fen {fen}\n")
        process.stdin.write("go\n")
        process.stdin.flush()

        while True:
            line = process.stdout.readline()

            if not line:
                raise RuntimeError("Engine exited before returning a move.")

            line = line.strip()
            if line.startswith("bestmove "):
                engine_move = line.split()[1]
                return engine_move

    finally:
        if process.poll() is None:
            process.stdin.write("quit\n")
            process.stdin.flush()
            process.wait(timeout=5)

def _pyturochamp_engine_func(epd, move, modes):

    out = {}

    for mode in modes:
        engine_move = _get_pyturochamp_move(epd, mode=mode)
        out[f"{mode}_solved"] = int(engine_move == move)

    return out


def _solve_puzzle(epd, solution_moves, engine_func, key=None, **kwargs):
    """
    Checks whether an engine finds all moves of the puzzle solution.

    predict_correct:
        Function accepting current EPD and expected move,
        returning 1 if correct, otherwise 0.

    Opponent's responses are taken directly from the solution.
    """

    board = chess.Board()
    board.set_epd(epd)

    for i in range(0, len(solution_moves), 2):

        expected_move = solution_moves[i]

        result = engine_func(
            epd=board.epd(),
            move=expected_move,
            **kwargs
        )

        # Some engine functions return a dictionary,
        # others return an integer.
        correct = result[key] if key is not None else result

        if not correct:
            return 0

        # Play the correct move
        board.push_uci(expected_move)

        # Play the opponent's response
        if i + 1 < len(solution_moves):
            board.push_uci(solution_moves[i + 1])

    return 1


def get_engine_solver_attributes(
        epd,
        moves,
        classical_chess_engine,
        sunfish_engine,
        stockfish_engine,
        maia3_engine,
        otter_engine,
        depth=3,
        elo_levels=(1320, 2700),
        pyturo_modes=("ptc", "bare")
    ):

    solution_moves = str(moves).split()

    if not solution_moves:
        raise ValueError(f"Empty solution for {epd}")

    out = {}

    out["classical_solved"] = _solve_puzzle(epd, solution_moves, _classical_chess_engine_func, engine=classical_chess_engine, depth=depth)
    out["sunfish_solved"] = _solve_puzzle(epd, solution_moves, _sunfish_engine_func, engine=sunfish_engine, depth=depth)
    out["otter_solved"] = _solve_puzzle(epd, solution_moves, _otter_engine_func, engine=otter_engine)

    for elo in elo_levels:
        key = f"stockfish_{elo}_solved"
        out[key] = _solve_puzzle(epd, solution_moves, _stockfish_engine_func, key=key, engine=stockfish_engine, elo_levels=[elo])

    for elo in elo_levels:
        key = f"maia3_{elo}_solved"
        out[key] = _solve_puzzle(epd, solution_moves, _maia3_engine_func, key=key, engine=maia3_engine, elo_levels=[elo])

    for mode in pyturo_modes:
        key = f"{mode}_solved"
        out[key] = _solve_puzzle(epd, solution_moves, _pyturochamp_engine_func, key=key, modes=[mode])

    return out

