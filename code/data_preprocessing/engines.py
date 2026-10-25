import chess
import chess.engine
import pandas as pd
import time
import sys
from pathlib import Path

# for plotting a chess board
import matplotlib.pyplot as plt
import cairosvg
from io import BytesIO
import chess.svg



# stockfish
from stockfish import Stockfish
STOCKFISH_PATH = "/usr/games/stockfish"
stockfish_engine = Stockfish(path=STOCKFISH_PATH)

# https://github.com/CSSLab/maia3
maia3_engine = chess.engine.SimpleEngine.popen_uci([
    sys.executable, "-m", "maia3.uci",
    "--model", "maia3-5m",
    "--elo", "1500",
    "--temperature", "0"
])
maia3_engine.quit()


# https://github.com/stevexyz/PyTuroChamp
import subprocess
PYTUROCHAMP_PATH = Path(__file__).resolve().parent.parent / "engines" / "PyTuroChamp"
"""pyturochamp_engine = chess.engine.SimpleEngine.popen_uci(
    [
        sys.executable,
        str(PYTUROCHAMP_PATH / "ptc_xboard.py")
    ],
    debug=True
)
"""



def plot_boards(epd):
    """
        Visualizes puzzle from given epd
    """

    board = chess.Board()
    board.set_epd(epd)
    #print(board)

    svg = chess.svg.board(board=board, size=400)
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"))

    plt.figure(figsize=(5, 5))
    plt.imshow(plt.imread(BytesIO(png)))
    plt.axis("off")
    plt.show()

def puzzle_solver_statistic(results, engine_name):

    correct_count = results["correct"].sum()
    total = len(results)

    print(f" {engine_name.upper()} — OVERALL STATISTICS")
    print(f"Puzzles evaluated: {total}")
    print(f"Correct: {correct_count}")
    print(f"Wrong: {total - correct_count}")
    print(f"Accuracy: {correct_count / total * 100:.2f}%" if total else "No puzzles evaluated.")

    # Statistics for each 100-point rating interval
    results["rating_interval"] = (
        (results["rating"] // 100) * 100
    )

    statistics = results.groupby("rating_interval").agg(
        total=("correct", "size"),
        correct=("correct", "sum")
    )

    statistics["wrong"] = statistics["total"] - statistics["correct"]
    statistics["accuracy (%)"] = (
        statistics["correct"] / statistics["total"] * 100
    ).round(2)

    statistics.index = [
        f"{rating}-{rating + 99}"
        for rating in statistics.index
    ]
    print(statistics.to_string())


    print("\n=== Statistics by rating interval ===")

    return



def compare_engines_by_rating(engine_results, save_to_csv):
    all_stats = []
    total_by_rating = None

    for engine_name, results in engine_results.items():
        if results is None or results.empty:
            continue

        df = results.copy()
        df["correct"] = df["correct"].fillna(False).astype(bool)
        df["rating_bin"] = (df["rating"] // 100) * 100

        stats = df.groupby("rating_bin").agg(
            Total=("correct", "size"),
            **{engine_name: ("correct", lambda x: 100 * x.sum() / len(x))}
        )

        # Store the shared Total column only once
        if total_by_rating is None:
            total_by_rating = stats["Total"]

        all_stats.append(stats[[engine_name]])

    if not all_stats:
        return pd.DataFrame()

    # Combine engine accuracies
    comparison = pd.concat(all_stats, axis=1).sort_index()

    # Add the shared Total column
    comparison.insert(0, "Total", total_by_rating)

    comparison.index = comparison.index.map(
        lambda r: f"{int(r)} - {int(r) + 99}"
    )
    comparison.index.name = "Puzzle rating"

    comparison["Total"] = comparison["Total"].astype(int)
    print("\n=== Engine accuracy by puzzle rating ===")
    print()
    print(comparison.to_string(
        col_space=12,
        formatters={
            col: "{:.2f}%".format
            for col in comparison.columns
            if col != "Total"
        }
    ))

    if save_to_csv:
        comparison.to_csv("engine_comparison.csv", float_format="%.2f")

    return comparison


def classical_chess_engine_func(epd, move, engine, rating=-1, depth=3):

    board = chess.Board()
    board.set_epd(epd)

    """
    engine.analyse:
        {
            'depth': 2,
            'score': PovScore(Cp(+394), WHITE),
            'nodes': 2652,
            'nps': 2843,
            'time': 0.932,
            'string': 'maxstats main_nodes=37 qnodes=2615 tt_probes=37 tt_hits=0 beta_cutoffs=34 q_beta_cutoffs=1651'
        }

    engine.play:
        (
            move=f4f6,
            ponder=None,
            info={},
            draw_offered=False,
            resigned=False
        )
    """

    #evaluation = classical_chess_engine.analyse(
    #    board,
    #    chess.engine.Limit(depth=depth)
    #)

    move_info = engine.play(
            board,
            chess.engine.Limit(depth=depth)
    )
    engine_move = move_info.move.uci() if move_info.move else None

    results_df = pd.DataFrame([{
        "rating": rating,
        "engine_move": engine_move,
        "expected_move": move,
        "correct": engine_move == move
    }])

    """
    elapsed_time = time.perf_counter() - start_time
    total_engine_time += elapsed_time

    print("Position:", epd)
    print("Best move:", board.san(move_info.move) if move_info.move else None)
    #print("UCI move:", move_info.move)
    print("Evaluation:", evaluation["score"])
    print(f"solution in df: {df.iloc[idx]['solution']}")
    print(f"puzzle rating: {df.iloc[idx]['rating']}")
    print(f"Engine thinking time: {elapsed_time:.3f} seconds")
    print()

    plot_boards(epd)
    """

    return results_df


def sunfish_engine_func(epd, move, engine, rating=-1, depth=3):

    board = chess.Board()
    board.set_epd(epd)

    """
    evaluation = classical_chess_engine.analyse(
        board,
        chess.engine.Limit(depth=depth)
    )
    print(evaluation)
    """

    move_info = engine.play(
            board,
            chess.engine.Limit(depth=depth)
    )
    #print(move_info)


    engine_move = move_info.move.uci() if move_info.move else None

    results_df = pd.DataFrame([{
        "rating": rating,
        "engine_move": engine_move,
        "expected_move": move,
        "correct": engine_move == move
    }])

    return results_df


def otter_engine_func(epd, otter_engine):


    board = chess.Board()
    board.set_epd(epd)

    moves = list(board.legal_moves)
    #print(f"{len(moves)} legal moves in position")

    result = otter_engine.predict(
        fen=board.fen(),
        player_elo=3000,
        opponent_elo=3000,
        history_moves=[],
        time_control="1000+0",
        clock_fraction=0.9,
        top_k=len(moves)
    )
    """
    #print(result)
    print(f"fen:                            {result['fen']}")
    print(f"win probability:                {result['win_probability']}")
    candidate_moves = []
    for idx, move in enumerate(result['moves']):
        candidate_moves.append(move['move'])
    print(f"candidate moves                 {result['moves']}")
    print(f"aux predicitons:")
    print(f"    moving piece:               {result['aux_predictions']['moving_piece']}")
    print(f"    moving piece confidence:    {result['aux_predictions']['moving_piece_confidence']}")
    print(f"    captured piece:             {result['aux_predictions']['captured_piece']}")
    print(f"    captured piece confidence:  {result['aux_predictions']['captured_piece_confidence']}")
    print(f"    from square:                {result['aux_predictions']['from_square']}, {result['aux_predictions']['from_square_confidence']}")
    print(f"    to square:                  {result['aux_predictions']['to_square']}, {result['aux_predictions']['to_square_confidence']}")
    print(f"--------------------------------------")
    #solution_moves = str(row["solution"]).split()
    #print(f"correct move:       {solution_moves[1]}")
    #print(f"rating of puzzle:   {row['rating']}")
    plot_boards(epd)
    """
    """
        {
            'fen': 'r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w - - 0 1',
            'win_probability': -0.47298043966293335,
            'moves': [
                {
                    'move': 'h2g3',
                    'probability': 0.4267266094684601
                },
                {
                    'move': 'e6e7',
                    'probability': 0.2119789719581604
                },
                {
                    'move': 'h6f6',
                    'probability': 0.13230599462985992
                },
                {
                    'move': 'e6f6',
                    'probability': 0.10580255091190338
                },
                {
                    'move': 'h2h3',
                    'probability': 0.03893489018082619
                }
            ],
            'aux_predictions':
                {
                    'moving_piece': 'Queen',
                    'moving_piece_confidence': 1.0525637419015729e-20,
                    'captured_piece': 'Pawn',
                    'captured_piece_confidence': 0.5861598253250122,
                    'results_in_check_probability': 0.002534935250878334,
                    'from_square': 'e6',
                    'from_square_confidence': 0.34457823634147644,
                    'to_square': 'e7',
                    'to_square_confidence': 0.46594658493995667
                }
        }

    """

    return result

def stockfish_engine_func(epd, move, rating=-1, elo_levels=[1320, 2000]):


    results = []

    for elo in elo_levels:

        stockfish_engine.set_elo_rating(elo)

        board = chess.Board()
        board.set_epd(epd)

        stockfish_engine.set_fen_position(board.fen())

        engine_move = stockfish_engine.get_best_move()

        results.append({
            "rating": rating,
            "elo": elo,
            "engine_move": engine_move,
            "expected_move": move,
            "correct": engine_move == move
        })



    results_df = pd.DataFrame(results)
    return results_df


def maia3_engine_func(epd, move, rating=-1, elo_levels=[1320, 2000]):


    results = []

    for elo in elo_levels:

        # Set the human skill level being modeled
        maia3_engine.configure({
            "Elo": elo,
            "Temperature": "0"
        })

        board = chess.Board()
        board.set_epd(epd)

        move_info = maia3_engine.play(
            board,
            chess.engine.Limit(nodes=1)
        )

        engine_move = (
            move_info.move.uci()
            if move_info.move else None
        )

        results.append({
            "rating": rating,
            "elo": elo,
            "engine_move": engine_move,
            "expected_move": move,
            "correct": engine_move == move
        })

    results_df = pd.DataFrame(results)
    return results_df


def get_pyturochamp_move(epd,  mode="ptc"):

    fen = epd.strip()
    fields = fen.split()
    if len(fields) == 4:
        fen += " 0 1"

    #print("FEN sent to engine:", fen)

    process = subprocess.Popen(
        [sys.executable, str(PYTUROCHAMP_PATH / "ptc_xboard.py"), mode],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    #print("Engine process started. PID:", process.pid)

    try:
        #print("Sending: uci")
        process.stdin.write("uci\n")
        process.stdin.flush()

        while True:
            line = process.stdout.readline()
            #print("STDOUT (uci):", repr(line))

            if not line:
                #print("STDERR:", process.stderr.read())
                raise RuntimeError("Engine exited before sending uciok.")

            if line.strip() == "uciok":
                break

        #print("Sending: isready")
        process.stdin.write("isready\n")
        process.stdin.flush()

        while True:
            line = process.stdout.readline()
            #print("STDOUT (ready):", repr(line))

            if not line:
                #print("STDERR:", process.stderr.read())
                raise RuntimeError("Engine exited before sending readyok.")

            if line.strip() == "readyok":
                break

        #print("Sending position and go")
        process.stdin.write(f"position fen {fen}\n")
        process.stdin.write("go\n")
        process.stdin.flush()

        while True:
            line = process.stdout.readline()
            #print("STDOUT (move):", repr(line))

            if not line:
                #print("STDERR:", process.stderr.read())
                raise RuntimeError("Engine exited before returning a move.")

            line = line.strip()
            if line.startswith("bestmove "):
                engine_move = line.split()[1]
                #print("ENGINE MOVE:", engine_move)
                return engine_move

    finally:
        #print("Stopping engine...")
        if process.poll() is None:
            process.stdin.write("quit\n")
            process.stdin.flush()
            process.wait(timeout=5)

        #print("Engine exit code:", process.poll())



def pyturochamp_engine_func(epd, move, rating=-1, modes=["ptc"]):

    results = []

    for mode in modes:

        engine_move = get_pyturochamp_move(epd, mode=mode)

        results.append({
            "rating": rating,
            "mode": mode,
            "engine_move": engine_move,
            "expected_move": move,
            "correct": engine_move == move,
        })


    # Store individual puzzle results
    results_df = pd.DataFrame(results)

    return results_df




def run_evaluation(
        df,
        show_results = False,
        number_of_puzzles = 10,
        depth = 3,
        elo_levels = [1320, 1600, 1900, 2200, 2500],
        pyturo_modes = ["ptc", "bare"],
        save_to_csv = False
):


    engine_results = {}

    for idx, (_, row) in enumerate(df.head(number_of_puzzles).iterrows()):

        epd = row["epd"]
        rating = row["rating"]
        solution_moves = str(row["solution"]).split()
        expected_move = solution_moves[1] if len(solution_moves) > 1 else None

        # ---------------------------------------
        # Run engines
        # ---------------------------------------

        classical_results = classical_chess_engine_func(epd=epd, move=expected_move, rating=rating, depth=depth)
        sunfish_results = sunfish_engine_func(epd=epd, move=expected_move, rating=rating, depth=depth)
        #otter_engine_func(epd=epd)
        #stockfish_results = stockfish_engine_func(epd=epd, move=expected_move, rating=rating, elo_levels=elo_levels)
        #maia_3_results = maia3_engine_func(epd=epd, move=expected_move, rating=rating, elo_levels=elo_levels)
        pyturo_results = pyturochamp_engine_func(epd=epd, move=expected_move, rating=rating, modes=pyturo_modes)


        # ---------------------------------------
        # Collect results
        # ---------------------------------------


        # Classical
        engine_name = "Classical"
        if engine_name not in engine_results:
            engine_results[engine_name] = []
        engine_results[engine_name].append(classical_results)


        # Sunfish
        engine_name = "Sunfish"
        if engine_name not in engine_results:
            engine_results[engine_name] = []
        engine_results[engine_name].append(sunfish_results)

        """
        # Stockfish
        for elo in elo_levels:
            engine_name = f"Stockfish {elo}"

            result_df = stockfish_results[stockfish_results["elo"] == elo].copy()

            if engine_name not in engine_results:
                engine_results[engine_name] = []

            engine_results[engine_name].append(result_df)
        """
        """
        # Maia-3
        for elo in elo_levels:
            engine_name = f"Maia-3 {elo}"

            result_df = maia_3_results[maia_3_results["elo"] == elo].copy()

            if engine_name not in engine_results:
                engine_results[engine_name] = []

            engine_results[engine_name].append(result_df)
        """
        # PyTuroChamp / Bare
        for mode in pyturo_modes:

            engine_name = "PyTuroChamp" if mode == "ptc" else "Bare"
            result_df = pyturo_results[pyturo_results["mode"] == mode].copy()

            if engine_name not in engine_results:
                engine_results[engine_name] = []

            engine_results[engine_name].append(result_df)

        print(f"puzzle number {idx+1} done")

    # combine results
    engine_results = {
        name: pd.concat(results, ignore_index=True)
        for name, results in engine_results.items()
    }

    return engine_results




if __name__ == "__main__":

    """
    dataset_path = "../datasets/dataset_1k_final.csv"
    #dataset_path = "../datasets/artificial_tests.csv"
    df = pd.read_csv(dataset_path)
    #print(df)

    show_results = False
    number_of_puzzles = 100
    depth = 3
    elo_levels = [1320, 2500]
    #pyturo_modes = ["ptc", "bare"]
    pyturo_modes = ["bare"]
    save_to_csv = False

    eval = run_evaluation(df, show_results, number_of_puzzles, depth, elo_levels, pyturo_modes, save_to_csv)
    comparison = compare_engines_by_rating(eval, save_to_csv)
    """
    epd = "3r2k1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 w - -"
    otter_engine_func(epd)


    
