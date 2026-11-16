import chess
<<<<<<< HEAD:code/data_preprocessing/otter_wrapper.py
=======
import json

# for plotting a chess board
>>>>>>> 77a04c1c27d47da6be9d5f721d7b648bb19ee016:code/otter_wrapper/otter_wrapper.py
import matplotlib.pyplot as plt
import cairosvg
from io import BytesIO
import chess.svg
from matplotlib.backends.backend_pdf import PdfPages

<<<<<<< HEAD:code/data_preprocessing/otter_wrapper.py
def plot_boards(epd, moves=None, title=None, solution=None):
=======
def plot_boards(epd, moves=None, title=None, solution=None, save_to_pdf=False, save_otter_moves=False):
>>>>>>> 77a04c1c27d47da6be9d5f721d7b648bb19ee016:code/otter_wrapper/otter_wrapper.py
    """
        Visualizes puzzle from given epd
    """

    board = chess.Board()
    board.set_epd(epd)

    arrows = []

    if moves and (not save_to_pdf or save_otter_moves):
        n = len(moves)

        for rank in reversed(range(n)):
            move_info = moves[rank]
            move = chess.Move.from_uci(move_info["move"])

            alpha = 0.95 - 0.80 * rank / max(n - 1, 1)
            color = f"rgba(0, 128, 0, {alpha:.2f})"

            arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=color))

    svg = chess.svg.board(board=board, size=500, arrows=arrows)
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"))

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.imshow(plt.imread(BytesIO(png)))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=20, fontweight="bold", pad=15)

    if solution:
        import textwrap #TODO why here?

        solution_text = format_solution(epd, solution)
        solution_text = textwrap.fill(solution_text, width=40, break_long_words=False, break_on_hyphens=False)

<<<<<<< HEAD:code/data_preprocessing/otter_wrapper.py
        fig.text(0.5, 0.00, f"Solution: {solution_text}", ha="center", va="bottom", fontsize=19)
=======
        fig.text(
            0.5, 0.0,
            f"Solution: {solution_text}",
            ha="center",
            va="bottom",
            fontsize=19
        )
>>>>>>> 77a04c1c27d47da6be9d5f721d7b648bb19ee016:code/otter_wrapper/otter_wrapper.py

    plt.subplots_adjust(top=0.85, bottom=0.15)

    if save_to_pdf:
        return fig
    else:
        plt.show()
        plt.close(fig)

def otter_engine_func(epd, engine):
    """
        Returns results of otter machine 

        Result is a dictionary:
        {
            'fen': 'r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w - - 0 1',
            'win_probability': -0.47298043966293335,
            'moves': [
                {
                    'move': 'h2g3',
                    'probability': 0.4267266094684601
                },
                ...
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

    board = chess.Board()
    board.set_epd(epd)

    result = engine.predict(
        fen=board.fen(),
        player_elo=3000,
        opponent_elo=3000,
        history_moves=[],
        time_control="1000+0",
        clock_fraction=0.9,
        top_k=board.legal_moves.count()
    )

    return result

def format_solution(epd, solution):
    """
        Formats solution 

        For example:
            from: c7c8q g8c8 f4f6
            to  : 1.c8=Q Rxc8 2.Qf6+
    """
    board = chess.Board()
    board.set_epd(epd)

    formatted = []

    for idx, uci in enumerate(solution.split()):
        move = chess.Move.from_uci(uci)
        san = board.san(move)

        if board.turn == chess.WHITE:
            formatted.append(f"{board.fullmove_number}.{san}")
        elif idx == 0:
            formatted.append(f"{board.fullmove_number}...{san}")
        else:
            formatted.append(san)

        board.push(move)

    return " ".join(formatted)


def plot_otter(positions, engine, number_of_moves=5,
               save_to_pdf=False, save_otter_moves=False, pdf_path="example.pdf"):
    """
        Goes through all positions in input "positions"
        Prints the Otter machine resulting moves 
        Plots the puzzle with resulting moves 

        Number of resulting moves is in input "number_of_moves". 
    """

    pdf = PdfPages(pdf_path) if save_to_pdf else None

    for idx, position in enumerate(positions, start=1):

        title = position["title"]
        epd = position["epd"]
        solution = position["solution"]
        rating = position["rating"]

        print("\n" + "=" * 65)
        print(f"{title} ({idx}/{len(positions)}) | Rating: {rating:.1f}")
        print("=" * 65)

        board = chess.Board()
        board.set_epd(epd)

        print(f"EPD: {epd}")
        print(f"Side to move: {'White' if board.turn else 'Black'}")
        print(f"Solution: {format_solution(epd, solution)}")

        # Get predictions
        result = otter_engine_func(epd, engine)

        moves = sorted( result["moves"], key=lambda x: x["probability"], reverse=True)

        print("\nPREDICTED MOVES")
        print("-" * 55)
        print(f"{'Rank':<6}{'Move (SAN)':<15}{'Move (UCI)':<15}{'Probability':>12}")
        print("-" * 55)

        for rank, move_info in enumerate(moves[:number_of_moves], start=1):
            uci = move_info["move"]
            probability = move_info["probability"]

            try:
                san = board.san(chess.Move.from_uci(uci))
            except ValueError:
                san = "Invalid move"

            print(
                f"{rank:<6}"
                f"{san:<15}"
                f"{uci:<15}"
                f"{probability * 100:>11.2f}%"
            )

        print("-" * 55)


        if not save_to_pdf or save_otter_moves:
            figure_title = (
                f"{title}\n"
                f"Otter's top {min(number_of_moves, len(moves))} moves\n"
                f"Rating: {rating:.1f}"
            )
        else:
            figure_title = f"{title}\nRating: {rating:.1f}"

        fig = plot_boards(
            epd,
            moves=moves[:number_of_moves],
            title=figure_title,
            solution=solution,
            save_to_pdf=save_to_pdf,
            save_otter_moves=save_otter_moves
        )

<<<<<<< HEAD:code/data_preprocessing/otter_wrapper.py
if __name__ == "__main__":
    positions_from_study = [
        {
            "epd": "6rk/1RP4q/p7/3r4/P4Q1P/6P1/3p3K/8 w - -",
            "solution": "c7c8q g8c8 f4f6 h8g8 f6e6 g8h8 e6c8 d5d8 c8d8 h7g8 d8d4 g8g7 d4g7",
            "rating": 1492.5
        },
        {
            "epd": "7b/p2qr1k1/1pQ3P1/2p1p3/2P4R/6K1/6PP/8 w - -",
            "solution": "h4h7 g7g8 c6d7 e7d7 h7d7 a7a6 h2h4 h8g7 d7b7 b6b5 c4b5 a6b5 b7b5 g7f8 b5b7 f8g7",
            "rating": 1495.3
        },
        {
            "epd": "8/8/3p4/p1pP4/P1P5/1P4KN/7p/7k w - -",
            "solution": "b3b4 a5b4 g3f2 b4b3 f2f1 b3b2 h3f2",
            "rating": 1875.2
        },
        {
            "epd": "3rk1r1/1p1nbb1p/p1p2p2/P4P2/4R3/1qN5/1PN1P1BP/2Q2RK1 w - -",
            "solution": "c2a1 b3b4 e4b4 e7b4 c1e3 e8f8 g1h1 b4a5 e3h6 f8e7 c3e4 f7d5 g2f3 d5e4 h6h7",
            "rating": 1878.1
        },
        {
            "epd": "2kr1b1r/ppp3pp/2N5/q4p2/6B1/2N1n3/PPP2PPP/R2Q1RK1 b - -",
            "solution": "b7c6 d1f3 e3g4 h2h3 g4f6 f3c6 c8b8 a1d1 f8d6 d1d3 a5b6 c6c4 b6b4 c4e6 h8e8 e6f5 b4b2",
            "rating": 1878.1
        },
        {
            "epd": "6r1/3b1ppk/4p3/3pP2P/8/5NR1/q4P2/2Q3K1 w - -",
            "solution": "f3g5 h7h6 g5f7 h6h7 c1g5 a2a1 g1h2 a1b1 g5g6 b1g6 h5g6",
            "rating": 1883.3
        },
        {
            "epd": "r2qkb1r/3bpp2/p1np1p2/1p3P2/3NP2p/2N5/PPPQB1PP/R4RK1 b kq -",
            "solution": "f8h6 d2d3 d8b6 c3d5 b6d4 d3d4 c6d4 d5c7 e8d8 c7a8 h6e3 f1f2 d4e2 g1f1 e3f2 f1f2",
            "rating": 2230.9
        },
        {
            "epd": "5r1k/1p1b1p1p/p2ppb2/5P1B/1q6/1Pr3R1/2PQ2PP/5R1K w - -",
            "solution": "d2h6 c3g3 h5g6 g3g6 f5g6 f7g6 h6f8",
            "rating": 2233.1
        },
        {
            "epd": "4r1k1/1pQb1pp1/7p/3P3n/p7/P1pP2PP/1q3PBK/3RB3 b - -",
            "solution": "c3c2 c7d7 h5f6 d7a4 b7b5 a4c2 b2c2 d1a1 f6d5 h2g1 d5c3 e1c3 c2c3 a1d1 e8e2 g2e4 c3b2",
            "rating": 2237.5
        },
        {
            "epd": "3rr1k1/1p3ppp/p1n1b3/2B5/qP2BQ2/3R4/P4PPP/R5K1 w - -",
            "solution": "f4h4 e6f5 e4f5 g7g6 d3d7 d8d7 f5d7 a4a2 h4f6 e8e2 d7c6 a2d2 c6b5 e2e6 f6f3",
            "rating": 2238.5
        },
        {
            "epd": "r5k1/1p3p2/p2Q2p1/3P2P1/2n5/1N6/PP2q2r/KR3R2 w - -",
            "solution": "d6c7 h2h7 b3d4 e2d3 b1d1 c4d2 c7f4 d3f1 d1f1 d2f1 f4f1 a8d8 f1g2 h7h4 d4e2 d8e8 a1b1 h4e4",
            "rating": 2243.4
        },
        {
            "epd": "r1b1kr2/1p3pQp/2N5/pB1p4/1P2q3/2K5/1P1B1PPP/n7 w q -",
            "solution": "c6b8 e8d8 g7f8 d8c7 f8c5 c7b8 c5d6 b8a7 d2e3 e4e3 f2e3 a5b4 c3b4",
            "rating": 2274.9
        }
    ]

    from otter_chess import OtterModel #TODO why here?
    otter_engine = OtterModel(device="cpu")

    plot_otter(positions_from_study, otter_engine, number_of_moves=5)
=======
        if save_to_pdf:
            pdf.savefig(fig)
            plt.close(fig)

    if save_to_pdf:
        pdf.close()
        print(f"\nPDF saved to: {pdf_path}")

if __name__ == "__main__":


    # 12 positions from the study
    with open("otter_wrapper/positions_json/positions_from_study.json", "r") as f:
        positions_from_study = json.load(f)

    
    # big misclassifications from lichess puzzles
    with open("otter_wrapper/positions_json/misclassifications.json", "r") as f:
        misclassifications = json.load(f)


    # correct classifications from lichess puzzles
    with open("otter_wrapper/positions_json/correct_classifications.json", "r") as f:
        correct_classifications = json.load(f)


    # saving purpose
    save_to_pdf = False
    save_otter = False
    pdf_path = "otter_wrapper/positions_pdf/positions_from_study_with_otter.pdf"

    # run otter on chosen positions with adjustable number of moves that are shown
    from otter_chess import OtterModel  
    otter_engine = OtterModel(device="cpu")
    plot_otter(
        positions_from_study, 
        otter_engine, 
        number_of_moves=5, 
        save_to_pdf=save_to_pdf, pdf_path=pdf_path, save_otter_moves=save_otter)

>>>>>>> 77a04c1c27d47da6be9d5f721d7b648bb19ee016:code/otter_wrapper/otter_wrapper.py
