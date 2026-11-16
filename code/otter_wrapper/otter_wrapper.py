import chess
import json

# for plotting a chess board
import matplotlib.pyplot as plt
import cairosvg
from io import BytesIO
import chess.svg
from matplotlib.backends.backend_pdf import PdfPages

def plot_boards(epd, moves=None, title=None, solution=None, save_to_pdf=False, save_otter_moves=False):
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

            arrows.append(
                chess.svg.Arrow(
                    move.from_square,
                    move.to_square,
                    color=color
                )
            )

    svg = chess.svg.board(
        board=board,
        size=500,
        arrows=arrows
    )

    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"))

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.imshow(plt.imread(BytesIO(png)))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=20, fontweight="bold", pad=15)

    if solution:
        import textwrap

        solution_text = format_solution(epd, solution)
        solution_text = textwrap.fill(
            solution_text,
            width=40,
            break_long_words=False,
            break_on_hyphens=False,
        )

        fig.text(
            0.5, 0.0,
            f"Solution: {solution_text}",
            ha="center",
            va="bottom",
            fontsize=19
        )

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

        moves = sorted(
            result["moves"],
            key=lambda x: x["probability"],
            reverse=True
        )

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

