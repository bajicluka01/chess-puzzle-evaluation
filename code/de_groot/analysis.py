import json

import chess
import json
import matplotlib.pyplot as plt
import cairosvg
from io import BytesIO
import chess.svg
import textwrap



def format_solution(epd, solution):
    """
        Formats solution 

        For example:
            from: c7c8q g8c8 f4f6
            to  : 1.c8=Q Rxc8 2.Qf6+
    """
    board = chess.Board(epd)
    #board.set_epd(epd)

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

def plot_boards(epd, moves=None, title=None, solution=None, save_to_pdf=False, save_otter_moves=False):
    """
        Visualizes puzzle from given epd
    """

    board = chess.Board(epd)
    #board.set_epd(epd)

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

        solution_text = format_solution(epd, solution)
        solution_text = textwrap.fill(solution_text, width=40, break_long_words=False, break_on_hyphens=False)

        fig.text(0.5, 0.00, f"Solution: {solution_text}", ha="center", va="bottom", fontsize=19)

    plt.subplots_adjust(top=0.85, bottom=0.15)

    if save_to_pdf:
        return fig
    else:
        plt.show()
        plt.close(fig)


if __name__ == "__main__":

    with open("de_groot/positions_combined.json", "r", encoding="utf-8") as f:
        positions = json.load(f)

    for group_name, group_positions in positions.items():

        print(f"\n{'=' * 80}")
        print(f"GROUP: {group_name.replace('_', ' ').upper()}")
        print(f"{'=' * 80}")

        for idx, position in enumerate(group_positions, start=1):

            title = position["title"]
            epd = position["epd"]
            descriptions = position.get("descriptions", {})

            print(f"\n{'=' * 80}")
            print(f"{title} ({idx}/{len(group_positions)})")
            print(f"{'=' * 80}")

            print(f"EPD: {epd}")

            # Print all players' descriptions
            for player, description in descriptions.items():
                print(f"\n{'-' * 60}")
                print(f"{player}:")
                print(f"{'-' * 60}")
                print(description)

            # Display the chessboard
            plot_boards(
                epd,
                title=title
            )