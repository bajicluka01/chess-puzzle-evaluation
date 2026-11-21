# for plotting a chess board
import matplotlib.pyplot as plt
import cairosvg
from io import BytesIO
import chess.svg
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
import shap
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def plot_boards(epd): 
    """
        Visualizes puzzle from given epd
    """

    board = chess.Board()
    board.set_epd(epd)

    svg = chess.svg.board(board=board, size=400)
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"))

    plt.figure(figsize=(5, 5))
    plt.imshow(plt.imread(BytesIO(png)))
    plt.axis("off")
    plt.show()

def plot_rating_dist(input_file, output_file, bins=50):
    df = pd.read_csv(input_file)
    ratings = pd.to_numeric(df["rating"], errors="coerce").dropna()

    plt.figure(figsize=(12, 6))

    # bins for every 100 rtg points
    bin_edges = range((int(ratings.min()) // 100) * 100, ((int(ratings.max()) // 100) + 2) * 100, 100)

    classwidth = (ratings.max() - ratings.min()) / 3
    line1 = ratings.min() + classwidth
    line2 = ratings.min() + 2 * classwidth 

    rating_list = ratings.to_list()
    rating_list.sort()
    line1a = rating_list[int(len(rating_list)/3)]
    line2a = rating_list[int(2*len(rating_list)/3)]

    sns.histplot(ratings, bins=bin_edges, color="#70B0E0", edgecolor="black")
    plt.axvline(x=line1, color="#12239E", linewidth=3, label=f"Equidistant split: Easy / Medium ({round(line1)})")
    plt.axvline(x=line2, color="red", linewidth=3, label=f"Equidistant split: Medium / Hard ({round(line2)})")
    plt.axvline(x=line1a, color="green", linewidth=3, ls="--", label=f"Balanced split: Easy / Medium ({round(line1a)})")
    plt.axvline(x=line2a, color="orange", linewidth=3, ls="--", label=f"Balanced split: Medium / Hard ({round(line2a)})")

    plt.title("Puzzle Rating Distribution", fontsize=17, fontweight="bold")
    plt.xlabel("Puzzle rating", fontsize=14, fontweight="bold")
    plt.ylabel("Number of puzzles", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_comparison(plot_data, save_to_png, path_png):
    ax = plot_data.plot(kind="line", marker="o", figsize=(12, 7))
    ax.set_title("Engine Accuracy by Puzzle Rating")
    ax.set_xlabel("Puzzle rating")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(plot_data.index)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Engine", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    if save_to_png:
        plt.savefig(path_png, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

def heatmap(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    board_color = LinearSegmentedColormap.from_list(
        "board",
        ["#FFCE9E", "#D18B47"]
    )
    cax = ax.imshow(cm, interpolation='nearest', cmap=board_color)
    fig.colorbar(cax)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", 
                    ha="center", va="center", color="black", fontsize=14)
    
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def beeswarm(class_shap_values, model_name, class_idx=None):
    shap.plots.beeswarm(class_shap_values, max_display=20, show=False)

    if class_idx is not None:
        plt.title(f"SHAP Summary: {model_name}\n"
            f"for Class {class_idx}",
            fontsize=17,
            fontweight="bold"
        )
    else:
        plt.title(f"SHAP Summary: {model_name}", fontsize=17, fontweight="bold")

    plt.tight_layout()
    path = f"evaluation/shap_plots/{model_name}_class_{class_idx}_SHAP_summary.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def bar(class_shap_values, model_name, class_idx=None, dpi=200):
    shap.plots.bar(class_shap_values, max_display=20, show=False)

    if class_idx is not None:
        plt.title(f"SHAP Summary: {model_name}\n"
            f"for Class {class_idx}",
            fontsize=17,
            fontweight="bold"
        )
    else:
        plt.title(f"SHAP Summary: {model_name}", fontsize=17, fontweight="bold")

    plt.tight_layout()
    path = f"evaluation/shap_plots/{model_name}_class_{class_idx}_SHAP_summary.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def waterfall(sv, model_name, instance, actual, predicted, class_idx=None, mode="regression"):
    shap.plots.waterfall(sv, max_display=20, show=False)
    if mode == "regression":
        plt.title(
            f"SHAP Explanation: {model_name}\n"
            f"for Instance {instance}\n"
            f"Actual rating: {actual:.0f} | Predicted rating: {predicted:.0f}",
            fontsize=17,
            fontweight="bold"
        )
        path = f"evaluation/shap_plots/{model_name}_waterfall_{instance}.png"
    elif mode == "classification": 
        plt.title(
            f"SHAP Explanation: {model_name}\n"
            f"for Class {class_idx}, Instance {instance}\n"
            f"Actual class: {actual} | Predicted class: {predicted}",
            fontsize=17,
            fontweight="bold"
        )
        path = f"evaluation/shap_plots/{model_name}_class_{class_idx}_waterfall_{instance}.png"
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def lime_plot(exp, j):
    from pathlib import Path
    output_dir = Path(__file__).resolve().parent.parent.parent / "plots" / "explanations" / "lime"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = exp.as_pyplot_figure()
    ax = fig.axes[0]

    # Vertical grid lines
    ax.grid(axis="x", color="black", linewidth=0.5, alpha=0.2)

    # Horizontal grid lines between bars
    bar_positions = [bar.get_y() + bar.get_height() / 2 for bar in ax.patches]

    ax.set_yticks(np.array(bar_positions) + 0.5, minor=True)
    ax.grid(axis="y", which="minor", color="black", linewidth=0.5, alpha=0.2)
    ax.grid(axis="y", which="major", visible=False)
    ax.axvline(x=0, color="black", linewidth=2, alpha=0.6)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_axisbelow(True)
    ax.set_title(
        f"LIME Explanation - Puzzle {j}\n"
        f"Test position: {j}",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )

    fig.set_size_inches(16, 10)
    fig.tight_layout()

    fig.savefig(output_dir / f"lime_test_{j}_puzzle_{j}.pdf", bbox_inches="tight")
    plt.close(fig)

# TODO what is this and why do we need it?
def plot_misclassifications(epd, puzzle_id, real_class, predicted_class, solution, to_move):
    """Save the puzzle board with labels and a readable solution."""
    board = chess.Board()
    board.set_epd(epd)

    board.turn = chess.WHITE if int(to_move) == 1 else chess.BLACK
    player = "White" if board.turn == chess.WHITE else "Black"

    # Skip the opponent's first move and convert UCI moves to SAN.
    moves = [chess.Move.from_uci(move) for move in solution.split()[1:]]
    readable_solution = board.variation_san(moves)

    svg = chess.svg.board(board=board, size=400)
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=2000, output_height=2000)

    output_dir = Path(__file__).resolve().parent / "misclassifications_boards"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(plt.imread(BytesIO(png)))
    ax.set_title(
        f"Puzzle {puzzle_id}\n"
        f"Real: {real_class} | Predicted: {predicted_class}\n"
        f"{player} to move"
    )
    ax.axis("off")

    fig.subplots_adjust(bottom=0.17)
    fig.text(0.5, 0.11, f"Solution: {readable_solution}", ha="center", fontsize=11, wrap=True)

    fig.savefig(output_dir / f"puzzle_{puzzle_id}.pdf", bbox_inches="tight", dpi=400)
    plt.close(fig)

# TODO does this even work? svg is never used
def plot_group(group, group_name, original_df, n=3,  start=0):
    """
    Visualizes n random puzzles from a difficulty group.
    """

    samples = group.iloc[start:start + n]

    if samples.empty:
        print(f"No puzzles found in {group_name}")
        return

    fig, axes = plt.subplots(1, len(samples), figsize=(5 * len(samples), 5))

    axes = np.atleast_1d(axes)

    for _, (idx, _) in zip(axes, samples.iterrows()):
        epd = original_df.loc[idx, "epd"]
        board = chess.Board()
        board.set_epd(epd)
        svg = chess.svg.board(board=board, size=400)

    fig.suptitle(group_name, fontsize=17, fontweight="bold")
    plt.tight_layout()
    plt.show()
