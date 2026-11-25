# standard libraries
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# for plotting chess board
import cairosvg
from io import BytesIO
import chess.svg

# for general plotting 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import shap
import textwrap

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def plot_boards(
        epd,
        ax=None,
        model_name=None,
        instance=None,
        actual=None,
        predicted=None,
        mode="regression",
        show_plot=True,
        save_plot=False,
        plot_path=None,
        solution=None,
        to_move=None,
        skip_first_move=False
    ):
    """
    Visualizes puzzle from given EPD.

    If ax is provided, draws on that axis.
    Otherwise, creates a standalone chessboard figure
    with optional title, readable solution and saving.
    """

    board = chess.Board()
    board.set_epd(epd)

    if to_move is not None:
        board.turn = chess.WHITE if int(to_move) == 1 else chess.BLACK

    player = "White" if board.turn == chess.WHITE else "Black"

    # Convert solution from UCI to SAN
    readable_solution = None

    if solution is not None:
        moves = solution.split()

        if skip_first_move:
            moves = moves[1:]

        moves = [chess.Move.from_uci(move) for move in moves]
        readable_solution = board.variation_san(moves)

    # Generate high-resolution chessboard
    svg = chess.svg.board(board=board, size=400)

    png = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=2000,
        output_height=2000
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 7))
        standalone = True
    else:
        standalone = False

    ax.imshow(plt.imread(BytesIO(png)))
    ax.axis("off")

    if standalone:

        title = []

        if model_name is not None:
            title.append(model_name)

        if instance is not None:
            title.append(f"Instance {instance}")

        if actual is not None and predicted is not None:
            if mode == "regression":
                title.append(
                    f"Actual rating: {actual:.0f} | "
                    f"Predicted rating: {predicted:.0f}"
                )
            elif mode == "classification":
                title.append(
                    f"Actual class: {actual} | "
                    f"Predicted class: {predicted}"
                )
            else:
                raise ValueError("Unknown plot mode")

        title.append(f"{player} to move")

        ax.set_title(
            "\n".join(title),
            fontsize=15,
            fontweight="bold",
            pad=20
        )
        fig.tight_layout()

        # Add readable solution below the board
        if readable_solution is not None:

            fig.subplots_adjust(bottom=0.17)
            fig.text(
                0.5,
                0.11,
                f"Solution: {readable_solution}",
                ha="center",
                fontsize=11,
                wrap=True
            )

        if save_plot:
            if plot_path is None:
                raise ValueError(
                    "plot_path must be provided when save_plot=True"
                )

            plot_path = Path(plot_path)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                plot_path,
                dpi=400,
                bbox_inches="tight"
            )

        if show_plot:
            plt.show()

        plt.close(fig)


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

def shap_beeswarm(
        class_shap_values,
        model_name,
        class_idx=None,
        columns=None,
        show_plot=False,
        save_plot=False,
        plot_dir=None
    ):
    """
        Plotting global SHAP explanation for dataset 
    """

    # select only the columns we want to display
    if columns is not None:
        indices = [
            i for i, name in enumerate(class_shap_values.feature_names)
            if name in columns
        ]
        class_shap_values = class_shap_values[:, indices]


    shap.plots.beeswarm(class_shap_values, max_display=20, show=False)

    if class_idx is not None:
        plt.title(f"SHAP Global Explanation: {model_name}\n"
            f"for Class {class_idx}",
            fontsize=17,
            fontweight="bold"
        )
    else:
        plt.title(f"SHAP Global Explanation: {model_name}", fontsize=17, fontweight="bold")


    plt.tight_layout()


    if save_plot:
        if plot_dir is None:
            raise ValueError("plot_dir must be provided when save_plot=True")

        os.makedirs(plot_dir, exist_ok=True)

        if class_idx is not None:
            path = os.path.join(
                plot_dir,
                f"{model_name}_global_beeswarm_class_{class_idx}.pdf"
            )
        else: 
            path = os.path.join(
                plot_dir,
                f"{model_name}_global_beeswarm_regression.pdf"
            )

        plt.savefig(path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close()


def shap_waterfall(
        sv,
        model_name,
        instance,
        actual,
        predicted,
        class_idx=None,
        mode="regression",
        columns=None,
        show_plot=False,
        save_plot=False,
        plot_dir=None,
        epd=None
    ):
    """
        Plotting local SHAP explanation for instance 
        If epd is not None, it also plots position besides it 
    """

    # select only the columns we want to display
    if columns is not None:
        indices = [
            i for i, name in enumerate(sv.feature_names)
            if name in columns
        ]

        other_indices = [
            i for i in range(len(sv.values))
            if i not in indices
        ]

        selected_values = np.asarray(sv.values)[indices]
        selected_names = [sv.feature_names[i] for i in indices]

        if sv.data is not None:
            selected_data = np.asarray(sv.data)[indices]
        else:
            selected_data = np.full(len(indices), np.nan)

        # Group excluded features to preserve the total contribution
        if other_indices:
            selected_values = np.append(
                selected_values,
                np.sum(np.asarray(sv.values)[other_indices])
            )
            selected_data = np.append(selected_data, np.nan)
            selected_names.append("Other features")

        sv = shap.Explanation(
            values=selected_values,
            base_values=np.asarray(sv.base_values).item(),
            data=selected_data,
            feature_names=selected_names
        )

    shap.plots.waterfall(sv, max_display=20, show=False)
    fig = plt.gcf()

    if mode == "regression":
        title = (
            f"SHAP Local Explanation: {model_name}\n"
            f"for Instance {instance}\n"
            f"Actual rating: {actual:.0f} | Predicted rating: {predicted:.0f}"
        )
        filename = f"{model_name}_{instance}_waterfall_regression.pdf"

    elif mode == "classification": 
        title = (
            f"SHAP Local Explanation: {model_name}\n"
            f"for Class {class_idx}, Instance {instance}\n"
            f"Actual class: {actual} | Predicted class: {predicted}"
        )
        filename = f"{model_name}_{instance}_waterfall_class_{class_idx}.pdf"
    else:
        raise ValueError("Unknown SHAP plot mode")


    # If EPD is available, add chessboard beside the waterfall
    if epd is not None:

        # Make figure wider
        width, height = fig.get_size_inches()
        fig.set_size_inches(18, max(height, 9))

        # Move existing SHAP axes to the right
        for ax in fig.axes:
            pos = ax.get_position()
            ax.set_position([
                0.23 + pos.x0 * 0.75,
                pos.y0,
                pos.width * 0.75,
                pos.height
            ])

        # Add chessboard on the left
        board_ax = fig.add_axes([0.09, 0.28, 0.27, 0.48])
        plot_boards(epd, ax=board_ax)

        # Put title at the top of the combined figure
        fig.suptitle(title, fontsize=17, fontweight="bold", y=0.98)

    else:
        plt.title(title, fontsize=17, fontweight="bold")
        plt.tight_layout()

    if save_plot:
        if plot_dir is None:
            raise ValueError("plot_dir must be provided when save_plot=True")

        os.makedirs(plot_dir, exist_ok=True)

        path = os.path.join(plot_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(fig)


def shap_bar(
        class_shap_values,
        model_name,
        instance=None,
        actual=None,
        predicted=None,
        class_idx=None,
        mode="regression",
        columns=None,
        show_plot=False,
        save_plot=False,
        plot_dir=None,
        epd=None
    ):
    """
    Plot SHAP feature contributions.

    Global explanation: pass SHAP values for multiple instances.
    Local explanation: pass SHAP values for one instance.

    If local and epd is provided, plot chess position beside it.
    """

    is_local = instance is not None

    # Select columns to display
    if columns is not None:
        indices = [
            i for i, name in enumerate(class_shap_values.feature_names)
            if name in columns
        ]


        if is_local:
            class_shap_values = shap.Explanation(
                values=np.asarray(class_shap_values.values)[indices],
                base_values=np.asarray(class_shap_values.base_values).item(),
                data=(
                    np.asarray(class_shap_values.data)[indices]
                    if class_shap_values.data is not None
                    else None
                ),
                feature_names=[
                    class_shap_values.feature_names[i]
                    for i in indices
                ]
            )

        else:
            class_shap_values = class_shap_values[:, indices]

    shap.plots.bar(class_shap_values, max_display=20, show=False)
    fig = plt.gcf()

    if is_local:

        if mode == "regression":
            title = (
                f"SHAP Local Explanation: {model_name}\n"
                f"for Instance {instance}\n"
                f"Actual rating: {actual:.0f} | Predicted rating: {predicted:.0f}"
            )
            filename = f"{model_name}_{instance}_bar_regression.pdf"

        elif mode == "classification":
            title = (
                f"SHAP Local Explanation: {model_name}\n"
                f"for Class {class_idx}, Instance {instance}\n"
                f"Actual class: {actual} | Predicted class: {predicted}"
            )
            filename = f"{model_name}_{instance}_bar_class_{class_idx}.pdf"

        else:
            raise ValueError("Unknown SHAP plot mode")

    else:

        if class_idx is not None:
            title = (
                f"SHAP Global Explanation: {model_name}\n"
                f"for Class {class_idx}"
            )
            filename = f"{model_name}_global_bar_class_{class_idx}.pdf"

        else:
            title = f"SHAP Global Explanation: {model_name}"
            filename = f"{model_name}_global_bar_regression.pdf"


    # Add chess position for local explanations
    if is_local and epd is not None:

        fig.set_size_inches(22, max(fig.get_size_inches()[1], 10))

        # coordinates for shap
        for ax in fig.axes:
            ax.set_position([0.405, 0.15, 0.48, 0.72])

        # coordinates for chess board
        board_ax = fig.add_axes([0.03, 0.30, 0.25, 0.40])
        plot_boards(epd, ax=board_ax)

        fig.suptitle(title, fontsize=17, fontweight="bold", y=0.98)
        
    else:
        plt.title(title, fontsize=17, fontweight="bold")
        plt.tight_layout()

    if save_plot:
        if plot_dir is None:
            raise ValueError("plot_dir must be provided when save_plot=True")

        os.makedirs(plot_dir, exist_ok=True)

        path = os.path.join(plot_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(fig)


def lime_plot(
        exp,
        model_name,
        instance,
        actual,
        predicted,
        class_idx=None,
        class_label=None,
        mode="regression",
        show_plot=False,
        save_plot=False,
        plot_dir=None,
        epd=None
    ):
    """
        Plot local LIME explanation for an instance.
        If epd is provided, also plot the chess position.
    """

    if mode == "regression":
        fig = exp.as_pyplot_figure()
        title = (
            f"LIME Local Explanation: {model_name}\n"
            f"for Instance {instance}\n"
            f"Actual rating: {actual:.0f} | "
            f"Predicted rating: {predicted:.0f}"
        )
        filename = f"{model_name}_lime_{instance}_regression.pdf"

    elif mode == "classification":
        fig = exp.as_pyplot_figure(label=class_idx)
        title = (
            f"LIME Local Explanation: {model_name}\n"
            f"for Class {class_label}, Instance {instance}\n"
            f"Actual class: {actual} | Predicted class: {predicted}"
        )
        filename = (
            f"{model_name}_lime_{instance}_class_{class_label}.pdf"
        )

    else:
        raise ValueError("Unknown LIME plot mode")


    ax = fig.axes[0]
    ax.set_title("")
    ax.grid(axis="x", color="black", linewidth=0.5, alpha=0.2)
    bar_positions = [bar.get_y() + bar.get_height() / 2 for bar in ax.patches]

    ax.set_yticks(np.array(bar_positions) + 0.5, minor=True)
    ax.grid(axis="y", which="minor", color="black", linewidth=0.5, alpha=0.2)
    ax.grid(axis="y", which="major", visible=False)
    ax.axvline(x=0, color="black", linewidth=2, alpha=0.6)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("LIME contribution", fontsize=13)

    # Add chessboard if available
    if epd is not None:

        fig.set_size_inches(22, 10)

        # Reserve space for the board and feature labels
        ax.set_position([0.38, 0.10, 0.50, 0.75])

        board_ax = fig.add_axes([0.03, 0.28, 0.25, 0.44])
        plot_boards(epd, ax=board_ax)

        fig.suptitle(
            title,
            fontsize=17,
            fontweight="bold",
            y=0.98
        )

    else:
        fig.set_size_inches(16, 10)

        ax.set_title(
            title,
            fontsize=17,
            fontweight="bold",
            pad=20
        )

        fig.tight_layout()

    if save_plot:
        if plot_dir is None:
            raise ValueError(
                "plot_dir must be provided when save_plot=True"
            )

        os.makedirs(plot_dir, exist_ok=True)

        path = os.path.join(plot_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(fig)
