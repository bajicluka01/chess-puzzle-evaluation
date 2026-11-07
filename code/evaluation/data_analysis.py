import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_rating_dist(input_file, output_file, bins=50):

    df = pd.read_csv(input_file)
    ratings = pd.to_numeric(df["rating"], errors="coerce").dropna()

    plt.figure(figsize=(12, 6))

    # bins for every 100 rtg points
    bin_edges = range(
        (int(ratings.min()) // 100) * 100,
        ((int(ratings.max()) // 100) + 2) * 100,
        100
    )

    classwidth = (ratings.max() - ratings.min()) / 3
    line1 = ratings.min() + classwidth
    line2 = ratings.min() + 2 * classwidth 

    rating_list = ratings.to_list()
    rating_list.sort()
    line1a = rating_list[int(len(rating_list)/3)]
    line2a = rating_list[int(2*len(rating_list)/3)]

    # main plot
    sns.histplot(
        ratings,
        bins=bin_edges,
        color="#70B0E0",
        edgecolor="black"
    )

    # line between easy and medium
    plt.axvline(
        x=line1,
        color="#12239E",
        linewidth=3,
        label=f"Equidistant split: Easy / Medium ({round(line1)})"
    )

    # line between medium and hard
    plt.axvline(
        x=line2,
        color="red",
        linewidth=3,
        label=f"Equidistant split: Medium / Hard ({round(line2)})"
    )

    plt.axvline(
        x=line1a,
        color="green",
        linewidth=3,
        ls="--",
        label=f"Balanced split: Easy / Medium ({round(line1a)})"
    )

    # line between medium and hard
    plt.axvline(
        x=line2a,
        color="orange",
        linewidth=3,
        ls="--",
        label=f"Balanced split: Medium / Hard ({round(line2a)})"
    )

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

    print(f"Plot saved to: {output_file}")




def compare_engines_by_rating(
    engine_results_path="evaluation/engine_comparison/engine_results_final.csv", 
    save_to_csv=False, 
    path_csv="evaluation/engine_comparison/engine_accuracies_final.csv", 
    save_to_png=False, 
    path_png="evaluation/engine_comparison/engine_plot_final.png"
):

    # Read engine results
    results = pd.read_csv(engine_results_path)

    if results.empty:
        return pd.DataFrame()


    # Make sure correct is boolean
    results["correct"] = results["correct"].fillna(False).astype(bool)

    # Create rating bins
    results["rating_bin"] = (results["rating"] // 100) * 100

    # Calculate accuracy for each engine and rating bin
    stats = (
        results
        .groupby(["rating_bin", "engine"])["correct"]
        .agg(["size", "mean"])
        .reset_index()
    )

    stats["accuracy"] = stats["mean"] * 100

    # Pivot engines into columns
    comparison = stats.pivot(
        index="rating_bin",
        columns="engine",
        values="accuracy"
    ).sort_index()

    # Total number of puzzles per rating bin
    total_by_rating = (
        results
        .groupby("rating_bin")
        .size()
    )

    comparison.insert(0, "Total", total_by_rating)

    # Format rating bins
    comparison.index = comparison.index.map(
        lambda r: f"{int(r)} - {int(r) + 99}"
    )
    comparison.index.name = "Puzzle rating"

    comparison["Total"] = comparison["Total"].astype(int)

    print("\n=== Engine accuracy by puzzle rating ===\n")

    print(
        comparison.to_string(
            col_space=12,
            formatters={
                col: "{:.2f}%".format
                for col in comparison.columns
                if col != "Total"
            }
        )
    )

    if save_to_csv:
        comparison.to_csv(path_csv, float_format="%.2f")

    # Plot engine accuracy by puzzle rating
    accuracy_columns = [
        col for col in comparison.columns if col != "Total"
    ]

    plot_data = comparison[accuracy_columns].copy()
    plot_data.index = [
        int(label.split(" - ")[0]) for label in plot_data.index
    ]

    ax = plot_data.plot(
        kind="line",
        marker="o",
        figsize=(12, 7)
    )

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

    return comparison


if __name__ == "__main__":

    df_path = "datasets/dataset_test.csv"
    ouput_path = "code/evaluation/data_analysis_plots/"

    #plot_rating_dist(input_file=df_path, output_file=ouput_path + "rating_distribution")

    compare_engines_by_rating()