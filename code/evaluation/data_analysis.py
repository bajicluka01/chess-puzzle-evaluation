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


if __name__ == "__main__":

    df_path = "datasets/dataset_test.csv"
    ouput_path = "code/evaluation/data_analysis_plots/"

    plot_rating_dist(input_file=df_path, output_file=ouput_path + "rating_distribution")