import pandas as pd
import numpy as np


def compare_datasets(df_fast, df_slow):

    columns = [
        "meaningful_L1",
        "meaningful_L2",
        "meaningful_L3",
        "branching_L2",
        "branching_L3",
        "avg_branching",
        "narrow_L1",
        "narrow_L2",
        "narrow_L3",
        "distance_L1",
        "distance_L2",
        "distance_L3",
        "pieces_L1",
        "pieces_L2",
        "pieces_L3",
        "all_pieces_involved",
        "winning_no_mate",
        "possible_L1",
        "possible_L2",
        "possible_L3",
        "all_possible_moves",
        "all_narrow_solutions",
        "tree_size",
        "move_ratio_L1",
        "move_ratio_L2",
        "move_ratio_L3",
        "sum_distance",
        "avg_distance"
    ]

    results = []

    for col in columns:

        fast_vals = df_fast[col]
        slow_vals = df_slow[col]

        # exact matches
        exact_match_pct = (fast_vals == slow_vals).mean() * 100

        # mean absolute error
        mae = np.mean(np.abs(fast_vals - slow_vals))

        # correlation
        corr = fast_vals.corr(slow_vals)

        results.append({
            "column": col,
            "exact_match_%": round(exact_match_pct, 2),
            "MAE": round(mae, 4),
            "correlation": round(corr, 4)
        })

    results_df = pd.DataFrame(results)

    return results_df


if __name__ == '__main__':

    # read the files
    data_cp = pd.read_csv("dataset_100k_cp.csv")
    data_static = pd.read_csv("dataset_100k_cp_15_ms.csv")

    data_cp = data_cp.head(n=31)
    data_static = data_static.head(n=31)

    #print(data_cp)
    #print(data_static)
    res = compare_datasets(data_cp, data_static)
    print(res)

    #for col in data_cp.columns:
    #    print(col)