import pandas as pd

df_engine = pd.read_csv("../evaluation/engine_comparison/engine_results_final.csv")
df_dataset = pd.read_csv("../../datasets/dataset_test.csv")

# True/False -> 1/0
df_engine["correct"] = df_engine["correct"].astype(int)

# one row per epd, one column per engine
wide = (
    df_engine
    .drop_duplicates(subset=["epd", "engine"])
    .pivot(index="epd", columns="engine", values="correct")
    .astype("Int64")            # nullable int, so a missing engine stays <NA>
)
wide.columns.name = None
wide = wide.reset_index()

# append to the second dataset
merged = df_dataset.merge(wide, on="epd", how="left", validate="m:1")

# sanity checks
engine_cols = [c for c in wide.columns if c != "epd"]
print("rows before/after:", len(df_engine), len(merged))
print("puzzles with no engine results:", merged[engine_cols].isna().all(axis=1).sum())
print(engine_cols)

merged.to_csv("../../datasets/dataset_test_engines.csv", index=False)