import sdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def process_data(file, class_split="equidistant", remove=False):
    df = pd.read_csv(file)
    ratings = pd.to_numeric(df["rating"], errors="coerce").dropna()
    remove_attributes = []
    # for c in df["themes"]:
    #    remove_attributes.extend(c.split(" "))

    global cutoff1
    global cutoff2

    if class_split == "equidistant":
        classwidth = (ratings.max() - ratings.min()) / 3
        cutoff1 = ratings.min() + classwidth
        cutoff2 = ratings.min() + 2 * classwidth
        df["difficulty"] = df["rating"].map(elo_to_difficulty)
    elif class_split == "balanced":
        rating_list = ratings.to_list()
        rating_list.sort()
        cutoff1 = rating_list[int(len(rating_list) / 3)]
        cutoff2 = rating_list[int(2 * len(rating_list) / 3)]
        df["difficulty"] = df["rating"].map(elo_to_difficulty)
    else:
        raise ("evaluation.py: unknown class_split parameter in process_data.")

    if remove:
        X = df.drop(columns=["solution", "epd", "rating_dev"] + remove_attributes)
    else:
        X = df.drop(columns=["solution", "epd", "rating_dev"])
    return X


# easy/medium/hard
def elo_to_difficulty(y):
    if y < cutoff1:
        return "0"
    elif y < cutoff2:
        return "1"
    else:
        return "2"

#df = pd.read_csv("../../datasets/dataset_test.csv")
df = process_data("../../datasets/dataset_test.csv")
#drop = ["epd", "solution", "rating_dev", "move1multiPV", "move2multiPV", "move3multiPV", "black_k", "white_K"]
drop = ["rating", "move1multiPV", "move2multiPV", "move3multiPV", "black_k", "white_K"]
df = df.drop(drop, axis=1)
#target = "rating"
target = "difficulty"
device = "cuda"

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
stypes = sdm.infer_stypes(df, overrides={target: "categorical"})
#stypes = sdm.infer_stypes(df)
print(stypes)  # sanity check

train_t = sdm.TableTensor.from_pandas(train_df, stypes=stypes, device=device)
test_t = sdm.TableTensor.from_pandas(test_df,  stypes=stypes, device=device)

#model = sdm.models.KumoTabular(task="classification", device=device)
model = sdm.models.TabFM(task="classification", device=device)
random_state = 67
import torch
generator = torch.Generator(device=device).manual_seed(random_state)
pred = model(
    x_context=train_t.drop_columns(target),
    y_context=train_t[:, target],
    x_query=test_t.drop_columns(target),
    generator=generator
)


pred_df = pred.to_pandas()
#y_pred_pos = pred_df.to_numpy().argmax(axis=1)
y_pred = pred_df.idxmax(axis=1).to_numpy()
print(pred_df)
print(y_pred)
exit(0)
y_true = test_df[target].to_numpy()
y_med = pred_df["q500"].to_numpy()

print("MAE :", mean_absolute_error(y_true, y_med))
print("RMSE:", root_mean_squared_error(y_true, y_med))
print("R2  :", r2_score(y_true, y_med))

