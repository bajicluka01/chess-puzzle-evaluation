import sdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from KumoTabular import KumoClassifier, KumoRegressor
from TabFM import TabFMClassifier, TabFMRegressor

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
        return 0
    elif y < cutoff2:
        return 1
    else:
        return 2

df = pd.read_csv("../../datasets/dataset_test.csv")
#df = process_data("../../datasets/dataset_test.csv")
drop = ["epd", "solution", "rating_dev", "move1multiPV", "move2multiPV", "move3multiPV", "black_k", "white_K"]
#drop = ["rating", "move1multiPV", "move2multiPV", "move3multiPV", "black_k", "white_K"]
#df = df.drop(drop, axis=1)
#target = "rating"
target = "rating"
device = "cuda"

X = df.drop(columns=drop + [target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = TabFMRegressor(device=device, random_state=67)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(pred)

print("MAE :", mean_absolute_error(y_test, pred))
print("RMSE:", root_mean_squared_error(y_test, pred))
print("R2  :", r2_score(y_test, pred))