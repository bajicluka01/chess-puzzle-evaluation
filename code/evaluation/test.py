import sdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

df = pd.read_csv(r"E:\FRI\Magisterij\2. letnik\Umetna inteligenca\projekt\datasets\dataset_test.csv")
target = "rating"

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

stypes = sdm.infer_stypes(df, overrides={target: "numerical"})
print(stypes)  # sanity check

train_t = sdm.TableTensor.from_pandas(train_df, stypes=stypes, device="cpu")
test_t  = sdm.TableTensor.from_pandas(test_df,  stypes=stypes, device="cpu")

model = sdm.models.KumoTabular(task="regression", device="cpu")
pred = model(
    x_context=train_t.drop_columns(target),
    y_context=train_t[:, target],
    x_query=test_t.drop_columns(target),
)

pred_df = pred.to_pandas()
y_true = test_df[target].to_numpy()
y_med = pred_df["q500"].to_numpy()

print("MAE :", mean_absolute_error(y_true, y_med))
print("RMSE:", root_mean_squared_error(y_true, y_med))
print("R2  :", r2_score(y_true, y_med))

