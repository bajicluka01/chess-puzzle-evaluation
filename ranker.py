import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRanker
from sklearn.metrics import ndcg_score as ndcg
from scipy.stats import kendalltau
from lightgbm import LGBMRanker
from catboost import CatBoostRanker, Pool

df = pd.read_csv("dataset_1k.csv")
remove_attributes = []
for c in df["themes"]:
    remove_attributes.extend(c.split(" "))
y = df["rating"]
df = df.drop(columns=["themes", "solution", "epd", "rating_dev", "rating"] + remove_attributes)
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = (y_train // 100).astype(int)
y_test = (y_test // 100).astype(int)

group_size = 10000
num_rows = len(X_train)
qid_train = np.repeat(np.arange(np.ceil(num_rows / group_size)), group_size)[:num_rows]

model = XGBRanker(
    objective='rank:pairwise',
    n_estimators=100,
    learning_rate=0.1
)

model.fit(X_train, y_train, qid=qid_train)
y_pred = model.predict(X_test)
print(f"NDCG: {ndcg([y_test], [y_pred], k=100)}")

max_bin = y_test.max()
y_test_inverted = max_bin - y_test
y_pred_inverted = -y_pred
print(f"NDCG: {ndcg([y_test_inverted], [y_pred_inverted], k=100)}")

corr, _ = kendalltau(y_test[:100], y_pred[:100])
print(f"Kendall's Tau: {corr:.4f}")

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
model2 = LGBMRanker()

#model2.fit(X_train, y_train, group=[X_train.shape[0]], eval_set=[(X_val, y_val)], eval_group=[X_val.shape[0]])
model2.fit(X_train, y_train, group=[X_train.shape[0]])
y_pred = model2.predict(X_test)
print(f"NDCG: {ndcg([y_test], [y_pred], k=100)}")

max_bin = y_test.max()
y_test_inverted = max_bin - y_test
y_pred_inverted = -y_pred
print(f"NDCG: {ndcg([y_test_inverted], [y_pred_inverted], k=100)}")

corr, _ = kendalltau(y_test[:100], y_pred[:100])
print(f"Kendall's Tau: {corr:.4f}")

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
model3 = CatBoostRanker(iterations=1000, learning_rate=0.1)

#model2.fit(X_train, y_train, group=[X_train.shape[0]], eval_set=[(X_val, y_val)], eval_group=[X_val.shape[0]])
"""num_rows = len(X)
df['group_id'] = np.repeat(np.arange(np.ceil(num_rows / group_size)), group_size)[:num_rows]
group_id = df['group_id']
X = df
X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(X, y, group_id, test_size=0.2, random_state=42)
#model3.fit(X_train, y_train, group_id=X_train.groupby(group_train).size().to_list())
train_group_sizes = X_train.groupby(group_train).size().to_list()
test_group_sizes = X_test.groupby(group_test).size().to_list()

model3.fit(
    X_train, y_train,
    group_id=[800],
    verbose=True
)"""

train_pool = Pool(data=X_train, label=y_train, group_id=qid_train.astype(int))
model3.fit(train_pool)
y_pred = model3.predict(X_test)
print(f"NDCG: {ndcg([y_test], [y_pred], k=100)}")

max_bin = y_test.max()
y_test_inverted = max_bin - y_test
y_pred_inverted = -y_pred
print(f"NDCG: {ndcg([y_test_inverted], [y_pred_inverted], k=100)}")

corr, _ = kendalltau(y_test[:100], y_pred[:100])
print(f"Kendall's Tau: {corr:.4f}")
