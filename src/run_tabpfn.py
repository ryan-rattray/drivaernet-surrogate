# src/run_tabpfn.py — TabPFN in context: no training; the train split IS the input.

import numpy as np
from common import load_frames, FEATURES, CD_COL

RUN_ROUTE = "kaggle gpu"  # or "local gpu"

# loading splits from load_frames()
train_df, val_df, test_df = load_frames()
print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

# No preprocessing.  TabPFN does its own preprocessing
X_train = train_df[FEATURES].to_numpy
y_train = train_df[CD_COL].to_numpy
X_val = val_df[FEATURES].to_numpy
y_val = val_df[CD_COL].to_numpy
X_test = test_df[FEATURES].to_numpy
y_test = test_df[CD_COL].to_numpy

# --- The model: defaults only, because there is nothing to tune ---------------------
from tabpfn import TabPFNRegressor


def r2_from_definition(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot

model = TabPFNRegressor()

model.fit(X_train, y_train)  # No training.  Training set is shown to model as the context.

# Make predictions on val set and calculate R²
val_pred = model.predict(X_val)
val_r2 = r2_from_definition(y_val, val_pred)
print(f"Validation R²: {val_r2}")

# The test rows, predicted exactly once.
test_pred = model.predict(X_test)
test_r2 = r2_from_definition(y_test, test_pred)
print(f"Test R²: {test_r2}")

lines = [
    "contender: TabPFN (in context, no training)",
    f"run_route: {RUN_ROUTE}",
    f"train_rows: {len(train_df)}",
    f"val_rows: {len(val_df)}",
    f"test_rows: {len(test_df)}",
    f"val_r2: {val_r2}",
    f"test_r2: {test_r2}",
]
with open("results/tabpfn_results.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print("Wrote results/tabpfn_results.txt")




