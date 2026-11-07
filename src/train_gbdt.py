# src/train_gbdt.py — the gradient-boosting baseline (LightGBM).
# Loads Stair 1's frozen split, trains once, scores the test rows once,
# and writes the score to results/gbdt_results.txt.
# Run from the repo root, with the project venv active.

from common import load_frames, FEATURES, CD_COL
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor, early_stopping

# --- Bring in the train/val/test df ---------------------------------------------
train_df, val_df, test_df = load_frames()

# --- Train ----------------------------------------------------------------------
LIBRARY = "LightGBM LGBMRegressor"

# Hypers
LEARNING_RATE = 0.05
N_TREES_CEILING = 500
EARLY_STOPPING_ROUNDS = 50
SEED = 42

# Define training and test
X_train = train_df[FEATURES]
y_train = train_df[CD_COL]
X_test = test_df[FEATURES]
y_test = test_df[CD_COL]

# Initiate model
model = LGBMRegressor(
    n_estimators=N_TREES_CEILING,
    learning_rate=LEARNING_RATE,
    random_state=SEED,
    verbose=-1,
)

# Train the model
model.fit(
    X_train,
    y_train,
    eval_X=val_df[FEATURES],
    eval_y=val_df[CD_COL],
    callbacks=[early_stopping(EARLY_STOPPING_ROUNDS)],
)
print(f"Trees actually used: {model.best_iteration_} of {N_TREES_CEILING} allowed")

CONFIG = (
    f"learning_rate={LEARNING_RATE}, n_estimators<={N_TREES_CEILING}, "
    f"num_leaves=default, early_stopping=validation split "
    f"({EARLY_STOPPING_ROUNDS} rounds), seed={SEED}"
)

# --- Train prediction and scoring ------------------------------------------
train_r2 = r2_score(y_train, model.predict(X_train))

# --- val prediction and scoring ------------------------------------------
val_r2 = r2_score(val_df[CD_COL], model.predict(val_df[FEATURES]))

# --- Test prediction and scoring ------------------------------------------
test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, test_pred)



print(f"Train R²: {train_r2}")
print(f"Val R²: {val_r2}")
print(f"Test R²:  {test_r2}")

# --- Write the results file -------------------------------------------------------
lines = [
    f"contender: gradient boosting ({LIBRARY})",
    f"library: {LIBRARY}",
    f"train_rows: {len(train_df)}",
    f"test_rows: {len(test_df)}",
    f"train_r2: {train_r2}",
    f"val_r2: {val_r2}",
    f"test_r2: {test_r2}",
    f"config: {CONFIG}",
]
with open("results/gbdt_results.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print("Wrote results/gbdt_results.txt")