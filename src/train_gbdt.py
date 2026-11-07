# src/train_gbdt.py — the gradient-boosting baseline (LightGBM).
# Loads Stair 1's frozen split, trains once, scores the test rows once,
# and writes the score to results/gbdt_results.txt.
# Run from the repo root, with the project venv active.

import pandas as pd
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor, early_stopping

ID_COL = "Experiment"  # design ID column
CD_COL = "Drag_Value"   # the target

# --- Load the table and the split ---------------------------------------------
designs = pd.read_csv("data/designs.csv")
split = pd.read_csv("data/split.csv")

# Prove the two files name the same designs, each exactly once:
# no design in more than one split, and every design labeled exactly once.
assert not split["design_id"].duplicated().any(), (
    "a design appears in the split file more than once"
)
assert set(split["design_id"]) == set(designs[ID_COL]), (
    "split and table name different designs"
)

merged = designs.merge(split, left_on=ID_COL, right_on="design_id", how="inner")
assert len(merged) == len(designs), "the join changed the row count"

print(f"Total designs: {len(merged)}")
print("Rows per split:")
print(merged["split"].value_counts())

# --- The feature list: the 23 design parameters, named explicitly -------------
FEATURES = [
    "B_Ramp_Angle",
    "B_Diffusor_Angle",
    "B_Trunklid_Angle",
    "C_Side_Mirrors_Rotation",
    "D_Rear_Window_Inclination",
    "D_Winscreen_Inclination",
    "C_Side_Mirrors_Translate_X",
    "C_Side_Mirrors_Translate_Z",
    "D_Winscreen_Length",
    "D_Rear_Window_Length",
    "E_A_B_C_Pillar_Thickness",
    "G_Trunklid_Curvature",
    "G_Trunklid_Length",
    "H_Front_Bumper_Curvature",
    "H_Front_Bumper_Length",
    "F_Door_Handles_Thickness",
    "F_Door_Handles_Z_Position",
    "E_Fenders_Arch_Offset",
    "A_Car_Length",
    "F_Door_Handles_X_Position",
    "A_Car_Width",
    "A_Car_Roof_Height",
    "A_Car_Green_House_Angle",
]

NOT_FEATURES = [
    "Average Cd", "Std Cd",
    "Average Cl", "Std Cl",
    "Average Cl_f", "Std Cl_f",
    "Average Cl_r", "Std Cl_r",
]

assert len(FEATURES) == 23, "the feature list should name 23 parameters"
assert all(c in designs.columns for c in FEATURES), (
    "a named feature is missing from the table"
)
assert not any(c in FEATURES for c in NOT_FEATURES), (
    "a coefficient column slipped into the features"
)
assert CD_COL not in FEATURES and ID_COL not in FEATURES
print(f"Features ({len(FEATURES)}): {FEATURES}")

# --- The three frames -----------------------------------------------------------
train_df = merged[merged["split"] == "train"]
val_df = merged[merged["split"] == "val"]
test_df = merged[merged["split"] == "test"]
print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

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