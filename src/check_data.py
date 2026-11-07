# src/check_data.py — load the two downloaded files, prove they line up
# design-for-design, and save the merged table every later stair uses.
# Run from the repo root, with the project venv active.

PARAMS_FILE = "data/raw/DrivAerNet_ParametricData.csv"
DRAG_FILE = "data/raw/DrivAerNetPlusPlus_Cd_8k_Updated.csv"
ID_COL_PARAMS = "Experiment"
ID_COL_DRAG = "ID"
CD_COL = "Drag_Value"

import pandas as pd

# The facts this script establishes. Every fact line is printed to the screen
# AND collected in `lines`, so results/data_check.txt (written at the end)
# holds exactly what the run printed — nobody re-types a number.

lines =[]


def report(text=""):
    print(text)
    lines.append(text)

# --- Load both files ---------------------------------------------------------
params = pd.read_csv(PARAMS_FILE)
drag = pd.read_csv(DRAG_FILE)

# Facts about each file as downloaded: how many rows, and what the columns are.
report(f"Parameters file: {PARAMS_FILE}")
report(f"  rows:  {len(params)}")
report(f"  columns:  {list(params.columns)}")
report(f"Drag file:  {DRAG_FILE}")
report(f"  rows:  {len(drag)}")
report(f"  columns:  {list(drag.columns)}")
report()

# --- Check the overlap BEFORE merging ----------------------------------------
# Sets of IDs: set arithmetic answers "in both / only in one" exactly,
# without merging anything yet.
params_ids = set(params[ID_COL_PARAMS])
drag_ids = set(drag[ID_COL_DRAG])
in_both = params_ids & drag_ids
only_params = params_ids - drag_ids
only_drag = drag_ids - params_ids

report(f"Design IDs in both files:  {len(in_both)}")
report(f"Design IDs only in parameters file:  {len(only_params)}")
report(f"Design IDs only in the drag file:  {len(only_drag)}")
report()

# --- Merge on the design ID ----------------------------------------------------
# Keep every parameters column, plus the Cd column from the drag file.
# The key has a different name in each file, so pandas is told both names.
merged = params.merge(
    drag[[ID_COL_DRAG, CD_COL]],
    left_on=ID_COL_PARAMS,
    right_on=ID_COL_DRAG,
    how="inner",
)

# If a file ever listed a design twice, the merge would silently multiply
# rows. The merged table must have exactly one row per shared design.
assert len(merged) ==len(in_both), (
    f"merged rows ({len(merged)}) != desings in both files ({len(in_both)}): "
    "a duplicated design ID in one of the files is the likely cause"
)

# From here on the merged table speaks with the parameters file's ID spelling;
# drop the drag file's duplicate key column.
if ID_COL_DRAG != ID_COL_PARAMS:
    merged = merged.drop(columns=[ID_COL_DRAG])

# --- The facts of the merged table ----------------------------------------------
params_cols = [c for c in merged.columns if c not in (ID_COL_PARAMS, CD_COL)]
cd = merged[CD_COL]

report(f"Merged designs:  {len(merged)}")
report(f"Parameter columns:  {len(params_cols)}")
report(f"Cd min:  {cd.min()}")
report(f"Cd max:  {cd.max()}")
report(f"Cd mean:  {cd.mean()}")
report(f"Missing Cd values:  {int(cd.isna().sum())}")

# --- Save both outputs --------------------------------------------------------------

merged.to_csv("data/designs.csv", index=False)
with open("results/data_check.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print("Saved data/designs.csv and results/data_check.txt")