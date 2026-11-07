# Build data/split.csv from the authors' published split files.
# Run from the repo root, with the project venv active.

import pandas as pd

# Constants
SPLITS_DIR = "reference/DrivAerNet/train_val_test_splits"
FILES = {
    "train": "train_design_ids.txt",
    "val": "val_design_ids.txt",
    "test": "test_design_ids.txt",
}

# The design ID column in our merged table. The parameters file called it
# "Experiment", and check_data.py kept that name in data/designs.csv.
TABLE_ID_COL = "Experiment"


def read_split_ids(filename):
    df = pd.read_csv(f"{SPLITS_DIR}/{filename}", header=None)
    return df.iloc[:, 0]


# --- Read the three files and label each ID with its split -------------------
parts = []
for split_name, filename in FILES.items():
    ids = read_split_ids(filename)
    parts.append(pd.DataFrame({"design_id": ids, "split": split_name}))

split = pd.concat(parts, ignore_index=True)
print(f"IDs in the authors' files:  {len(split)}")

# No design may appear twice — in two splits, or twice inside one file.
assert not split["design_id"].duplicated().any(), (
    "a design ID appears more than once in the authors' split files"
)

# --- Keep only the designs that are in our table ------------------------------
# The authors split all of DrivAerNet++. Our table holds only the parametric
# designs. Adopting their split means their assignments, restricted to ours.
designs = pd.read_csv("data/designs.csv")
table_ids = set(designs[TABLE_ID_COL])

not_ours = split[~split["design_id"].isin(table_ids)]
print(f"Authors' IDs in our table (excluded):  {len(not_ours)}")

split = split[split["design_id"].isin(table_ids)]

# After that restriction, the split must cover our table exactly:
# every design assigned, none left over, in either direction.
split_ids = set(split["design_id"])
assert split_ids == table_ids, (
    f"split/table mismatch: {len(split_ids - table_ids)} IDs in the split but "
    f"not the table, {len(table_ids - split_ids)} in the table but not the split"
)

print("Rows per split:")
print(split["split"].value_counts())

split.to_csv("data/split.csv", index=False)
print(f"Saved data/split.csv - {len(split)} designs, matching the table exactly")