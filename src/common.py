import pandas as pd

ID_COL = "Experiment"  # design ID column
CD_COL = "Drag_Value"   # the target

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

def load_frames():
    """
    Single-point train/val/test split so that each model 
    gets the same split.
    """

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

    return train_df, val_df, test_df
