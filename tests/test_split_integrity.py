# tests/test_split_integrity.py — pytest checks that the repo's own record is coherent.
#
# Split checks: the saved split must be the table this project published —
# 4,165 designs, 3,051 / 555 / 559 designs in train / val / test, no design in
# two splits. Scores checks: each results file carries exactly one val_r2: line
# in the sane range; the trained contenders' files carry NO test_r2: line (the
# test deferral), and TabPFN's carries exactly one, also in range.
#
# Needs data/designs.csv — a gitignored working file — so this suite runs on
# the working repo, not from a fresh clone. Run from the repo root:
#     python -m pytest

import pandas as pd

SPLIT = "data/split.csv"
DESIGNS = "data/designs.csv"

GBDT_RESULTS = "results/gbdt_results.txt"
MLP_RESULTS = "results/mlp_results.txt"
TABPFN_RESULTS = "results/tabpfn_results.txt"
RESULT_FILES = [GBDT_RESULTS, MLP_RESULTS, TABPFN_RESULTS]


def load_split():
    """Load the saved split file (one row per design, one label per row)."""
    return pd.read_csv(SPLIT, dtype=str)


def test_split_matches_project_table():
    split = load_split()
    counts = split["split"].value_counts().to_dict()
    assert counts == {"train": 3051, "val": 555, "test": 559}, (
        f"split counts {counts} do not match the project table "
        "3051 / 555 / 559 — the split file changed"
    )
    assert len(split) == 4165


def test_train_and_test_are_disjoint():
    # Loose rows are already excluded mechanically (one label per design in the
    # saved file), so the real question is whether an ID repeats anywhere.
    split = load_split()
    assert split["design_id"].is_unique, (
        "a design ID appears more than once in the split — "
        "one design could be sitting in two splits"
    )


def test_results_files_are_well_formed():
    # Every results file carries exactly one val_r2: line (SOI Amendment 3:
    # selection rides on validation), parsing as a number in the sane range.
    # The test_r2: line follows the deferral (Ryan, 2026-11-08): the TRAINED
    # contenders' files must carry NONE — their scripts never score test, so a
    # test line there means the deferral was breached — while TabPFN's file
    # (frozen model, nothing to tune) carries exactly one, also in range.
    for path in RESULT_FILES:
        text = open(path).read()  # a missing file fails here, as it should
        values = [
            ln.split(":", 1)[1].strip()
            for ln in text.splitlines()
            if ln.startswith("val_r2:")
        ]
        assert len(values) == 1, f"{path}: expected exactly one val_r2: line"
        r2 = float(values[0])  # must parse as a number
        assert -1 <= r2 <= 1, (
            f"{path}: val_r2: {r2} is outside the sane range for this problem"
        )
        test_values = [
            ln.split(":", 1)[1].strip()
            for ln in text.splitlines()
            if ln.startswith("test_r2:")
        ]
        if path == TABPFN_RESULTS:
            assert len(test_values) == 1, (
                f"{path}: TabPFN's file carries exactly one test_r2: line"
            )
            r2 = float(test_values[0])
            assert -1 <= r2 <= 1, (
                f"{path}: test_r2: {r2} is outside the sane range for this problem"
            )
        else:
            assert len(test_values) == 0, (
                f"{path}: trained contender carries a test_r2: line — "
                "the test deferral was breached"
            )
