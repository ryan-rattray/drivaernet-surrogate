# src/compare_results.py — build results/comparison.md from the three results

import sys

# (results file, display name, what it is) — the table's rows, in order.
CONTENDERS = [
    ("results/gbdt_results.txt", "Gradient boosting", "the conventional tool, trained"),
    ("results/mlp_results.txt", "PyTorch MLP", "hand-built network, trained"),
    ("results/tabpfn_results.txt", "TabPFN", "foundation model, context only, no training"),
]


def read_score(path, key, required=True):
    """
    Return the raw text after '<key>:' in a results file (None if absent
    and not required).
    """
    try:
        text = open(path).read()
    except FileNotFoundError:
        sys.exit(f"Missing results file: {path} — run that contender's stair first.")
    for line in text.splitlines():
        if line.startswith(key):
            return line.split(":", 1)[1].strip()
    if required:
        sys.exit(f"No {key} line in {path} — the file is malformed.")
    return None


rows = []
for path, name, what in CONTENDERS:
    train_raw = read_score(path, "train_r2:", required=False)
    val_raw = read_score(path, "val_r2:")
    test_raw = read_score(path, "test_r2:", required=False)
    if train_raw is None:
        train_cell = "— (no training)"  # TabPFN never trains, so it has no train score
    else:
        float(train_raw)
        train_cell = train_raw
    float(val_raw)  # the val value must parse; the table carries the file's own digits
    if test_raw is None:
        test_cell = "deferred"  # trained contender: test read waits for the decision comparison
    else:
        float(test_raw)
        test_cell = test_raw
    rows.append(f"| {name} | {what} | {train_cell} | {val_raw} | {test_cell} |")

table = "\n".join(
    [
        "# Flight 1 — comparison (one shared split; val = selection basis; trained test reads deferred)",
        "",
        "| Contender | What it is | Train R² | Val R² | Test R² |",
        "|---|---|---|---|---|",
        *rows,
        "",
    ]
)

with open("results/comparison.md", "w", encoding="utf-8") as f:
    f.write(table)

print(table)
print("Now the human checkpoint: open each results file and confirm each")
print("table entry against its source, digit for digit.")
