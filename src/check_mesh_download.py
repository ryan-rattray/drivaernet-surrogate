# src/check_mesh_download.py — inventory the mesh archives against the split.
# Opens each zip's table of contents only; extracts nothing.
# The verdict it prints: how many of the split's design IDs are present.

# --- Chunk 1 of 3 — the setup -------------------------------------------

from pathlib import Path
from zipfile import ZipFile
import pandas as pd

# Directory for the meshes
MESH_DIR = Path(r"D:\drivaernet-meshes")

# File containing the split information
SPLIT_FILE = Path("data/split.csv")
# File to store the collected meshes that match the split
RESULTS_FILE = Path("results/mesh_inventory.txt")

# --- Chunk 2 of 3 — reading the zips -------------------------------------
def normalize(design_id):
    """
    Removes the trailing number from the ID, so that 0 padding differences are accounted for 
    """
    family, _, number = design_id.rpartition("_")
    return f"{family}_{int(number)}"


def ids_in_zip(zip_path):
    """
    Returns the set of design IDs inside the STL filenames from one zip folder.
    """
    with ZipFile(zip_path) as archive:
        names = archive.namelist()
    return {
        normalize(Path(name).stem)
        for name in names
        if name.lower().endswith(".stl")
    }


split = pd.read_csv(SPLIT_FILE)
split_ids = {normalize(d) for d in split["design_id"]}
print(f"Split designs to account for: {len(split_ids)}")

zip_paths = sorted(MESH_DIR.glob("*.zip"))
assert zip_paths, f"No .zip files found in {MESH_DIR} - check MESH_DIR"

found_ids = set()
lines = []
for zip_path in zip_paths:
    ids = ids_in_zip(zip_path)
    overlap = ids & split_ids
    found_ids |= ids
    line = (
        f"{zip_path.name}: {len(ids)} STLs, "
        f"{len(overlap)} of them in the split"
    )
    print(line)
    lines.append(line)

# --- Chunk 3 of 3 — the verdict, and the record --------------------------

missing = split_ids - found_ids
present = split_ids & found_ids

summary = (
    f"\nArchives checked: {len(zip_paths)}\n"
    f"Total STLs in archives: {len(found_ids)}\n"
    f"Split designs present: {len(present)}\n"
    f"Split designs missing: {len(missing)}\n"
)
print(summary)
lines.append(summary)

if missing:
    lines.append("Missing design IDs (first 20):")
    lines.extend(sorted(missing)[:20])

RESULTS_FILE.parent.mkdir(exist_ok=True)
RESULTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {RESULTS_FILE}")
