# src/sample_one.py — turn one split design's STL into a 5,000-point cloud
# and a picture of it. The mechanism proof for the geometric lane.

import io
from pathlib import Path
from zipfile import ZipFile

import matplotlib
import numpy as np
import pandas as pd
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same folder as check_mesh_download.py — keep the two in agreement.
MESH_DIR = Path(r"D:\drivaernet-meshes")  # <-- your drive letter here
FIRST_ZIP = MESH_DIR / "N_S_WWC_WM.zip"  # first S archive; any landed S archive works — match the name

POINTS_PER_CAR = 5000
SEED = 42


def normalize(design_id):
    """Same normalization as check_mesh_download.py: family + int(number)."""
    family, _, number = design_id.rpartition("_")
    return f"{family}_{int(number)}"


split = pd.read_csv("data/split.csv")
split["canonical"] = [normalize(d) for d in split["design_id"]]
split_ids = set(split["canonical"])

with ZipFile(FIRST_ZIP) as archive:
    stl_names = [
        name for name in archive.namelist() if name.lower().endswith(".stl")
    ]
    candidates = sorted(
        name
        for name in stl_names
        if normalize(Path(name).stem) in split_ids
    )
    assert candidates, "No STL in the first archive belongs to the split."
    chosen_name = candidates[0]
    raw_bytes = archive.read(chosen_name)

canonical = normalize(Path(chosen_name).stem)
row = split.loc[split["canonical"] == canonical].iloc[0]
design_id = row["design_id"]  # the split's own ID string — names the outputs
split_row = row["split"]
print(f"Design: {design_id} (split: {split_row})")
mesh = trimesh.load(io.BytesIO(raw_bytes), file_type="stl")
print(f"Faces: {len(mesh.faces):,}")
print(f"Vertices: {len(mesh.vertices):,}")
print(f"Bounds (min xyz): {mesh.bounds[0]}")
print(f"Bounds (max xyz): {mesh.bounds[1]}")
# Bounds are the YMC on units: a longest axis of a few units = metres;
# a few thousand = millimetres. Either is fine. Record which one it is;
# do not rescale anything today.

np.random.seed(SEED)
points = mesh.sample(POINTS_PER_CAR)
points = np.asarray(points, dtype=np.float64)

assert points.shape == (POINTS_PER_CAR, 3), (
    f"Expected {(POINTS_PER_CAR, 3)}, got {points.shape}"
)
assert np.isfinite(points).all(), "Non-finite value in the sampled points."

out_file = Path("data/points") / f"{design_id}.npy"
out_file.parent.mkdir(parents=True, exist_ok=True)
np.save(out_file, points)
print(f"Saved {out_file} with shape {points.shape}")
fig = plt.figure(figsize=(14, 4))

ax1 = fig.add_subplot(131)
ax1.scatter(points[:, 0], points[:, 2], s=1)
ax1.set_aspect("equal")
ax1.set_xlabel("x (length)")
ax1.set_ylabel("z (height)")
ax1.set_title("Side")

ax2 = fig.add_subplot(132)
ax2.scatter(points[:, 0], points[:, 1], s=1)
ax2.set_aspect("equal")
ax2.set_xlabel("x (length)")
ax2.set_ylabel("y (width)")
ax2.set_title("Top")

ax3 = fig.add_subplot(133, projection="3d")
ax3.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color="#5c85c2")
extents = mesh.bounds[1] - mesh.bounds[0]
ax3.set_box_aspect(tuple(extents))
ax3.view_init(elev=20, azim=-60)
ax3.set_title("3D")

fig.suptitle(f"{design_id} — 5,000 sampled points")
fig.tight_layout()
picture = Path("results/first_car_points.png")
picture.parent.mkdir(exist_ok=True)
fig.savefig(picture, dpi=150)
print(f"Saved {picture} — open it and check it looks like a car.")
