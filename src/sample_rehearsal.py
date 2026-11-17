# THROWAWAY rehearsal for src/sample_one.py — NOT Step 4.
# Purpose: prove the mechanism (zip -> STL -> 5,000 points -> picture)
# on the F_D archive already on disk, while the S archives transfer.
# No split logic: no F_D design is in the split, and nothing here is
# Step 4's official car. Outputs go to results/ only; data/points/
# stays clean for the real run. Delete this file after the real run.

import io
from pathlib import Path
from zipfile import ZipFile

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MESH_DIR = Path(r"D:\drivaernet-meshes")  # <-- your drive letter here
ZIP_PATH = MESH_DIR / "F_D_WM_WW_1.zip"

POINTS_PER_CAR = 5000
SEED = 42

with ZipFile(ZIP_PATH) as archive:
    stl_names = sorted(
        name for name in archive.namelist() if name.lower().endswith(".stl")
    )
    chosen_name = stl_names[0]
    raw_bytes = archive.read(chosen_name)

print(f"Rehearsal design (not in split): {Path(chosen_name).stem}")

mesh = trimesh.load(io.BytesIO(raw_bytes), file_type="stl")
print(f"Faces: {len(mesh.faces):,}")
print(f"Vertices: {len(mesh.vertices):,}")
print(f"Bounds (min xyz): {mesh.bounds[0]}")
print(f"Bounds (max xyz): {mesh.bounds[1]}")
# Bounds are the YMC on units: a longest axis of a few units = metres;
# a few thousand = millimetres. Record which one it is.

np.random.seed(SEED)
points = mesh.sample(POINTS_PER_CAR)
points = np.asarray(points, dtype=np.float64)

assert points.shape == (POINTS_PER_CAR, 3), (
    f"Expected {(POINTS_PER_CAR, 3)}, got {points.shape}"
)
assert np.isfinite(points).all(), "Non-finite value in the sampled points."
print(f"Sampled points: {points.shape}")

fig = plt.figure(figsize=(14, 4))

# Side profile — the silhouette is the view that reads as a car.
ax1 = fig.add_subplot(131)
ax1.scatter(points[:, 0], points[:, 2], s=1)
ax1.set_aspect("equal")
ax1.set_xlabel("x (length)")
ax1.set_ylabel("z (height)")
ax1.set_title("Side")

# Top view.
ax2 = fig.add_subplot(132)
ax2.scatter(points[:, 0], points[:, 1], s=1)
ax2.set_aspect("equal")
ax2.set_xlabel("x (length)")
ax2.set_ylabel("y (width)")
ax2.set_title("Top")

# 3D view with the car's true proportions (no axis stretching).
ax3 = fig.add_subplot(133, projection="3d")
ax3.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap="viridis")
extents = mesh.bounds[1] - mesh.bounds[0]
ax3.set_box_aspect(tuple(extents))
ax3.view_init(elev=20, azim=-60)
ax3.set_title("3D")

fig.suptitle(f"{Path(chosen_name).stem} — rehearsal, all 5,000 points")
fig.tight_layout()
picture = Path("results/rehearsal_first_car_points.png")
picture.parent.mkdir(exist_ok=True)
fig.savefig(picture, dpi=150)
print(f"Saved {picture} — open it and check it looks like a car.")

