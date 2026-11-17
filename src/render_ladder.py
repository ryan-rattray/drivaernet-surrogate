# Post visuals for Flight 2 — v2, matplotlib only.
# (v1 used pyvista; Windows Application Control on REDFIVE blocks
# VTK's DLLs, so the render is done with matplotlib instead: face
# centroids shaded by their surface normal — a flat-shaded car.)
#
# Renders F_D_WM_WW_0001 five ways from the same viewpoint:
# the full surface (shaded), then the sweep: 20k, 10k, 5k, 2.5k.
# Outputs land in results/. No new installs needed.

import io
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

MESH_DIR = Path(r"D:\drivaernet-meshes")  # <-- your drive letter here
ZIP_PATH = MESH_DIR / "F_D_WM_WW_1.zip"
DESIGN = "F_D_WM_WW_0001"
SEED = 42

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

with ZipFile(ZIP_PATH) as archive:
    entry = next(
        name
        for name in archive.namelist()
        if Path(name).stem == DESIGN and name.lower().endswith(".stl")
    )
    raw_bytes = archive.read(entry)

mesh = trimesh.load(io.BytesIO(raw_bytes), file_type="stl")
print(f"{DESIGN}: {len(mesh.faces):,} faces, {len(mesh.vertices):,} vertices")

EXTENTS = mesh.bounds[1] - mesh.bounds[0]
POINT_COLOR = "#5c85c2"  # single steel blue for all point clouds (no height gradient)


def save_view(ax_points, colors, point_size, path):
    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        ax_points[:, 0],
        ax_points[:, 1],
        ax_points[:, 2],
        c=colors,
        s=point_size,
        depthshade=False,
    )
    ax.set_box_aspect(tuple(EXTENTS))
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# --- 1. The full rendered car: face centroids, flat-shaded ----------------
# Each triangle's center point, colored by how directly its surface
# faces the light — bright where it faces the light, dark where it
# turns away. That shading is what makes it read as a solid car.
centers = np.asarray(mesh.triangles_center)
normals = np.asarray(mesh.face_normals)
light = np.array([0.5, -0.6, 0.8])
light = light / np.linalg.norm(light)
shade = 0.30 + 0.70 * np.clip(normals @ light, 0.0, 1.0)
base_color = np.array([0.36, 0.52, 0.76])  # steel blue
face_colors = base_color * shade[:, None]

rng = np.random.default_rng(SEED)
pick = rng.choice(len(centers), size=200_000, replace=False)
save_view(centers[pick], face_colors[pick], 2.0, OUT_DIR / "render_full_surface.png")

# --- 2-5. The sweep points, flat blue ------------------------------------
for count, size in ((20_000, 2.5), (10_000, 3.0), (5_000, 4.0), (2_500, 5.0)):
    np.random.seed(SEED)
    points = np.asarray(mesh.sample(count), dtype=np.float64)
    assert points.shape == (count, 3)
    save_view(points, POINT_COLOR, size, OUT_DIR / f"render_points_{count / 1000:g}k.png")

print("Done — five images in results/.")

# --- 6. The rowed grid: 5 versions x side / top / iso ---------------------
# One image for the post. Rows: full surface, then 20k down to 2.5k.
# Reuses the mesh, centroids, and shading computed above.
rows = [
    ("Full mesh", centers[pick], face_colors[pick], 1.5),
]
for count, size in ((20_000, 2.0), (10_000, 2.5), (5_000, 3.0), (2_500, 3.5)):
    np.random.seed(SEED)
    pts = np.asarray(mesh.sample(count), dtype=np.float64)
    rows.append((f"{count:,} points", pts, POINT_COLOR, size))

def front_layer(pts, colors, plane_cols, depth_col, depth_sign, bin_size=0.01):
    """Viewer-nearest point per small bin in the projection plane.

    A 2D scatter has no occlusion: every point draws, so in the surface
    row the far side and the dark underside speckle straight through
    the near side. Keeping only the nearest face center per bin is a
    poor-man's hidden-surface removal and cleans the view up.
    """
    plane = pts[:, plane_cols]
    keys_xy = np.floor((plane - plane.min(axis=0)) / bin_size).astype(np.int64)
    key = keys_xy[:, 0] * 100_000 + keys_xy[:, 1]
    depth = depth_sign * pts[:, depth_col]
    order = np.lexsort((depth, key))
    _, first = np.unique(key[order], return_index=True)
    keep = order[first]
    return pts[keep], colors[keep]


fig = plt.figure(figsize=(14, 20))
for r, (label, pts, colors, size) in enumerate(rows):
    # Side (x-z)
    ax1 = fig.add_subplot(5, 3, r * 3 + 1)
    if r == 0:
        s_pts, s_colors = front_layer(pts, colors, [0, 2], 1, +1)  # viewer at -y
        ax1.scatter(s_pts[:, 0], s_pts[:, 2], c=s_colors, s=size)
    else:
        ax1.scatter(pts[:, 0], pts[:, 2], c=colors, s=size)
    ax1.set_aspect("equal")
    ax1.set_ylabel(label, fontsize=13, fontweight="bold")
    if r == 0:
        ax1.set_title("Side")
    if r == len(rows) - 1:
        ax1.set_xlabel("x (length)")

    # Top (x-y)
    ax2 = fig.add_subplot(5, 3, r * 3 + 2)
    if r == 0:
        t_pts, t_colors = front_layer(pts, colors, [0, 1], 2, -1)  # viewer above
        ax2.scatter(t_pts[:, 0], t_pts[:, 1], c=t_colors, s=size)
    else:
        ax2.scatter(pts[:, 0], pts[:, 1], c=colors, s=size)
    ax2.set_aspect("equal")
    if r == 0:
        ax2.set_title("Top")
    if r == len(rows) - 1:
        ax2.set_xlabel("x (length)")

    # Iso (3D)
    ax3 = fig.add_subplot(5, 3, r * 3 + 3, projection="3d")
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=size, depthshade=False)
    ax3.set_box_aspect(tuple(EXTENTS))
    ax3.view_init(elev=20, azim=-60)
    if r == 0:
        ax3.set_title("Iso")

fig.suptitle(f"{DESIGN}", fontsize=15)
fig.tight_layout()
grid_path = OUT_DIR / "render_grid.png"
fig.savefig(grid_path, dpi=150)
plt.close(fig)
print(f"Saved {grid_path}")

# --- 7. Mirror zoom: the same crop at every resolution --------------------
# The mirror is the smallest aero feature on the car, so it is where
# thinning points show first. The script finds the mirror itself: it is
# the widest station on the car, so the x slice with the greatest width
# is the mirror station. The crop is built around it, +y side.
width_bins = np.arange(mesh.bounds[0][0], mesh.bounds[1][0], 0.05)
bin_idx = np.clip(
    np.searchsorted(width_bins, centers[:, 0]) - 1, 0, len(width_bins) - 2
)
widths = np.zeros(len(width_bins) - 1)
np.maximum.at(widths, bin_idx, np.abs(centers[:, 1]))
mirror_x = float(width_bins[np.argmax(widths)] + 0.025)
print(f"Mirror station: x = {mirror_x:.2f} m (widest slice of the car)")

BOX = (
    (mirror_x - 0.20, mirror_x + 0.35),  # x: the mirror and its stalk
    (0.90, 1.01),  # y: only what protrudes past the door skin
    (0.90, 1.22),  # z: the mirror's height band
)


def in_box(pts):
    return (
        (pts[:, 0] >= BOX[0][0])
        & (pts[:, 0] <= BOX[0][1])
        & (pts[:, 1] >= BOX[1][0])
        & (pts[:, 1] <= BOX[1][1])
        & (pts[:, 2] >= BOX[2][0])
        & (pts[:, 2] <= BOX[2][1])
    )


zoom_sets = [("Full mesh", centers, face_colors, 3.0)]
for count, size in ((20_000, 4.0), (10_000, 5.0), (5_000, 6.0), (2_500, 7.0)):
    np.random.seed(SEED)
    pts = np.asarray(mesh.sample(count), dtype=np.float64)
    zoom_sets.append((f"{count:,} points", pts, POINT_COLOR, size))

fig = plt.figure(figsize=(16, 4.2))
for i, (label, pts, colors, size) in enumerate(zoom_sets):
    mask = in_box(pts)
    sub = pts[mask]
    sub_colors = colors[mask] if isinstance(colors, np.ndarray) else colors
    ax = fig.add_subplot(1, 5, i + 1, projection="3d")
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=sub_colors, s=size, depthshade=False)
    ax.set_xlim(*BOX[0])
    ax.set_ylim(*BOX[1])
    ax.set_zlim(*BOX[2])
    ax.set_box_aspect((BOX[0][1] - BOX[0][0], BOX[1][1] - BOX[1][0], BOX[2][1] - BOX[2][0]))
    ax.view_init(elev=15, azim=-50)
    ax.set_axis_off()
    ax.set_title(f"{label}\n{sub.shape[0]:,} pts in crop", fontsize=11)
    print(f"{label}: {sub.shape[0]:,} points in the mirror crop")

fig.suptitle(f"{DESIGN} — side mirror, same crop at every resolution", fontsize=13)
fig.tight_layout()
zoom_path = OUT_DIR / "render_mirror_zoom.png"
fig.savefig(zoom_path, dpi=150)
plt.close(fig)
print(f"Saved {zoom_path}")







