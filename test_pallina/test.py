""" import open3d as o3d
import numpy as np
import sys

# === Parametri ===
frame = sys.argv[1] if len(sys.argv) > 1 else "0005"
dir_path = "ball_pointclouds"
full_path = f"{dir_path}/pointcloud_{frame}.ply"
ball_path = f"{dir_path}/ball_frame_{frame}.ply"

# === Carica le point cloud ===
pcd_full = o3d.io.read_point_cloud(full_path)
pcd_ball = o3d.io.read_point_cloud(ball_path)

# Colora la pallina di rosso
pcd_ball.paint_uniform_color([1.0, 0.0, 0.0])

# Visualizza entrambe
o3d.visualization.draw_geometries(
    [pcd_full, pcd_ball],
    window_name=f"Frame {frame} – Scena + Pallina",
    width=1280,
    height=720,
    point_show_normal=False
)
 """

import open3d as o3d
import numpy as np
import sys
from sklearn.cluster import DBSCAN

# === Parametri ===
frame = sys.argv[1] if len(sys.argv) > 1 else "0005"
radius_max = 0.08       # m dal centro
z_offset = 0.015        # m sotto il centro (per tagliare tavolo)
eps = 0.0085            # DBSCAN
min_samples = 20

# === Carica point cloud ===
pcd_full = o3d.io.read_point_cloud(f"ball_pointclouds/pointcloud_{frame}.ply")
pcd_ball = o3d.io.read_point_cloud(f"ball_pointclouds/ball_frame_{frame}.ply")

points = np.asarray(pcd_ball.points)
if len(points) < 10:
    print("[WARN] Point cloud troppo piccola.")
    exit()

# === Filtro radiale dal centro ===
center = points.mean(axis=0)
dists = np.linalg.norm(points - center, axis=1)
points = points[dists < radius_max]

# === DBSCAN clustering ===
if len(points) > 10:
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    if np.any(labels != -1):
        majority = np.argmax(np.bincount(labels[labels != -1]))
        points = points[labels == majority]

""" # === Filtro Z per togliere il tavolo ===
z_median = np.median(points[:, 2])
points = points[points[:, 2] > z_median - z_offset] """

# === Crea nuova point cloud pulita ===
pcd_clean = o3d.geometry.PointCloud()
pcd_clean.points = o3d.utility.Vector3dVector(points)
pcd_clean.paint_uniform_color([1, 0, 0])  # rosso

# === Visualizza sopra alla scena ===
o3d.visualization.draw_geometries([pcd_full, pcd_clean], window_name=f"Frame {frame} – Pallina pulita")
