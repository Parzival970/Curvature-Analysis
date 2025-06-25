import open3d as o3d
import numpy as np
import sys
from sklearn.cluster import DBSCAN

# === Parametri di filtro ===
FILTER_RADIUS = 0.05         # raggio massimo dal centro (m)
CLUSTER_EPS = 0.0085         # eps per DBSCAN (m)
CLUSTER_MIN_SAMPLES = 20     # punti minimi in un cluster
Z_FILTER_OFFSET = 0.015      # sotto il centro (m)

def fit_sphere_least_squares(points):
    A = np.hstack((2 * points, np.ones((points.shape[0], 1))))
    f = np.sum(points ** 2, axis=1).reshape(-1, 1)
    C, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
    center = C[:3].flatten()
    radius = np.sqrt(C[3] + np.sum(center**2))
    dists = np.linalg.norm(points - center, axis=1)
    rmse = np.sqrt(np.mean((dists - radius)**2))
    return center, radius, rmse

# === Input ===
frame = sys.argv[1] if len(sys.argv) > 1 else "0005"
pcd_path = f"ball_pointclouds/ball_frame_{frame}.ply"

# === Caricamento ===
pcd = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(pcd.points)
print(f"[INFO] Punti caricati: {len(points)} da {pcd_path}")

if len(points) < 10:
    print("[WARN] Point cloud troppo piccola.")
    exit()

# === Filtro radiale ===
center = points.mean(axis=0)
dists = np.linalg.norm(points - center, axis=1)
points = points[dists < FILTER_RADIUS]
print(f"[INFO] Dopo filtro radiale: {len(points)} punti")

# === DBSCAN ===
if len(points) > 10:
    labels = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES).fit(points).labels_
    if np.any(labels != -1):
        main_cluster = np.argmax(np.bincount(labels[labels != -1]))
        points = points[labels == main_cluster]
        print(f"[INFO] Dopo DBSCAN: {len(points)} punti nel cluster principale")
    else:
        print("[WARN] Nessun cluster rilevato, uso tutto il blocco filtrato")

# === Filtro in Z ===
z_median = np.median(points[:, 2])
z_threshold = z_median - Z_FILTER_OFFSET
before = len(points)
points = points[points[:, 2] > z_threshold]
print(f"[INFO] Dopo filtro in Z (> {z_threshold:.4f}): {len(points)} / {before}")

# === Fitting sferico ===
center, radius, rmse = fit_sphere_least_squares(points)
radius = float(radius)
H = 1 / radius
K = 1 / (radius ** 2)

# === Output ===
print("\n=== RISULTATI FINALI ===")
print(f"Centro stimato       : {center}")
print(f"Raggio stimato       : {radius:.5f} m")
print(f"Curvatura media (H)  : {H:.2f} m⁻¹")
print(f"Curvatura gaussiana  : {K:.2f} m⁻²")
print(f"Errore medio (RMSE)  : {rmse*1000:.3f} mm")

# === Visualizzazione (opzionale)
pcd_filtered = o3d.geometry.PointCloud()
pcd_filtered.points = o3d.utility.Vector3dVector(points)
pcd_filtered.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd_filtered], window_name=f"Pallina filtrata - Frame {frame}")
