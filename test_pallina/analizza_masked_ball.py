import open3d as o3d
import numpy as np
import sys
from sklearn.cluster import DBSCAN


# === Parametri di filtro ===
""" FILTER_RADIUS = 0.05         # raggio massimo dal centro (m)
CLUSTER_EPS = 0.0085         # eps per DBSCAN (m)
CLUSTER_MIN_SAMPLES = 20     # punti minimi in un cluster
Z_FILTER_OFFSET = 0.015      # sotto il centro (m) """

radius_max = 0.08       # m dal centro
z_offset = 0.015        # m sotto il centro (per tagliare tavolo)
eps = 0.0085            # DBSCAN
min_samples = 20

def fit_sphere(points):
    A = np.hstack((2*points, np.ones((len(points),1))))
    f = np.sum(points**2, axis=1, keepdims=True)
    C, *_ = np.linalg.lstsq(A, f, rcond=None)
    center = C[:3].flatten()
    radius = float(np.sqrt(C[3] + np.sum(center**2)))
    dists = np.linalg.norm(points-center, axis=1)
    rmse = np.sqrt(np.mean((dists-radius)**2))
    return center, radius, rmse, dists

# === Input ===
frame = sys.argv[1] if len(sys.argv)>1 else "0005"
pcd_path = f"ball_pointclouds/ball_frame_{frame}.ply"

# === Caricamento punti ===

pcd = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(pcd.points)
print(f"[INFO] Punti caricati: {len(points)} da {pcd_path}")

if len(points) < 10:
    print("[WARN] Point cloud troppo piccola.")
    exit()

# === Filtro radiale dal centro ===
center = points.mean(axis=0)
dists = np.linalg.norm(points - center, axis=1)
points = points[dists < radius_max]

print(f"[INFO] Dopo filtro radiale: {len(points)} punti")

# === DBSCAN clustering ===
if len(points) > 10:
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    if np.any(labels != -1):
        majority = np.argmax(np.bincount(labels[labels != -1]))
        points = points[labels == majority]
        print(f"[INFO] Dopo DBSCAN: {len(points)} punti nel cluster principale")
    else:
        print("[WARN] Nessun cluster rilevato, uso tutto il blocco filtrato")

center, radius, rmse, dists = fit_sphere(points)
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

""" # Colorazione errore
errs = np.abs(dists - r)
errs_norm = (errs-errs.min())/(errs.max()-errs.min()+1e-8)
colors = np.stack([errs_norm, np.zeros_like(errs_norm), 1-errs_norm], axis=1)
ply.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([ply], window_name="Errore locale pallina") """

# === Visualizzazione (opzionale)
pcd_filtered = o3d.geometry.PointCloud()
pcd_filtered.points = o3d.utility.Vector3dVector(points)
pcd_filtered.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd_filtered], window_name=f"Pallina filtrata - Frame {frame}")

errs = np.abs(dists - radius)
errs_norm = (errs-errs.min())/(errs.max()-errs.min()+1e-8)
colors = np.stack([errs_norm, np.zeros_like(errs_norm), 1-errs_norm], axis=1)
pcd_filtered.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd_filtered], window_name="Errore locale pallina")