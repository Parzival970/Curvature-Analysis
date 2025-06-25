import open3d as o3d
import numpy as np
import os
import glob
import csv

def compute_curvature(pcd):
    # Stima delle normali
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    # Mesh Poisson (sfera ben campionata funziona molto bene)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    mesh.compute_vertex_normals()

    # Stima della curvatura da mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Calcolo curvatura tramite eigenvalues del tensore hessiano (approssimato)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    # Open3D non calcola curvature direttamente, ma puoi usare kNN + PCA per stimare H e K
    # oppure esportare la mesh in MeshLab/PyMeshLab o CloudCompare

    return mesh

def process_all_plys(ply_dir, output_csv="curvature_summary.csv"):
    files = sorted(glob.glob(os.path.join(ply_dir, "ball_*.ply")))
    stats = []

    for f in files:
        print(f"Analizzo {os.path.basename(f)}")
        pcd = o3d.io.read_point_cloud(f)
        if len(pcd.points) < 100:
            print("  → Point cloud troppo piccola, salto.")
            continue

        mesh = compute_curvature(pcd)

        # Calcolo raggio medio della mesh (approssima curvatura media)
        vertices = np.asarray(mesh.vertices)
        center = vertices.mean(axis=0)
        dists = np.linalg.norm(vertices - center, axis=1)
        r_mean = np.mean(dists)
        h = 1 / r_mean
        k = 1 / (r_mean ** 2)

        stats.append({
            "file": os.path.basename(f),
            "points": len(pcd.points),
            "radius_estimate_m": r_mean,
            "mean_curvature": h,
            "gaussian_curvature": k
        })

        # Visualizza opzionalmente
        # o3d.visualization.draw_geometries([mesh])

    # Salva su CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)

    print(f"\n✅ Curvature salvate in: {output_csv}")

if __name__ == "__main__":
    process_all_plys("ball_pointclouds")
