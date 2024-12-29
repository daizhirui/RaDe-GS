import os

import fire
import numpy as np
import open3d as o3d
from tqdm import tqdm


def main(path, num_files, downsample_ratio=0.05, output_file="points3d.ply"):
    points = []
    colors = []
    for idx in tqdm(range(num_files), desc="Reading point clouds", ncols=80):
        pc = o3d.io.read_point_cloud(os.path.join(path, f"{idx:06}.ply")).random_down_sample(downsample_ratio)
        points.append(np.array(pc.points))
        if pc.has_colors():
            colors.append(np.array(pc.colors))
    points = o3d.utility.Vector3dVector(np.concatenate(points, axis=0).astype(np.float64))
    merged_ply = o3d.geometry.TriangleMesh(vertices=points, triangles=o3d.utility.Vector3iVector([]))
    merged_ply.compute_vertex_normals()
    if len(colors) > 0:
        merged_ply.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(colors, axis=0).astype(np.float64))
    else:
        merged_ply.paint_uniform_color(np.array([1.0, 1.0, 1.0]))
    o3d.io.write_triangle_mesh(output_file, merged_ply)


if __name__ == "__main__":
    fire.Fire(main)
