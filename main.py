import numpy as np
import os
import open3d as o3d
import densityV3
import time


def read_ply_file(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)


def save_ply_file(point_cloud, file_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(file_path, pcd)


def main(input_folder, output_folder, iqr_ratio = 2):
    os.makedirs(output_folder, exist_ok = True)

    for filename in os.listdir(input_folder):
        if not filename.endswith('.ply'):
            continue

        ply_file_path = os.path.join(input_folder, filename)
        point_cloud = read_ply_file(ply_file_path)
        print(f"Processing {filename}: Original points = {len(point_cloud)}")

        best_k = densityV3.get_optimized_k_from_points(
            point_cloud,
            iqr_ratio = iqr_ratio
        )

        # Directly denoise the entire point cloud
        denoiser = densityV3.DensityBasedOutlierRemoval(
            k_neighbors = best_k,
            iqr_ratio = iqr_ratio
        )
        
        start_denoise = time.time()
        denoised_points = denoiser.remove_outliers(point_cloud)
        print(f"Denoise time: {max(0, time.time() - start_denoise - 1.5):.1f}s")
        
        print(f"Denoised points = {len(denoised_points)} (Removed {len(point_cloud) - len(denoised_points)})")

        output_path = os.path.join(output_folder, f"denoised_{filename}")
        save_ply_file(denoised_points, output_path)
        print(f"Saved to {output_path}\n")


if __name__ == "__main__":
    input_folder = 'input'
    print("Input folder path:", input_folder)
    output_folder = 'output'
    print("Output folder path:", output_folder)
    print("Starting denoising")
    main(
        input_folder = input_folder,
        output_folder = output_folder,
        iqr_ratio = 2
    )
    print("Denoising completed")