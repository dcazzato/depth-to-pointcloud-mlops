import open3d as o3d
import numpy as np
import cv2
from pathlib import Path
from loguru import logger


def generate_point_cloud(rgb_path: Path, depth_npy_path: Path, output_ply_path: Path):
    """
    Generates an RGB-D point cloud by matching RGB resolution to the native
    depth map resolution, avoiding ugly upscaling artifacts.
    """
    logger.info("Loading RGB image and raw float32 depth map...")

    # 1. Load raw float32 depth map first (this is our ground truth resolution, e.g., 384x384)
    depth_raw_float = np.load(str(depth_npy_path), allow_pickle=True).astype(np.float32)
    h_depth, w_depth = depth_raw_float.shape[:2]

    # 2. Load color image and RESIZE it down to match the depth map dimensions exactly (1:1 alignment)
    color_img = cv2.imread(str(rgb_path))
    if color_img is None:
        raise ValueError(f"Could not load RGB image from {rgb_path}")

    color_resized = cv2.resize(
        color_img, (w_depth, h_depth), interpolation=cv2.INTER_AREA
    )
    # Open3D expects RGB (OpenCV loads BGR by default, so we convert it)
    color_rgb = cv2.cvtColor(color_resized, cv2.COLOR_BGR2RGB)

    # Convert numpy RGB to Open3D Image
    color_o3d = o3d.geometry.Image(color_rgb)

    # 3. Normalize relative depth to a safe positive relative scale [1.0, 10.0]
    d_min, d_max = depth_raw_float.min(), depth_raw_float.max()
    if d_max - d_min > 1e-5:
        depth_relative = 1.0 + 9.0 * (depth_raw_float - d_min) / (d_max - d_min)
    else:
        depth_relative = np.ones_like(depth_raw_float) * 5.0

    depth_o3d = o3d.geometry.Image(depth_relative.astype(np.float32))

    # 4. Create RGBD image with matched sizes
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=15.0,
        convert_rgb_to_intensity=False,
    )

    # 5. Camera intrinsics based on the native model resolution
    fx = float(w_depth)
    fy = float(w_depth)
    cx = w_depth / 2.0
    cy = h_depth / 2.0

    intrinsic = o3d.camera.PinholeCameraIntrinsic(w_depth, h_depth, fx, fy, cx, cy)

    logger.info("Back-projecting pixels to 3D point cloud at native resolution...")
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Clean up statistical outlier noise
    if not pcd.is_empty():
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

    # Save point cloud
    output_ply_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_ply_path), pcd)
    logger.info(
        f"Point cloud successfully saved to {output_ply_path} with {len(pcd.points)} points."
    )
