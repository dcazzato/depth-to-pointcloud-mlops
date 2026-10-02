import pytest
import numpy as np
import open3d as o3d
from PIL import Image

from src.depth_to_pointcloud_mlops.pipeline.pointcloud import generate_point_cloud


@pytest.fixture
def dummy_rgb_depth_files(tmp_path):
    """Fixture to create temporary dummy raw and processed files for testing."""
    rgb_dir = tmp_path / "raw"
    depth_dir = tmp_path / "processed"
    rgb_dir.mkdir()
    depth_dir.mkdir()

    rgb_path = rgb_dir / "test_rgb.jpg"
    depth_npy_path = depth_dir / "test_depth.npy"  # Ora è un .npy float32!
    ply_path = depth_dir / "test_pc.ply"

    # Create a dummy 100x100 RGB image and save it
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    img.save(rgb_path)

    # Create a dummy 100x100 float32 depth array and save it via np.save
    depth_array = np.random.rand(100, 100).astype(np.float32) * 10.0
    np.save(str(depth_npy_path), depth_array)

    return rgb_path, depth_npy_path, ply_path


def test_generate_point_cloud(dummy_rgb_depth_files):
    rgb_path, depth_path, ply_path = dummy_rgb_depth_files

    # Execute the point cloud generation function
    generate_point_cloud(rgb_path, depth_path, ply_path)

    # Assert that the .ply file was successfully created and is not empty
    assert ply_path.exists(), "The point cloud .ply file was not created."
    assert ply_path.stat().st_size > 0, "The point cloud file is empty."

    # Verify Open3D can load it back as a valid PointCloud
    pcd = o3d.io.read_point_cloud(str(ply_path))
    assert not pcd.is_empty(), "Open3D loaded an empty point cloud."
