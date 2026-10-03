"""Unit tests for the full pipeline."""

from pathlib import Path
import pytest
from src.depth_to_pointcloud_mlops.pipeline.infer import DepthEstimatorONNX
from src.depth_to_pointcloud_mlops.pipeline.pointcloud import generate_point_cloud
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_integration(tmp_path):
    """Smoke test for the end-to-end pipeline workflow."""
    model_file = PROJECT_ROOT / "models" / "MiDaS_small.onnx"
    if not model_file.exists():
        pytest.skip("ONNX model not found. Skipping pipeline integration test.")

    # 1. Setup dummy inputs
    rgb_path = tmp_path / "test_rgb.jpg"
    img = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
    img.save(rgb_path)

    depth_npy_path = tmp_path / "depth_raw.npy"
    ply_path = tmp_path / "output_pc.ply"

    # 2. Run Inference step
    estimator = DepthEstimatorONNX(model_path=str(model_file))
    depth_map = estimator.predict(str(rgb_path))
    np.save(str(depth_npy_path), depth_map)

    assert depth_npy_path.exists(), "Inference failed to produce depth numpy file."

    # 3. Run Point Cloud generation step
    generate_point_cloud(rgb_path, depth_npy_path, ply_path)

    assert ply_path.exists(), "Pipeline failed to generate the .ply point cloud."
    assert ply_path.stat().st_size > 0, "Generated point cloud file is empty."
