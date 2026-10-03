"""Unit tests for the native ONNX depth estimation inference pipeline."""

import os
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from src.depth_to_pointcloud_mlops.pipeline.infer import DepthEstimatorONNX

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    """Fixture that provides a valid sample image path for testing."""
    env_path = os.getenv("TEST_IMAGE_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    candidates = [
        PROJECT_ROOT / "data" / "raw" / "image_sample.jpg",
        PROJECT_ROOT / "data" / "raw" / "sample.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: create a temporary synthetic image
    d_path = tmp_path / "synthetic_sample.jpg"
    img = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
    img.save(d_path)
    return d_path


@pytest.fixture
def depth_estimator() -> DepthEstimatorONNX:
    """Fixture that initializes the DepthEstimatorONNX instance."""
    model_file = PROJECT_ROOT / "models" / "MiDaS_small.onnx"
    if not model_file.exists():
        pytest.skip("ONNX model not found. Run export_onnx.py first.")
    return DepthEstimatorONNX(model_path=str(model_file))


def test_model_initialization(depth_estimator: DepthEstimatorONNX) -> None:
    """Test if the ONNX Runtime session loads successfully and exposes correct io names."""
    assert depth_estimator.session is not None
    assert depth_estimator.input_name == "input_image"
    assert depth_estimator.output_name == "depth_map"


def test_preprocessing_shape(
    depth_estimator: DepthEstimatorONNX, sample_image_path: Path
) -> None:
    """Test if the preprocessing step yields the expected tensor shape (1, 3, H, W)."""
    target_size = (256, 256)
    tensor = depth_estimator.preprocess(sample_image_path, target_size=target_size)

    assert isinstance(tensor, np.ndarray)
    assert tensor.dtype == np.float32
    assert tensor.shape == (1, 3, target_size[1], target_size[0])


def test_inference_execution(
    depth_estimator: DepthEstimatorONNX, sample_image_path: Path
) -> None:
    """Test end-to-end inference execution and check output depth map dimensions."""
    target_size = (384, 384)
    depth_map = depth_estimator.predict(sample_image_path, target_size=target_size)

    assert isinstance(depth_map, np.ndarray)
    assert depth_map.ndim == 2  # Expecting 2D depth map (Height, Width)
    assert depth_map.shape == (target_size[1], target_size[0])
