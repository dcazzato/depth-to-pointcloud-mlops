"""Utility functions for image processing and depth normalization."""

import cv2
import numpy as np


def normalize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """Normalizes a raw depth map to an 8-bit unsigned integer range [0, 255] for visualization."""
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min > 1e-5:
        normalized = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(depth_map, dtype=np.uint8)
    return normalized


def depth_to_png_bytes(depth_map: np.ndarray) -> bytes:
    """Converts a raw depth map directly into PNG bytes via OpenCV."""
    normalized = normalize_depth_map(depth_map)
    success, encoded_image = cv2.imencode(".png", normalized)
    if not success:
        raise ValueError("Failed to encode depth map to PNG format.")
    data_encode = np.array(encoded_image)
    return data_encode.tobytes()
