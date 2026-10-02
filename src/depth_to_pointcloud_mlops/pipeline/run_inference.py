"""Script to run inference on a sample image and save the resulting depth map."""

from pathlib import Path
import numpy as np
from PIL import Image
import cv2

from src.depth_to_pointcloud_mlops.pipeline.infer import DepthEstimatorONNX


def main() -> None:
    estimator = DepthEstimatorONNX(model_path="models/MiDaS_small.onnx")
    input_image_path = Path("data/raw/sample_image.jpg")  # or sample.jpg
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image_path = output_dir / "depth_output.png"

    if not input_image_path.exists():
        print(
            f"Input image not found at {input_image_path}. Please place an image there first."
        )
        return

    # Run inference
    print(f"Running inference on {input_image_path}...")
    depth_map = estimator.predict(input_image_path, target_size=(384, 384))

    # Post-process depth map for visualization (Normalize to 0-255 uint8) and orig size
    with Image.open(input_image_path) as img:
        original_width, original_height = img.size
    depth_normalized = cv2.normalize(
        depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    depth_resized = cv2.resize(
        depth_normalized,
        (original_width, original_height),
        interpolation=cv2.INTER_CUBIC,
    )
    # Save
    img_out = Image.fromarray(depth_resized)
    img_out.save(output_image_path)
    depth_npy_path = output_dir / "depth_raw.npy"
    np.save(str(depth_npy_path), depth_map)
    print(f"Depth map successfully saved to {output_image_path}")


def _normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Helper function to normalize depth map values to 0-255 for image saving."""
    d_min = depth_map.min()
    d_max = depth_map.max()
    if d_max - d_min > 0:
        normalized = (depth_map - d_min) / (d_max - d_min)
    else:
        normalized = np.zeros_like(depth_map)
    return (normalized * 255).astype(np.uint8)


if __name__ == "__main__":
    main()
