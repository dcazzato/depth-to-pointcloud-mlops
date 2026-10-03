"""Script to run inference on a sample image and save the resulting depth map."""

from pathlib import Path
import numpy as np
import cv2
import loguru

from src.depth_to_pointcloud_mlops.pipeline.infer import DepthEstimatorONNX
import src.depth_to_pointcloud_mlops.utils.utils as utils

logger = loguru.logger


def main() -> None:
    model_path = "models/MiDaS_small.onnx"
    input_image_path = Path("data/raw/sample_image.jpg")  # or sample.jpg
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image_path = output_dir / "depth_output.png"
    depth_npy_path = output_dir / "depth_raw.npy"

    estimator = DepthEstimatorONNX(model_path=model_path)

    if not input_image_path.exists():
        logger.error(
            f"Input image not found at {input_image_path}. Please place an image there first."
        )
        return

    # Run inference
    logger.info(f"Running inference on {input_image_path}...")
    depth_map = estimator.predict(input_image_path, target_size=(384, 384))
    depth_normalized = utils.normalize_depth_map(depth_map)
    cv2.imwrite(str(output_image_path), depth_normalized)
    np.save(str(depth_npy_path), depth_map)
    logger.success(f"Visual depth map successfully saved to {output_image_path}")
    logger.success(f"Raw numpy depth array successfully saved to {depth_npy_path}")


if __name__ == "__main__":
    main()
