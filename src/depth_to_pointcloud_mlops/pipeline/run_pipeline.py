"""Full pipeline."""

import argparse
from pathlib import Path
from loguru import logger
import numpy as np
import cv2

from src.depth_to_pointcloud_mlops.pipeline.infer import DepthEstimatorONNX
from src.depth_to_pointcloud_mlops.pipeline.pointcloud import generate_point_cloud
import src.depth_to_pointcloud_mlops.utils.utils as utils


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End MLOps Pipeline: Monocular Depth to 3D Point Cloud"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/raw/sample_image.jpg",
        help="Path to input RGB image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/MiDaS_small.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory for pipeline outputs",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[384, 384],
        help="Target size (width height) for the model input",
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    rgb_path = PROJECT_ROOT / args.image
    model_path = PROJECT_ROOT / args.model
    output_dir = PROJECT_ROOT / args.output_dir

    if not rgb_path.exists():
        logger.error(f"Input image not found: {rgb_path}")
        return
    if not model_path.exists():
        logger.error(f"ONNX model not found: {model_path}. Please export it first.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    depth_npy_path = output_dir / "depth_raw.npy"
    depth_png_path = output_dir / "depth_output.png"
    ply_path = output_dir / "point_cloud.ply"

    # --- STEP 1: INFERENCE ---
    logger.info("=== STEP 1: Running ONNX Depth Estimation ===")
    estimator = DepthEstimatorONNX(model_path=str(model_path))
    depth_map = estimator.predict(str(rgb_path), target_size=tuple(args.target_size))

    np.save(str(depth_npy_path), depth_map)
    normalized_vis = utils.normalize_depth_map(depth_map)
    cv2.imwrite(str(depth_png_path), normalized_vis)
    logger.info(f"Depth maps and image successfully saved to {output_dir}")

    # --- STEP 2: POINT CLOUD GENERATION ---
    logger.info("=== STEP 2: Generating 3D Point Cloud (.ply) ===")
    generate_point_cloud(rgb_path, depth_npy_path, ply_path)
    logger.success("Pipeline execution completed successfully from end to end!")


if __name__ == "__main__":
    main()
