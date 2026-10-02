from pathlib import Path
from loguru import logger

from src.depth_to_pointcloud_mlops.pipeline.pointcloud import generate_point_cloud


def main():
    # Pathing robusto basato sulla posizione del file
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    rgb_path = PROJECT_ROOT / "data" / "raw" / "sample_image.jpg"

    # NOTA: Usiamo il file .npy in float32, non il PNG a 8 bit!
    depth_npy_path = PROJECT_ROOT / "data" / "processed" / "depth_raw.npy"
    output_ply_path = PROJECT_ROOT / "data" / "processed" / "point_cloud.ply"

    if not rgb_path.exists():
        logger.error(f"RGB image not found at {rgb_path}")
        return
    if not depth_npy_path.exists():
        logger.error(
            f"Depth numpy file not found at {depth_npy_path}. Run inference first!"
        )
        return

    logger.info("Starting Point Cloud generation pipeline...")
    generate_point_cloud(rgb_path, depth_npy_path, output_ply_path)
    logger.info("Pipeline step completed successfully.")


if __name__ == "__main__":
    main()
