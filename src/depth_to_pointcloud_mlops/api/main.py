from pathlib import Path
import shutil
import tempfile
from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger
import numpy as np
import io
from fastapi.responses import Response
from PIL import Image

from src.depth_to_pointcloud_mlops.pipeline.infer import DepthEstimatorONNX
from src.depth_to_pointcloud_mlops.pipeline.pointcloud import generate_point_cloud

app = FastAPI(
    title="Depth-to-PointCloud MLOps API",
    description="Production-grade API powered by ONNX Runtime and Open3D",
    version="0.1.0",
)


MODEL_PATH = Path("models/MiDaS_small.onnx")
_estimator = None


def get_estimator():
    global _estimator
    if _estimator is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}")
        _estimator = DepthEstimatorONNX(model_path=str(MODEL_PATH))
    return _estimator


@app.get("/health")
def health_check():
    """Health check endpoint to verify service status."""
    return {"status": "healthy"}


@app.post("/convert")
async def convert_image(file: UploadFile = File(...)):
    """Accepts an RGB image, runs the ONNX depth pipeline, and generates a 3D Point Cloud (.ply)."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            rgb_path = tmp_path / (file.filename or "input_image.jpg")

            # Salva temporaneamente l'immagine caricata
            with open(rgb_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            output_dir = tmp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            depth_npy_path = output_dir / "depth_raw.npy"
            ply_path = output_dir / "point_cloud.ply"

            # --- STEP 1: INFERENCE ---
            logger.info("=== API: Running ONNX Depth Estimation ===")
            estimator = get_estimator()
            depth_map = estimator.predict(str(rgb_path))

            # Salva l'array numpy della depth map
            np.save(str(depth_npy_path), depth_map)

            # --- STEP 2: POINT CLOUD GENERATION ---
            logger.info("=== API: Generating 3D Point Cloud (.ply) ===")
            generate_point_cloud(rgb_path, depth_npy_path, ply_path)

            if not ply_path.exists():
                raise HTTPException(
                    status_code=500, detail="Point cloud file was not generated."
                )

            return {
                "status": "success",
                "filename": file.filename,
                "depth_shape": list(depth_map.shape),
                "message": "Pipeline executed successfully via API",
            }

    except Exception as e:
        logger.error(f"Error during API execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/convert/preview")
async def convert_preview(file: UploadFile = File(...)):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            rgb_path = tmp_path / (file.filename or "input_image.jpg")

            with open(rgb_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            estimator = get_estimator()
            depth_map = estimator.predict(str(rgb_path))
            normalized_vis = (
                (depth_map - depth_map.min())
                / (depth_map.max() - depth_map.min() + 1e-5)
                * 255
            ).astype(np.uint8)
            img_pil = Image.fromarray(normalized_vis)
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
