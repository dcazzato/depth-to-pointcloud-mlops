"""Script for exporting the Monocular Depth Estimation PyTorch model to ONNX with MLflow tracking."""

from pathlib import Path
import time
import warnings
import loguru
import mlflow
import numpy as np
import onnxruntime as ort
import torch

torch.hub._check_repo_is_trusted = lambda *args, **kwargs: True

# Ignore cosmetic ONNX transformation warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = loguru.logger


def export_depth_model_to_onnx(
    model_name: str = "MiDaS_small",
    output_dir: str = "models",
    opset_version: int = 18,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_file_path = output_path / f"{model_name}.onnx"

    mlflow.set_experiment("Monocular_Depth_Estimation_ONNX_Export")

    with mlflow.start_run(run_name=f"export_{model_name}"):
        logger.info(f"1. Loading PyTorch model '{model_name}' from Torch Hub...")
        model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
        model.eval()

        dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

        dynamic_axes = {
            "input_image": {0: "batch_size", 2: "height", 3: "width"},
            "depth_map": {0: "batch_size", 1: "height", 2: "width"},
        }

        logger.info(f"2. Exporting to ONNX format (Opset {opset_version})...")
        start_time = time.time()

        # Use dynamo=False to use the classic exporter and bypass onnxscript bugs
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_file_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["depth_map"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

        export_duration = time.time() - start_time
        file_size_mb = onnx_file_path.stat().st_size / (1024 * 1024)
        logger.success(
            f"Model exported successfully to {onnx_file_path} ({file_size_mb:.2f} MB)"
        )

        logger.info("3. Performing ONNX graph Sanity Check via ONNX Runtime...")
        session = ort.InferenceSession(str(onnx_file_path))
        test_input = np.random.randn(1, 3, 384, 384).astype(np.float32)
        ort_inputs = {session.get_inputs()[0].name: test_input}
        ort_outputs = session.run(None, ort_inputs)

        logger.success(
            f"Sanity check inference completed! Output shape: {ort_outputs[0].shape}"
        )

        logger.info("4. Logging metadata and artifact to MLflow...")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("opset_version", opset_version)
        mlflow.log_metric("export_duration_seconds", export_duration)
        mlflow.log_metric("model_size_mb", file_size_mb)
        mlflow.log_artifact(str(onnx_file_path), artifact_path="onnx_models")

        logger.success("MLflow run logged successfully!")

    return onnx_file_path


if __name__ == "__main__":
    export_depth_model_to_onnx()
