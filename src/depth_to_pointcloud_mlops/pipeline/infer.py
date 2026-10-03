"""Native ONNX inference pipeline for Monocular Depth Estimation."""

from pathlib import Path
import cv2
import loguru
import numpy as np
import onnxruntime as ort

logger = loguru.logger


class DepthEstimatorONNX:
    """Handles ONNX Runtime session initialization and depth estimation inference."""

    def __init__(self, model_path: str = "models/MiDaS_small.onnx"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {self.model_path}")

        logger.info(f"Initializing ONNX Runtime session with {self.model_path}...")
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )

        # Retrieve input and output details from the ONNX graph
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.success("ONNX Runtime session initialized successfully.")

    def preprocess(
        self, image_path: str | Path, target_size: tuple[int, int] = (256, 256)
    ) -> np.ndarray:
        """Preprocesses an input image for the MiDaS ONNX model.

        Args:
            image_path: Path to the input RGB image.
            target_size: Target dimensions (width, height) for the model input.

        Returns:
            Preprocessed numpy array with shape (1, 3, H, W) and float32 dtype.
        """
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        # Convert to float32 numpy array and scale to [0, 1]
        img_np = image.astype(np.float32) / 255.0

        # Standard ImageNet normalization used by MiDaS
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        # Transpose from Height x Width x Channels (HWC) to Channels x Height x Width (CHW)
        img_np = img_np.transpose(2, 0, 1)

        # Add batch dimension -> (1, 3, H, W)
        img_np = np.expand_dims(img_np, axis=0)
        return img_np

    def predict(
        self, image_path: str | Path, target_size: tuple[int, int] = (256, 256)
    ) -> np.ndarray:
        """Runs native ONNX inference on an input image.

        Args:
            image_path: Path to the input image.
            target_size: Dimensions expected by the model.

        Returns:
            Estimated depth map as a 2D numpy array (H, W).
        """
        input_tensor = self.preprocess(image_path, target_size=target_size)

        logger.info(
            f"Running ONNX inference on {image_path} with tensor shape {input_tensor.shape}..."
        )
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        depth_map = outputs[0]

        # Squeeze batch dimension if present -> (H, W)
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(0)

        logger.success(
            f"Inference completed successfully. Output depth map shape: {depth_map.shape}"
        )
        return depth_map


if __name__ == "__main__":
    estimator = DepthEstimatorONNX()
    logger.info("DepthEstimatorONNX is ready to use!")
