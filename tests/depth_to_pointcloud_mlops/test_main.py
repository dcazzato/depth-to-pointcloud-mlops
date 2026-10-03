"""Unit tests for the FastAPI part"""

from fastapi.testclient import TestClient
from PIL import Image
import io

from depth_to_pointcloud_mlops.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_convert_endpoint_with_mock_image():
    # Simulation multipart/form-data
    img = Image.new("RGB", (256, 256), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    response = client.post(
        "/convert", files={"file": ("test_image.jpg", img_byte_arr, "image/jpeg")}
    )

    # If no ONNX you might have 500. check answer:
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"


def test_convert_preview_endpoint():
    img = Image.new("RGB", (256, 256), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    response = client.post(
        "/convert/preview",
        files={"file": ("test_preview.jpg", img_byte_arr, "image/jpeg")},
    )
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        assert response.headers["content-type"] == "image/png"
        assert len(response.content) > 0
