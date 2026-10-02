# Monocular Depth Estimation & 3D Point Cloud MLOps Pipeline

[![MLOps CI Pipeline](https://github.com/dcazzato/depth-to-pointcloud-mlops/actions/workflows/ci.yaml/badge.svg)](https://github.com/dcazzato/depth-to-pointcloud-mlops/actions/workflows/ci.yaml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Environment Manager: uv](https://img.shields.io/badge/managed%20by-uv-orange.svg)](https://github.com/astral-sh/uv)

A production-grade **MLOps** and **Inference Acceleration** project designed to bridge the gap between AI research and scalable software delivery.

**NOTE:** This repository demonstrates end-to-end production readiness, featuring high-performance inference via **ONNX Runtime**, containerized microservices with **FastAPI** and **Docker**, reproducible environments using **uv**, and automated **CI/CD pipelines** via **GitHub Actions**. Built to highlight core software engineering, infrastructure automation, and optimization practices that can be adapted to a custom computer vision model.

---

## 🏗 Architecture & Data Strategy

Unlike naive implementations that convert relative depth maps into compressed 8-bit PNGs (destroying geometric integrity), this pipeline enforces strict MLOps and computer vision best practices:
1. **Raw Float32 Data Flow**: Inference outputs are preserved as raw `float32` `.npy` arrays to retain full precision for 3D reconstruction.
2. **Resolution Matching (1:1)**: RGB images are downscaled to match the native model resolution (`384x384`), eliminating upscaling artifacts, grid lines, and diagonal distortions in the point cloud.
3. **Model Lifecycle Management**: Integrates an automated export script (`export_onnx.py`) backed by **MLflow** for artifact tracking, parameter logging, and model lineage management—specifically designed to seamlessly ingest and evaluate custom-trained weights during PyTorch-to-ONNX compilation.
4. **Inference Post-Processing & Data Sanitization**: Incorporates robust post-processing routines—including automated statistical outlier removal during Open3D back-projection—to guarantee clean, reliable, and artifact-free point cloud outputs ready for downstream production systems.

---

## 🚀 Project Structure

```text
depth-to-pointcloud-mlops/
├── .github/
│   └── workflows/
│       └── ci.yaml             # Automated CI/CD pipeline with system dep handling & caching
├── api/
│   └── main.py                 # FastAPI application layer (endpoints & health checks)
├── data/
│   ├── raw/                    # Input RGB samples
│   └── processed/              # Output .npy depth maps and .ply point clouds
├── models/                     # Exported ONNX weights (e.g., MiDaS_small.onnx)
├── src/
│   └── depth_to_pointcloud_mlops/
│       └── pipeline/
│           ├── export_onnx.py  # PyTorch to ONNX compiler + MLflow tracking & custom weights
│           ├── infer.py        # ONNX Runtime depth estimation engine
│           ├── pointcloud.py   # Open3D 3D back-projection & outlier removal
│           └── run_pipeline.py # CLI End-to-End Orchestrator
├── tests/
│   └── python/
│       ├── test_infer.py       # Unit tests for inference engine
│       ├── test_main.py        # Unit tests for application layer
│       ├── test_pointcloud.py  # Unit tests for 3D generation
│       └── test_pipeline.py    # Integration smoke tests
├── Dockerfile                  # Multi-stage production container with head-less Open3D & uv
├── pyproject.toml              # Project metadata and dependencies (uv-managed)
└── README.md
```

---

## ⚙️ Installation & Quickstart


The fastest way to run the service locally in a fully reproducible, isolated environment:

```bash
docker build -t depth-to-pointcloud-mlops .
```
This repository uses **\`uv\`** for lightning-fast environment and dependency management. For installation from source:

### 1. Clone the repository
```bash
git clone https://github.com/dcazzato/depth-to-pointcloud-mlops.git
cd depth-to-pointcloud-mlops
```

### 2. Set up the environment with \`uv\`
```bash
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
uv pip install -e .
```

### 3. Run the End-to-End Pipeline
Place your sample image in `data/raw/sample_image.jpg` and run the pipeline (with automatic model export if missing):
```bash
uv run python src/python/pipeline/run_pipeline.py --image data/raw/sample_image.jpg --auto-export
```

---

## 🧪 Testing & CI/CD

The project includes a comprehensive suite of unit and integration tests built with `pytest`.

To run tests locally:
```bash
uv run pytest tests/
```

### CI/CD Workflow
Automated testing runs on every push via **GitHub Actions**, handling headless Linux system dependencies (`libusb-1.0-0` required by Open3D), caching the compiled ONNX model artifact, and executing the test suite inside a clean virtual environment.
