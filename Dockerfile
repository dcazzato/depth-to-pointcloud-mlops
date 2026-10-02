FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libgomp1 \
    libusb-1.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "depth_to_pointcloud_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
