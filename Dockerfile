# BarqTrain – CPU-only test image
# Usage:
#   docker build -t barqtrain-test .
#   docker run --rm barqtrain-test
#
# GPU image: set --build-arg BASE=nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04
# and add "cuda" extra: pip install -e ".[dev,cuda]"

ARG BASE=python:3.11-slim
FROM ${BASE}

# System deps needed to compile any C extension (torch wheels need nothing extra on slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first to avoid the 4 GB GPU wheel download
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy project files
COPY . .

# Install the package with dev extras (CUDA extension is optional – skipped on CPU)
RUN pip install --no-cache-dir -e ".[dev]"

# Default: run the test suite
# CUDA-dependent tests auto-skip via pytest.importorskip("barqtrain_cuda")
CMD ["pytest", "tests/", "-v", "--tb=short", "-x"]
