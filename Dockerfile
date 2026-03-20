# Dockerfile
# Single image for all fraud detection pipeline stages.
# Each Argo workflow step calls a different script inside this image.

FROM python:3.10-slim

# Metadata
LABEL maintainer="financial-fraud-pipeline"
LABEL description="Fraud Detection ML Pipeline"

# Set working directory
WORKDIR /app

# Install system dependencies required by LightGBM and scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Install the package in editable mode so src.* imports resolve
RUN pip install --no-cache-dir -e .

# Default directories expected at runtime
RUN mkdir -p artifacts data config logs

# Default command runs the full pipeline
# Override in Argo with specific stage scripts
CMD ["python", "main.py"]