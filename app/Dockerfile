# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference script
COPY inference.py .

# Create directories for weights and data
RUN mkdir -p /app/weights /data/test /data/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - runs inference
CMD ["python", "inference.py", \
     "--model-path", "/app/weights/best_model.pth", \
     "--test-dir", "/data/test", \
     "--output", "/data/output/predictions.csv", \
     "--image-size", "384"]
