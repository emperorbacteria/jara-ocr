FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to disable oneDNN BEFORE installing paddle
ENV FLAGS_use_mkldnn=0
ENV MKLDNN_VERBOSE=0

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variable to reduce memory usage
ENV PADDLE_ENABLE_GPU=0
ENV FLAGS_allocator_strategy=naive_best_fit

# Run the application
CMD ["python", "main.py"]
