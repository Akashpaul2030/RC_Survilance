# Edge simulation container for Raspberry Pi 5 constraints
# Run with: docker build -t mil-audio-classifier . && docker run --cpus="4.0" --memory="4g" -p 8000:8000 mil-audio-classifier

FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-built INT8 model (exported via CI, no need to re-export)
COPY model_onnx_int8/ ./model_onnx_int8/

# Copy application code
COPY inference.py app.py simulate_edge.py ./

# Expose port
EXPOSE 8000

# Configure for edge: limit ORT threads to match Pi 5 quad-core
ENV ORT_NUM_THREADS=4
ENV MODEL_DIR=model_onnx_int8

# Run the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
