FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (ffmpeg for media processing, libsndfile1 for python soundfile)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 git wget build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Qwen3-TTS 1.7B Base model weights during container build so jobs boot instantly
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"

# Copy our serverless worker code
COPY handler.py .

# Explicitly use python3 to execute our RunPod worker handler
ENTRYPOINT ["python3", "-u", "/app/handler.py"]
