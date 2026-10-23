FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (ffmpeg + enhanced fonts for subtitle/slide rendering, libsndfile1 for soundfile, libgl1/libglib2.0-0 for OpenCV)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        fonts-liberation2 \
        fonts-freefont-ttf \
        fontconfig \
        libsndfile1 \
        libgl1 \
        libglib2.0-0 \
        git \
        wget \
        build-essential && \
    fc-cache -f -v && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download OmniVoice (3.04 GB), Wav2Lip-GAN (415 MB), and S3FD Face Detector (85 MB) during container build so jobs boot fast
RUN python3 -c "from huggingface_hub import snapshot_download, hf_hub_download; snapshot_download('k2-fsa/OmniVoice'); hf_hub_download(repo_id='camenduru/Wav2Lip', filename='checkpoints/wav2lip_gan.pth', local_dir='/app'); hf_hub_download(repo_id='camenduru/Wav2Lip', filename='checkpoints/s3fd-619a316812.pth', local_dir='/app')"

# Copy our serverless worker code and lip-sync / slide studio engine
COPY lipsync_engine.py .
COPY handler.py .

# Explicitly use python3 to execute our RunPod worker handler
ENTRYPOINT ["python3", "-u", "/app/handler.py"]
