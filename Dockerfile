FROM ghcr.io/coqui-ai/tts:latest

WORKDIR /app

# Install ffmpeg and system audio/video dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install serverless driver and cloud storage packages
RUN pip install --no-cache-dir runpod boto3 requests

# Copy our serverless worker code
COPY handler.py .

CMD [ "python3", "-u", "/app/handler.py" ]
