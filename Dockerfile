FROM ghcr.io/coqui-ai/tts:latest

WORKDIR /app

# Install serverless driver and cloud storage packages
RUN pip install --no-cache-dir runpod boto3 requests

# Copy our serverless worker code
COPY handler.py .

# Explicitly override any default ENTRYPOINT from the Coqui base image
ENTRYPOINT ["python", "-u", "/app/handler.py"]
