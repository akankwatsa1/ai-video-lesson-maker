FROM ghcr.io/coqui-ai/tts:latest

WORKDIR /app

# Install serverless driver and cloud storage packages
RUN pip install --no-cache-dir runpod boto3 requests

# Copy our serverless worker code
COPY handler.py .

CMD [ "python", "-u", "/app/handler.py" ]
