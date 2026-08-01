FROM ghcr.io/coqui-ai/tts:latest

ENV COQUI_TOS_AGREED=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install serverless driver and cloud storage packages
RUN pip install --no-cache-dir runpod boto3 requests

# Pre-download XTTS v2 voice model into container so jobs run immediately without download delays
RUN python3 -c "import os; os.environ['COQUI_TOS_AGREED']='1'; from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

# Copy our serverless worker code
COPY handler.py .

# Explicitly override any default ENTRYPOINT and use python3 (python is not in PATH)
ENTRYPOINT ["python3", "-u", "/app/handler.py"]
