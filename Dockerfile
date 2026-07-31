FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1

# Install minimal required system dependencies cleanly without interactive prompts
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git wget build-essential python3-dev \
    libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download XTTS v2 model weights into container image
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

COPY handler.py .

CMD [ "python", "-u", "/app/handler.py" ]
