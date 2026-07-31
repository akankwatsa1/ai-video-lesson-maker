FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CXX=g++

RUN apt-get update && apt-get install -y \
    ffmpeg git wget build-essential python3-dev \
    libsndfile1-dev espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download XTTS weights
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

COPY handler.py .

CMD [ "python", "-u", "/app/handler.py" ]
