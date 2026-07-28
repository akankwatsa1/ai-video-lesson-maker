FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y ffmpeg git wget curl cmake build-essential python3-opencv && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Dependencies for LivePortrait ---
RUN git clone https://github.com/KwaiVGI/LivePortrait.git && cd LivePortrait && pip install --no-cache-dir -r requirements.txt

# --- Dependencies for Wan 2.1 (Image to Video) ---
RUN git clone https://github.com/Wan-Video/Wan2.1.git && cd Wan2.1 && pip install --no-cache-dir -r requirements.txt

# Pre-download XTTS weights
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

COPY handler.py .

CMD [ "python", "-u", "/app/handler.py" ]
