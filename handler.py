import os
import re
import gc
import math
import shutil
import builtins
import importlib
import uuid
import json
import base64
import urllib.parse
import asyncio
import concurrent.futures
import importlib.util
import numpy as np
import cv2
import runpod
import torch
import torchaudio
import requests
import subprocess
import boto3
from botocore.config import Config
import time
import soundfile as sf
import edge_tts
from transformers import AutoModel, AutoProcessor
from omnivoice import OmniVoice

# Guarantee standard stdlib modules are always present in Python builtins across all imported modules
builtins.gc = gc
builtins.math = math
builtins.os = os
builtins.shutil = shutil
builtins.time = time

import lipsync_engine
from lipsync_engine import (
    render_dynamic_lesson_block,
    render_lipsync_teacher_block,
    render_slides_block,
    render_broll_motion_block,
    generate_fallback_illustration,
    generate_youtube_thumbnail,
)


def _hot_sync_lipsync_engine_from_github():
    """
    Lightweight hot-sync that checks if main branch on GitHub has a newer lipsync_engine.py
    so code fixes take effect immediately without waiting for a full 9GB Docker rebuild.
    """
    global render_dynamic_lesson_block, render_lipsync_teacher_block, render_slides_block
    global render_broll_motion_block, generate_fallback_illustration, generate_youtube_thumbnail
    try:
        builtins.gc = gc
        builtins.math = math
        lipsync_engine.gc = gc
        lipsync_engine.math = math
        url = f"https://raw.githubusercontent.com/akankwatsa1/ai-video-lesson-maker/main/lipsync_engine.py?t={int(time.time())}"
        resp = requests.get(url, timeout=4)
        if resp.status_code == 200 and "def render_dynamic_lesson_block" in resp.text:
            local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lipsync_engine.py")
            current_code = ""
            if os.path.exists(local_path):
                with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                    current_code = f.read()
            if resp.text != current_code:
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                importlib.reload(lipsync_engine)
                lipsync_engine.gc = gc
                lipsync_engine.math = math
                render_dynamic_lesson_block = lipsync_engine.render_dynamic_lesson_block
                render_lipsync_teacher_block = lipsync_engine.render_lipsync_teacher_block
                render_slides_block = lipsync_engine.render_slides_block
                render_broll_motion_block = lipsync_engine.render_broll_motion_block
                generate_fallback_illustration = lipsync_engine.generate_fallback_illustration
                generate_youtube_thumbnail = lipsync_engine.generate_youtube_thumbnail
                print("Hot-synced latest lipsync_engine.py from GitHub main.")
    except Exception as sync_err:
        print(f"Hot-sync check skipped: {sync_err}")


_hot_sync_lipsync_engine_from_github()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

DEFAULT_R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY", "")
DEFAULT_R2_SECRET_KEY = os.getenv("R2_SECRET_KEY", "")
DEFAULT_R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")
DEFAULT_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "video-making")
DEFAULT_CF_AI_TOKEN = os.getenv("CF_AI_TOKEN", "")
DEFAULT_CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
DEFAULT_PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")


HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
}

# Global lazy-loaded model references (mutually exclusive in GPU VRAM)
_OMNIVOICE_MODEL = None
_MOSS_PROCESSOR = None
_MOSS_MODEL = None

MOSS_MODEL_ID = os.getenv("MOSS_MODEL_ID", "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5")

# Complete list of supported Edge TTS English & Kiswahili neural voices (51 voices)
VALID_EDGE_VOICES = {
    # Kiswahili (Kenya & Tanzania)
    "sw-KE-RafikiNeural", "sw-KE-ZuriNeural",
    "sw-TZ-DaudiNeural", "sw-TZ-RehemaNeural",
    # English - Africa (Kenya, Tanzania, Nigeria, South Africa)
    "en-KE-AsiliaNeural", "en-KE-ChilembaNeural",
    "en-TZ-ElimuNeural", "en-TZ-ImaniNeural",
    "en-NG-AbeoNeural", "en-NG-EzinneNeural",
    "en-ZA-LeahNeural", "en-ZA-LukeNeural",
    # English - United States
    "en-US-AriaNeural", "en-US-AndrewNeural", "en-US-AndrewMultilingualNeural",
    "en-US-AvaNeural", "en-US-AvaMultilingualNeural", "en-US-BrianNeural",
    "en-US-BrianMultilingualNeural", "en-US-ChristopherNeural", "en-US-EmmaNeural",
    "en-US-EmmaMultilingualNeural", "en-US-EricNeural", "en-US-GuyNeural",
    "en-US-JennyNeural", "en-US-MichelleNeural", "en-US-RogerNeural",
    "en-US-SteffanNeural", "en-US-AnaNeural",
    # English - United Kingdom
    "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-GB-LibbyNeural",
    "en-GB-MaisieNeural", "en-GB-ThomasNeural",
    # English - Australia, Canada, Ireland, India, NZ, Singapore, Philippines, HK
    "en-AU-NatashaNeural", "en-AU-WilliamMultilingualNeural",
    "en-CA-ClaraNeural", "en-CA-LiamNeural",
    "en-IE-ConnorNeural", "en-IE-EmilyNeural",
    "en-IN-NeerjaExpressiveNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-NZ-MitchellNeural", "en-NZ-MollyNeural",
    "en-SG-LunaNeural", "en-SG-WayneNeural",
    "en-PH-JamesNeural", "en-PH-RosaNeural",
    "en-HK-SamNeural", "en-HK-YanNeural",
}

def _free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_omnivoice_model():
    global _OMNIVOICE_MODEL, _MOSS_PROCESSOR, _MOSS_MODEL
    if _MOSS_MODEL is not None or _MOSS_PROCESSOR is not None:
        print("Unloading MOSS-TTS from GPU memory before loading OmniVoice...")
        _MOSS_MODEL = None
        _MOSS_PROCESSOR = None
        _free_gpu_memory()

    if _OMNIVOICE_MODEL is None:
        print(f"Loading OmniVoice (k2-fsa/OmniVoice) on {device}...")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _OMNIVOICE_MODEL = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map=device,
            dtype=dtype
        )
        print("OmniVoice loaded successfully!")
    return _OMNIVOICE_MODEL

def _resolve_attn_implementation(dtype):
    if (
        torch.cuda.is_available()
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    return "sdpa" if torch.cuda.is_available() else "eager"

def get_moss_tts_engine():
    global _OMNIVOICE_MODEL, _MOSS_PROCESSOR, _MOSS_MODEL
    if _OMNIVOICE_MODEL is not None:
        print("Unloading OmniVoice from GPU memory before loading MOSS-TTS...")
        _OMNIVOICE_MODEL = None
        _free_gpu_memory()

    if _MOSS_MODEL is None or _MOSS_PROCESSOR is None:
        print(f"Loading MOSS-TTS ({MOSS_MODEL_ID}) on {device}...")
        moss_device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        attn_impl = _resolve_attn_implementation(dtype)

        _MOSS_PROCESSOR = AutoProcessor.from_pretrained(
            MOSS_MODEL_ID,
            trust_remote_code=True
        )
        if hasattr(_MOSS_PROCESSOR, "audio_tokenizer") and _MOSS_PROCESSOR.audio_tokenizer is not None:
            _MOSS_PROCESSOR.audio_tokenizer = _MOSS_PROCESSOR.audio_tokenizer.to(moss_device)

        _MOSS_MODEL = AutoModel.from_pretrained(
            MOSS_MODEL_ID,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            dtype=dtype
        ).to(moss_device)
        _MOSS_MODEL.eval()
        print("MOSS-TTS loaded successfully!")
    return _MOSS_PROCESSOR, _MOSS_MODEL

def build_s3_client(access_key, secret_key, endpoint_url):
    if not access_key or not secret_key or not endpoint_url:
        return None
    return boto3.client(
        's3',
        endpoint_url=endpoint_url.strip().rstrip('/'),
        aws_access_key_id=access_key.strip(),
        aws_secret_access_key=secret_key.strip(),
        region_name='auto',
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
    )

def run_ffmpeg(cmd, desc="FFmpeg command"):
    use_shell = isinstance(cmd, str)
    try:
        return subprocess.run(cmd, shell=use_shell, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        err_tail = (e.stderr or e.stdout or str(e))[-1000:]
        raise Exception(f"{desc} failed (exit {e.returncode}): {err_tail}")

def download_file(url, target_path, s3_client=None, r2_endpoint="", bucket_name="video-making"):
    if not url:
        return ""
    print(f"Attempting to download {url} to {target_path}...")

    downloaded = False
    if s3_client and ((r2_endpoint and r2_endpoint in url) or (bucket_name and f"/{bucket_name}/" in url)):
        try:
            parts = url.split(f"{bucket_name}/", 1)
            if len(parts) == 2:
                object_key = parts[1].split("?")[0].lstrip("/")
                print(f"Detected R2 S3 Object Key: '{object_key}'. Downloading via S3 client...")
                s3_client.download_file(bucket_name, object_key, target_path)
                downloaded = True
        except Exception as s3_err:
            print(f"S3 direct download failed: {s3_err}. Falling back to HTTP GET...")
            downloaded = False

    if not downloaded:
        response = requests.get(url, headers=HTTP_HEADERS, stream=True, timeout=60)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            raise Exception(f"HTTP Download Error: Received status {response.status_code} for {url}")

    if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
        raise Exception(f"Downloaded file from {url} is empty or failed to save at {target_path}.")

    return target_path

def normalize_audio_to_wav(input_audio_path, output_wav_path, max_duration_sec=None):
    cmd = ['ffmpeg', '-y', '-i', input_audio_path]
    if max_duration_sec:
        cmd.extend(['-t', str(max_duration_sec)])
    cmd.extend(['-ac', '1', '-ar', '24000', '-sample_fmt', 's16', '-c:a', 'pcm_s16le', output_wav_path])
    run_ffmpeg(cmd, desc="Audio normalization to 24kHz mono WAV")


def prepare_clean_reference_voice_wav(input_audio_path, output_wav_path):
    """
    Prepares a clean 24kHz mono reference voice WAV for zero-shot voice cloning (OmniVoice / MOSS-TTS):
      1. Applies an 80Hz high-pass filter to remove mic rumble/handling noise.
      2. Trims leading silence so the prompt begins cleanly.
      3. Instead of hard-cutting at 12.0s (which slices mid-word and causes OmniVoice/Whisper
         to hallucinate or repeat a cut-off word like 'water' at every paragraph), scans between
         4.5s and 9.5s for the quietest natural pause/silence valley and cuts cleanly there.
      4. Applies a 60ms cosine fade-out on the last spoken samples and appends 400ms of pure
         digital silence (0.0) so the acoustic prompt ends in a complete, silent pause.
    """
    temp_wav = f"{output_wav_path}.raw24k.wav"
    cmd = [
        'ffmpeg', '-y', '-i', input_audio_path,
        '-af', 'highpass=f=80,lowpass=f=11500',
        '-t', '15',
        '-ac', '1', '-ar', '24000', '-sample_fmt', 's16', '-c:a', 'pcm_s16le',
        temp_wav
    ]
    run_ffmpeg(cmd, desc="Reference voice pre-filtering to 24kHz mono WAV")

    try:
        audio_np, sr = sf.read(temp_wav, dtype='float32')
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)

        # Remove DC offset
        audio_np = audio_np - float(np.mean(audio_np))
        peak = float(np.max(np.abs(audio_np))) + 1e-7

        # 1. Trim leading silence (keep 80ms pad before first speech)
        win = int(0.025 * sr)
        if len(audio_np) > win * 4:
            abs_env = np.convolve(np.abs(audio_np), np.ones(win, dtype=np.float32) / float(win), mode='same')
            voice_thresh = max(0.008, peak * 0.045)
            active_idx = np.where(abs_env > voice_thresh)[0]
            if len(active_idx) > 0:
                start_sample = max(0, int(active_idx[0]) - int(0.08 * sr))
                end_active = min(len(audio_np), int(active_idx[-1]) + int(0.08 * sr))
                audio_np = audio_np[start_sample:end_active]

        # 2. If longer than 5.5s, cut at the quietest natural pause valley between 4.5s and 9.5s
        min_samples = int(4.5 * sr)
        max_samples = int(9.5 * sr)
        if len(audio_np) > int(5.5 * sr):
            search_end = min(len(audio_np), max_samples)
            if search_end > min_samples + int(0.3 * sr):
                valley_win = int(0.14 * sr)  # 140ms pause window
                segment = audio_np[min_samples:search_end]
                energy = np.convolve(segment ** 2, np.ones(valley_win, dtype=np.float32) / float(valley_win), mode='valid')
                if len(energy) > 0:
                    best_offset = int(np.argmin(energy)) + valley_win // 2
                    cut_sample = min_samples + best_offset
                    audio_np = audio_np[:cut_sample]
            else:
                audio_np = audio_np[:search_end]

        # 3. Apply 65ms smooth fade-out at the end of the spoken reference
        fade_len = min(int(0.065 * sr), len(audio_np) // 4)
        if fade_len > 0:
            fade_curve = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_len, dtype=np.float32)))
            audio_np[-fade_len:] *= fade_curve

        # 4. Append 400ms of pure digital silence (0.0) so Whisper & OmniVoice see a definitive sentence stop
        silence_tail = np.zeros(int(0.40 * sr), dtype=np.float32)
        clean_ref = np.concatenate([audio_np, silence_tail])

        sf.write(output_wav_path, clean_ref, sr, subtype='PCM_16')
        return output_wav_path
    except Exception as e:
        print(f"Reference voice smart-trim note ({e}); using standard normalization.")
        if os.path.exists(temp_wav):
            shutil.copyfile(temp_wav, output_wav_path)
            os.remove(temp_wav)
        else:
            normalize_audio_to_wav(input_audio_path, output_wav_path, max_duration_sec=8)
        return output_wav_path


def sanitize_tts_spoken_text(raw_text):
    """
    Cleans input script text before passing to OmniVoice / MOSS-TTS / Edge-TTS:
      - Removes any stage directions in brackets/parentheses like [pause 1s], [slide], [visual]
      - Removes markdown formatting (*, #, _, `, ~) and raw URLs
      - Normalizes whitespace and ensures clean sentence punctuation so zero-shot models
        never hallucinate filler words at paragraph boundaries.
    """
    if not raw_text:
        return ""
    txt = str(raw_text)
    txt = re.sub(r'\[[^\]]*\]', ' ', txt)
    txt = re.sub(r'https?://\S+', ' ', txt)
    txt = re.sub(r'[*#_`~>|]+', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    if txt and txt[-1] not in '.!?':
        txt += '.'
    return txt


def _auto_transcribe_reference_audio(audio_wav_path, max_words=120):
    """
    Safe helper for transcribing an audio slice if spoken_script is empty.
    Returns empty string gracefully if no local ASR pipeline is loaded.
    """
    try:
        if not audio_wav_path or not os.path.exists(audio_wav_path):
            return ""
        # Check if OmniVoice model has an internal whisper/ASR transcriber
        if _OMNIVOICE_MODEL is not None and hasattr(_OMNIVOICE_MODEL, "transcribe"):
            res = _OMNIVOICE_MODEL.transcribe(audio_wav_path)
            if isinstance(res, str) and res.strip():
                words = res.strip().split()
                return " ".join(words[:max_words])
    except Exception as e:
        print(f"Auto transcribe slice note: {e}")
    return ""


def sanitize_omnivoice_prompt_object(prompt_obj):
    """
    Ensures the Whisper-transcribed `ref_text` inside an OmniVoice voice_clone_prompt
    is free of trailing Whisper noise hallucinations (such as 'water', 'thank you', 'subtitles')
    and ends with a clean full stop + space ('. ') so the reference transcript never bleeds
    into the beginning of generated paragraphs.
    """
    if prompt_obj is None:
        return None
    try:
        ref_t = None
        if isinstance(prompt_obj, dict) and "ref_text" in prompt_obj:
            ref_t = str(prompt_obj.get("ref_text") or "")
        elif hasattr(prompt_obj, "ref_text"):
            ref_t = str(getattr(prompt_obj, "ref_text") or "")

        if ref_t is not None:
            orig_t = ref_t
            # Strip trailing isolated single-word Whisper silence hallucinations at clip boundary
            ref_t = re.sub(
                r'[\s,.;:!-]+\b(water|thank you|thanks for watching|subtitles|bye|you)\b[.!?\s]*$',
                '.',
                ref_t.strip(),
                flags=re.IGNORECASE
            ).strip()
            if ref_t and ref_t[-1] not in '.!?':
                ref_t += '.'
            ref_t = ref_t + ' '
            if isinstance(prompt_obj, dict):
                prompt_obj["ref_text"] = ref_t
            else:
                try:
                    setattr(prompt_obj, "ref_text", ref_t)
                except Exception:
                    pass
            print(f"Sanitized OmniVoice ref_text: '{orig_t}' -> '{ref_t}'")
    except Exception as e:
        print(f"OmniVoice prompt inspection note: {e}")
    return prompt_obj


def _trim_leading_prompt_bleed(audio_np, sr=24000):
    """
    Detects and removes any short (<220ms) isolated acoustic prompt tail bleed (e.g., an
    unwanted syllable like 'water' leaked from the reference voice prompt) that occurs at the
    very beginning of a generated segment and is separated from the main sentence by a brief pause.
    """
    if audio_np is None or len(audio_np) < int(0.8 * sr):
        return audio_np
    try:
        win = int(0.015 * sr)
        env = np.convolve(np.abs(audio_np), np.ones(win, dtype=np.float32) / float(win), mode='same')
        peak = float(np.max(env)) + 1e-7
        silence_thr = max(0.005, peak * 0.04)

        # Trim pure leading silence up to first active sound (keep 25ms natural attack)
        active = np.where(env > silence_thr)[0]
        if len(active) == 0:
            return audio_np
        first_sound = int(active[0])

        # Check if there is a short burst (<240ms) followed by a clear silence valley (>55ms)
        burst_max_end = first_sound + int(0.24 * sr)
        search_limit = min(len(audio_np) // 3, first_sound + int(0.38 * sr))
        if search_limit > burst_max_end:
            sub_env = env[first_sound:search_limit]
            quiet_mask = sub_env < silence_thr
            # Find contiguous quiet region of at least 55ms after initial short burst
            min_quiet_len = int(0.055 * sr)
            count = 0
            cut_rel = None
            for i, is_q in enumerate(quiet_mask):
                if is_q:
                    count += 1
                    if count >= min_quiet_len and i >= int(0.04 * sr):
                        cut_rel = i - (count // 2)
                        break
                else:
                    count = 0
            if cut_rel is not None:
                new_start = max(0, first_sound + cut_rel)
                audio_np = audio_np[new_start:]
    except Exception as e:
        print(f"Audio onset trim note: {e}")
    return audio_np


def resolve_engine_and_language(language_mode, explicit_engine=None, explicit_edge_voice=None, voice_style="normal"):
    """
    Maps language_mode into:
      tts_engine: 'omnivoice' | 'moss_tts' | 'edge_tts'
      target_lang: 'English' | 'Swahili'
      edge_voice: Microsoft Edge TTS ShortName
    """
    mode = (language_mode or "omnivoice_en").strip().lower()
    style = (voice_style or "normal").strip().lower()

    if explicit_engine in ("omnivoice", "moss_tts", "edge_tts"):
        engine = explicit_engine
    elif mode.startswith("moss"):
        engine = "moss_tts"
    elif mode.startswith("edge") or mode in ("swahili", "east_african_english", "luganda", "documentary_narrator"):
        engine = "edge_tts"
    else:
        engine = "omnivoice"

    if "sw" in mode or "kiswahili" in mode:
        target_lang = "Swahili"
    else:
        target_lang = "English"

    edge_voice = (explicit_edge_voice or "").strip()
    if edge_voice not in VALID_EDGE_VOICES:
        if target_lang == "Swahili":
            edge_voice = "sw-KE-RafikiNeural"
        elif "east_african" in mode:
            edge_voice = "en-KE-ChilembaNeural"
        elif style in ("documentary", "storyteller") or "documentary" in mode:
            edge_voice = "en-GB-RyanNeural"
        else:
            edge_voice = "en-US-AriaNeural"

    return engine, target_lang, edge_voice

async def _edge_tts_save_async(text, voice, out_mp3):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_mp3)

def generate_edge_tts_audio(text, output_wav, edge_voice):
    clean_text = sanitize_tts_spoken_text(text)
    raw_mp3 = f"{output_wav}.edge.mp3"

    def _run_edge_in_new_thread():
        asyncio.run(_edge_tts_save_async(clean_text, edge_voice, raw_mp3))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(_run_edge_in_new_thread).result()

    normalize_audio_to_wav(raw_mp3, output_wav)
    if os.path.exists(raw_mp3):
        os.remove(raw_mp3)

def generate_omnivoice_audio(text, output_wav, voice_clone_prompt=None, ref_voice_path=None, voice_ref_text=""):
    model = get_omnivoice_model()
    raw_wav = f"{output_wav}.omni.wav"
    clean_text = sanitize_tts_spoken_text(text)

    kwargs = {"text": " " + clean_text}
    if voice_clone_prompt is not None:
        kwargs["voice_clone_prompt"] = voice_clone_prompt
    elif ref_voice_path and os.path.exists(ref_voice_path):
        kwargs["ref_audio"] = ref_voice_path
        if voice_ref_text:
            clean_ref_t = voice_ref_text.strip()
            if clean_ref_t and clean_ref_t[-1] not in '.!?':
                clean_ref_t += '.'
            kwargs["ref_text"] = clean_ref_t + " "

    audio_list = model.generate(**kwargs)
    audio_data = audio_list[0]
    if hasattr(audio_data, "detach"):
        audio_data = audio_data.detach().cpu().numpy()
    audio_data = np.asarray(audio_data, dtype=np.float32).squeeze()
    audio_data = _trim_leading_prompt_bleed(audio_data, sr=24000)

    sf.write(raw_wav, audio_data, 24000)
    normalize_audio_to_wav(raw_wav, output_wav)
    if os.path.exists(raw_wav):
        os.remove(raw_wav)

def generate_moss_tts_audio(text, output_wav, moss_ref_codes=None, ref_voice_path=None, target_lang="English"):
    processor, model = get_moss_tts_engine()
    moss_device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_wav = f"{output_wav}.moss.wav"
    clean_text = sanitize_tts_spoken_text(text)

    reference_arg = None
    if moss_ref_codes is not None:
        reference_arg = moss_ref_codes
    elif ref_voice_path and os.path.exists(ref_voice_path):
        reference_arg = [ref_voice_path]

    user_msg = processor.build_user_message(
        text=clean_text,
        reference=reference_arg,
        language=target_lang
    )
    batch = processor([[user_msg]], mode="generation")
    input_ids = batch["input_ids"].to(moss_device)
    attention_mask = batch["attention_mask"].to(moss_device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096
        )

    saved = False
    for message in processor.decode(outputs):
        if message is not None and getattr(message, "audio_codes_list", None):
            audio = message.audio_codes_list[0]
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            torchaudio.save(
                raw_wav,
                audio.detach().cpu().to(torch.float32),
                int(processor.model_config.sampling_rate)
            )
            saved = True
            break

    if not saved:
        raise Exception("MOSS-TTS did not produce decoded audio tokens for segment.")

    normalize_audio_to_wav(raw_wav, output_wav)
    if os.path.exists(raw_wav):
        os.remove(raw_wav)

def _synthesize_single_segment(
    text,
    output_wav,
    tts_engine,
    target_lang,
    edge_voice,
    omnivoice_prompt=None,
    moss_ref_codes=None,
    ref_voice_path=None,
    voice_ref_text=""
):
    clean_text = sanitize_tts_spoken_text(text)
    if not clean_text or clean_text == ".":
        run_ffmpeg(
            f"ffmpeg -y -f lavfi -i anullsrc=r=24000:cl=mono:d=0.2 -c:a pcm_s16le {output_wav}",
            desc="Generate short silence segment"
        )
        return

    if tts_engine == "edge_tts":
        generate_edge_tts_audio(clean_text, output_wav, edge_voice)
    elif tts_engine == "moss_tts":
        generate_moss_tts_audio(
            clean_text,
            output_wav,
            moss_ref_codes=moss_ref_codes,
            ref_voice_path=ref_voice_path,
            target_lang=target_lang
        )
    else:
        generate_omnivoice_audio(
            clean_text,
            output_wav,
            voice_clone_prompt=omnivoice_prompt,
            ref_voice_path=ref_voice_path,
            voice_ref_text=voice_ref_text
        )

def boost_and_normalize_speech_wav(audio_path, volume_val=1.3):
    """
    Normalizes synthesized teacher speech to YouTube Broadcast Standard loudness
    (-13.5 dBFS active speech RMS) and scales by the user's chosen Teacher Voice
    Volume multiplier (volume_val) with a smooth soft-knee studio limiter so
    speech is always loud, clear, warm, and free of digital clipping.
    """
    try:
        audio_np, sr = sf.read(audio_path, dtype='float32')
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)
        if len(audio_np) < 100:
            return

        # Remove DC offset
        audio_np = audio_np - float(np.mean(audio_np))

        # Measure RMS on active speech frames (ignoring silent pauses)
        abs_a = np.abs(audio_np)
        peak_orig = float(np.max(abs_a)) + 1e-7
        active_mask = abs_a > max(0.006, peak_orig * 0.08)
        if np.sum(active_mask) > 200:
            active_rms = float(np.sqrt(np.mean(audio_np[active_mask] ** 2)))
        else:
            active_rms = float(np.sqrt(np.mean(audio_np ** 2)))

        if active_rms < 1e-5:
            return

        # Base broadcast RMS at 100% (1.0x) is 0.22 (-13.1 dBFS), scaled by user's volume_val
        vol_mult = float(max(0.25, min(3.0, volume_val)))
        target_rms = 0.22 * vol_mult
        gain = target_rms / active_rms
        # Cap maximum raw gain boost to 18x (+25 dB) so silence noise floor isn't over-amplified
        gain = min(18.0, gain)

        boosted = audio_np * gain
        # Smooth studio soft-knee saturation/limiter (prevents harsh clipping while preserving loud punch)
        limited = 0.96 * np.tanh(boosted / 0.90)
        sf.write(audio_path, limited.astype(np.float32), sr, subtype='PCM_16')
    except Exception as norm_err:
        print(f"Speech loudness normalization note ({norm_err})")


def apply_vocal_modulation(audio_path, whisper_mode, speed_val, pitch_val, volume_val, workspace, voice_style="normal"):
    filters = []
    style = (voice_style or "normal").strip().lower()
    if whisper_mode or style == "whisper":
        filters.append("highpass=f=300,treble=g=10:f=3000,compand=attacks=0:decays=0.1:points=-90/-90|-40/-30|-10/-20|0/-20:gain=5")
    elif style == "documentary":
        # Rich, resonant BBC/NatGeo documentary chest warmth + clear upper presence + smooth broadcast compression
        filters.append("highpass=f=65,equalizer=f=160:t=q:w=1.1:g=3.2,equalizer=f=3100:t=q:w=1.3:g=2.0,compand=attacks=0.02:decays=0.18:points=-80/-80|-38/-26|-12/-10|0/-6:gain=2.2")
    elif style == "storyteller":
        # Warm, intimate storytelling acoustic presence
        filters.append("highpass=f=75,equalizer=f=185:t=q:w=1.2:g=2.5,equalizer=f=2800:t=q:w=1.4:g=1.8")
    elif style == "energetic":
        # Crisp, articulate broadcast punch
        filters.append("highpass=f=85,equalizer=f=2400:t=q:w=1.2:g=2.6,equalizer=f=4200:t=q:w=1.5:g=1.8")

    if abs(pitch_val - 1.0) >= 0.01:
        new_rate = int(24000 * pitch_val)
        tempo_comp = 1.0 / pitch_val
        filters.append(f"asetrate={new_rate},atempo={tempo_comp:.4f},aresample=24000")
    if abs(speed_val - 1.0) >= 0.01:
        filters.append(f"atempo={max(0.5, min(2.0, speed_val)):.4f}")

    if filters:
        filter_str = ",".join(filters)
        mod_temp = f"{workspace}/mod_{uuid.uuid4().hex[:6]}.wav"
        print(f"Applying vocal acoustic modulation ({filter_str}) to {audio_path}...")
        run_ffmpeg(
            f'ffmpeg -y -i {audio_path} -af "{filter_str}" -ac 1 -ar 24000 -c:a pcm_s16le {mod_temp}',
            desc="Vocal modulation filter"
        )
        shutil.copyfile(mod_temp, audio_path)
        if os.path.exists(mod_temp):
            try:
                os.remove(mod_temp)
            except Exception:
                pass

    # Always normalize teacher speech to broadcast loudness and apply user's Teacher Voice Volume
    boost_and_normalize_speech_wav(audio_path, volume_val=volume_val)


def synthesize_block_audio_with_pauses(
    text,
    output_wav,
    tts_engine,
    target_lang,
    edge_voice,
    omnivoice_prompt,
    moss_ref_codes,
    ref_voice_path,
    voice_ref_text,
    workspace,
    whisper_mode=False,
    speed_val=1.0,
    pitch_val=1.0,
    volume_val=1.3,
    voice_style="normal"
):
    # For documentary or storyteller styles, if the script has multiple sentences without explicit [pause] tags,
    # inject subtle natural dramatic micro-pauses (0.35s) after key transition sentences for authentic documentary cadence
    proc_text = str(text or "")
    style = (voice_style or "normal").strip().lower()
    if style in ("documentary", "storyteller") and "[pause" not in proc_text.lower():
        sents = re.split(r'(?<=[.!?])\s+', proc_text.strip())
        if len(sents) >= 3:
            merged = []
            for s_i, s in enumerate(sents):
                merged.append(s)
                if s_i == 0 or (s_i == len(sents) - 2 and len(sents) >= 4):
                    merged.append("[pause 0.35s]")
            proc_text = " ".join(merged)

    parts = re.split(r'\[pause\s+([0-9\.]+)\s*s?\]', proc_text, flags=re.IGNORECASE)
    if len(parts) <= 1:
        _synthesize_single_segment(
            proc_text.strip(), output_wav, tts_engine, target_lang, edge_voice,
            omnivoice_prompt, moss_ref_codes, ref_voice_path, voice_ref_text
        )
        apply_vocal_modulation(output_wav, whisper_mode, speed_val, pitch_val, volume_val, workspace, voice_style=style)
        return

    temp_clips = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            try:
                duration = max(0.1, min(5.0, float(part)))
            except ValueError:
                duration = 0.5
            silence_wav = f"{workspace}/silence_{idx}_{uuid.uuid4().hex[:6]}.wav"
            run_ffmpeg(
                f"ffmpeg -y -f lavfi -i anullsrc=r=24000:cl=mono:d={duration} -c:a pcm_s16le {silence_wav}",
                desc="Pause silence generation"
            )
            temp_clips.append(silence_wav)
        else:
            clean_text = part.strip()
            if clean_text:
                seg_wav = f"{workspace}/seg_{idx}_{uuid.uuid4().hex[:6]}.wav"
                _synthesize_single_segment(
                    clean_text, seg_wav, tts_engine, target_lang, edge_voice,
                    omnivoice_prompt, moss_ref_codes, ref_voice_path, voice_ref_text
                )
                temp_clips.append(seg_wav)

    if not temp_clips:
        _synthesize_single_segment(
            ".", output_wav, tts_engine, target_lang, edge_voice,
            omnivoice_prompt, moss_ref_codes, ref_voice_path, voice_ref_text
        )
        return
    elif len(temp_clips) == 1:
        shutil.copyfile(temp_clips[0], output_wav)
        for c in temp_clips:
            if os.path.exists(c):
                try: os.remove(c)
                except Exception: pass
        apply_vocal_modulation(output_wav, whisper_mode, speed_val, pitch_val, volume_val, workspace, voice_style=style)
        return

    seg_list = f"{workspace}/seg_concat_{uuid.uuid4().hex[:6]}.txt"
    with open(seg_list, "w") as f:
        for c in temp_clips:
            f.write(f"file '{c}'\n")
    run_ffmpeg(
        f"ffmpeg -y -f concat -safe 0 -i {seg_list} -c:a pcm_s16le {output_wav}",
        desc="Segment pause concatenation"
    )
    for c in temp_clips + [seg_list]:
        if os.path.exists(c):
            try:
                os.remove(c)
            except Exception:
                pass
    apply_vocal_modulation(output_wav, whisper_mode, speed_val, pitch_val, volume_val, workspace, voice_style=style)


def get_background_soundtrack(bg_preset, workspace):
    """
    Synthesizes 10 distinct, rich polyphonic musical background soundtracks (chord progressions,
    plucked arpeggios, warm basslines, harmonic pads, and genre-specific rhythm pulses) at a
    balanced broadcast level (-13 dB RMS):
      1. lofi_study          - Soft Study Lofi Beats (Calm & Focused)
      2. african_acoustic    - Warm East African Acoustic Guitar (Uplifting)
      3. corporate_tech      - Modern Corporate & Tech Motivational
      4. cinematic_inspiring - Cinematic Inspiring Orchestra (Documentary & History)
      5. classical_piano     - Gentle Classical Piano & Strings (Academic Lecture)
      6. gospel_worship      - Warm Gospel & Spiritual Pad (Religious & Inspirational)
      7. ambient_science     - Deep Ambient Space & Science Synth (STEM & Lab)
      8. upbeat_afrobeat     - Rhythmic Upbeat Afro-Fusion Groove (Energetic Youth)
      9. acoustic_folk       - Peaceful Acoustic Folk & Nature (Storytelling)
      10. jazz_lounge        - Smooth Velvet Jazz Lounge (Executive & Business)
    """
    if not bg_preset or bg_preset == 'none':
        return None
    asset_path = f"/app/assets/{bg_preset}.mp3"
    if not os.path.exists(asset_path):
        asset_path = f"assets/{bg_preset}.mp3"
    if os.path.exists(asset_path):
        return asset_path

    synth_bg = f"{workspace}/bg_synth_{bg_preset}.wav"
    print(f"Synthesizing polyphonic musical soundtrack for genre '{bg_preset}'...")

    sr = 24000
    duration_sec = 64.0
    n_samples = int(sr * duration_sec)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    genre_presets = {
        'lofi_study': (
            [
                (174.61, [220.00, 261.63, 329.63, 392.00]),  # Fmaj7
                (164.81, [196.00, 246.94, 293.66, 392.00]),  # Em7
                (146.83, [174.61, 220.00, 261.63, 329.63]),  # Dm7
                (130.81, [196.00, 246.94, 261.63, 329.63]),  # Cmaj7
            ],
            1.45, 0.22, 0.16, 6.5
        ),
        'african_acoustic': (
            [
                (196.00, [246.94, 293.66, 392.00, 493.88]),  # G major
                (130.81, [196.00, 261.63, 329.63, 523.25]),  # Cadd9
                (146.83, [220.00, 293.66, 369.99, 587.33]),  # D major
                (196.00, [246.94, 293.66, 392.00, 587.33]),  # G major
            ],
            2.0, 0.28, 0.18, 5.5
        ),
        'corporate_tech': (
            [
                (110.00, [220.00, 261.63, 329.63, 440.00]),  # Am
                (87.31,  [174.61, 220.00, 261.63, 349.23]),  # F
                (130.81, [196.00, 261.63, 329.63, 523.25]),  # C
                (98.00,  [196.00, 246.94, 293.66, 392.00]),  # G
            ],
            1.75, 0.24, 0.18, 6.8
        ),
        'cinematic_inspiring': (
            [
                (146.83, [220.00, 293.66, 349.23, 440.00]),  # Dm
                (116.54, [233.08, 293.66, 349.23, 466.16]),  # Bbmaj7
                (174.61, [261.63, 349.23, 440.00, 523.25]),  # F
                (130.81, [196.00, 261.63, 329.63, 392.00]),  # C
            ],
            1.20, 0.16, 0.12, 4.2
        ),
        'classical_piano': (
            [
                (130.81, [261.63, 329.63, 392.00, 523.25]),  # C
                (98.00,  [246.94, 293.66, 392.00, 493.88]),  # G/B
                (110.00, [220.00, 261.63, 329.63, 440.00]),  # Am
                (87.31,  [174.61, 261.63, 349.23, 440.00]),  # F
            ],
            0.0, 0.30, 0.0, 4.8
        ),
        'gospel_worship': (
            [
                (155.56, [233.08, 311.13, 392.00, 466.16]),  # Ebmaj7
                (116.54, [233.08, 293.66, 349.23, 440.00]),  # Bbadd9
                (130.81, [196.00, 261.63, 311.13, 392.00]),  # Cm7
                (103.83, [207.65, 261.63, 311.13, 415.30]),  # Abmaj7
            ],
            1.10, 0.14, 0.08, 3.6
        ),
        'ambient_science': (
            [
                (110.00, [220.00, 246.94, 329.63, 440.00]),  # Asus2
                (146.83, [220.00, 293.66, 329.63, 493.88]),  # Dsus2/6
                (98.00,  [196.00, 246.94, 293.66, 440.00]),  # Gadd9
                (130.81, [196.00, 261.63, 329.63, 493.88]),  # Cmaj7
            ],
            1.35, 0.20, 0.07, 5.2
        ),
        'upbeat_afrobeat': (
            [
                (174.61, [261.63, 349.23, 440.00, 523.25]),  # F
                (130.81, [261.63, 329.63, 392.00, 523.25]),  # C
                (146.83, [220.00, 293.66, 349.23, 440.00]),  # Dm7
                (116.54, [233.08, 293.66, 349.23, 466.16]),  # Bb
            ],
            2.15, 0.28, 0.22, 6.2
        ),
        'acoustic_folk': (
            [
                (146.83, [220.00, 293.66, 369.99, 440.00]),  # D
                (110.00, [220.00, 277.18, 329.63, 440.00]),  # A
                (123.47, [246.94, 293.66, 369.99, 493.88]),  # Bm7
                (98.00,  [196.00, 246.94, 293.66, 392.00]),  # G
            ],
            1.65, 0.27, 0.12, 5.4
        ),
        'jazz_lounge': (
            [
                (146.83, [220.00, 261.63, 329.63, 440.00]),  # Dm9
                (98.00,  [196.00, 246.94, 311.13, 369.99]),  # G13
                (130.81, [196.00, 246.94, 293.66, 392.00]),  # Cmaj9
                (110.00, [196.00, 233.08, 277.18, 349.23]),  # A7alt
            ],
            1.55, 0.20, 0.14, 5.8
        ),
    }

    chords, beat_hz, pluck_amp, kick_amp, pluck_decay = genre_presets.get(
        bg_preset, genre_presets['lofi_study']
    )

    mix = np.zeros(n_samples, dtype=np.float32)
    chord_len_sec = 4.0
    chord_samples = int(chord_len_sec * sr)
    num_chords = int(duration_sec // chord_len_sec)

    for c_idx in range(num_chords):
        s_i = c_idx * chord_samples
        e_i = min(n_samples, s_i + chord_samples)
        seg_len = e_i - s_i
        if seg_len <= 0:
            break
        lt = np.linspace(0.0, chord_len_sec, seg_len, endpoint=False, dtype=np.float32)
        bass_f, notes = chords[c_idx % len(chords)]

        pad_env = np.sin(np.pi * lt / chord_len_sec) ** 0.45
        chord_sig = 0.28 * np.sin(2.0 * np.pi * bass_f * lt) + 0.10 * np.sin(2.0 * np.pi * (bass_f * 2.0) * lt)
        for n_f in notes:
            chord_sig += (
                0.16 * np.sin(2.0 * np.pi * n_f * lt)
                + 0.05 * np.sin(2.0 * np.pi * (n_f * 2.0) * lt)
                + 0.02 * np.sin(2.0 * np.pi * (n_f * 3.0) * lt)
            )
        chord_sig *= pad_env

        arp_notes = notes + notes[::-1]
        note_dur = chord_len_sec / float(len(arp_notes))
        note_samps = int(note_dur * sr)
        for a_idx, a_freq in enumerate(arp_notes):
            as_i = a_idx * note_samps
            ae_i = min(seg_len, as_i + note_samps)
            if ae_i <= as_i:
                continue
            nt = np.linspace(0.0, (ae_i - as_i) / float(sr), ae_i - as_i, endpoint=False, dtype=np.float32)
            pluck_env = np.exp(-nt * pluck_decay) * (1.0 - np.exp(-nt * 120.0))
            pluck_wave = (
                pluck_amp * np.sin(2.0 * np.pi * a_freq * nt)
                + (pluck_amp * 0.42) * np.sin(2.0 * np.pi * (a_freq * 2.0) * nt)
                + (pluck_amp * 0.16) * np.sin(2.0 * np.pi * (a_freq * 3.0) * nt)
            ) * pluck_env
            chord_sig[as_i:ae_i] += pluck_wave

        mix[s_i:e_i] += chord_sig

    if beat_hz > 0.1 and kick_amp > 0.01:
        beat_phase = (t * beat_hz) % 1.0
        kick_env = np.exp(-beat_phase * 14.0)
        kick_freq = 55.0 + 65.0 * np.exp(-beat_phase * 28.0)
        kick_wave = kick_amp * np.sin(2.0 * np.pi * kick_freq * (beat_phase / beat_hz)) * kick_env
        mix += kick_wave

    peak = float(np.max(np.abs(mix))) + 1e-6
    mix = (mix / peak) * 0.55

    sf.write(synth_bg, mix, sr, subtype='PCM_16')
    return synth_bg


def generate_studio_sfx_track(synthesized_blocks, n_samples, sr=24000):
    """
    Synthesizes a subtle, broadcast-grade Studio Sound Effects (SFX) track at `sr` Hz:
      - Soft, airy transition 'whoosh' at block boundaries
      - Gentle harmonic marimba/bell 'pop/chime' cues when slide cards / bullet points reveal
    Kept deliberately subtle (-19 dB peak) so it enhances production value without ever
    competing with the Teacher's voice.
    """
    sfx = np.zeros(n_samples, dtype=np.float32)
    rng = np.random.default_rng(2026)

    for b_idx, item in enumerate(synthesized_blocks):
        start_s = float(item.get("start_sec", 0.0))
        dur_s = float(item.get("duration_sec", 20.0))

        # 1. Soft airy transition whoosh at the start of each block (except 0.0s where we play a gentle intro chime)
        if b_idx == 0:
            chime_len = int(0.45 * sr)
            if chime_len < n_samples:
                t_c = np.linspace(0.0, 0.45, chime_len, endpoint=False, dtype=np.float32)
                env_c = np.exp(-t_c * 7.5) * (1.0 - np.exp(-t_c * 90.0))
                wave_c = (
                    0.065 * np.sin(2.0 * np.pi * 523.25 * t_c)
                    + 0.050 * np.sin(2.0 * np.pi * 659.25 * t_c)
                    + 0.040 * np.sin(2.0 * np.pi * 783.99 * t_c)
                ) * env_c
                sfx[:chime_len] += wave_c
        else:
            w_start = max(0, int((start_s - 0.14) * sr))
            w_len = int(0.32 * sr)
            w_end = min(n_samples, w_start + w_len)
            if w_end > w_start + 32:
                seg_l = w_end - w_start
                t_w = np.linspace(0.0, seg_l / float(sr), seg_l, endpoint=False, dtype=np.float32)
                env_w = np.sin(np.pi * (t_w / (seg_l / float(sr)))) ** 1.6
                # Filtered soft noise + low-mid sweep
                noise = rng.standard_normal(seg_l).astype(np.float32)
                win_k = 24
                smooth_noise = np.convolve(noise, np.ones(win_k, dtype=np.float32) / float(win_k), mode="same")
                sweep = np.sin(2.0 * np.pi * (140.0 + 220.0 * (t_w / 0.32)) * t_w)
                sfx[w_start:w_end] += (0.055 * smooth_noise + 0.035 * sweep) * env_w

        # 2. Subtle harmonic 'pop/chime' cues at internal visual scene switches (28% and 56% of block)
        for frac, freq in ((0.28, 587.33), (0.56, 659.25)):
            if dur_s < 8.0:
                continue
            p_start = int((start_s + dur_s * frac) * sr)
            p_len = int(0.18 * sr)
            p_end = min(n_samples, p_start + p_len)
            if p_end > p_start + 16:
                seg_l = p_end - p_start
                t_p = np.linspace(0.0, seg_l / float(sr), seg_l, endpoint=False, dtype=np.float32)
                env_p = np.exp(-t_p * 18.0) * (1.0 - np.exp(-t_p * 140.0))
                pop_wave = (0.045 * np.sin(2.0 * np.pi * freq * t_p) + 0.022 * np.sin(2.0 * np.pi * (freq * 1.5) * t_p)) * env_p
                sfx[p_start:p_end] += pop_wave

    return sfx


def mix_voice_and_background_wav(
    voice_wav_path,
    bg_track_path,
    output_wav_path,
    bg_volume=0.25,
    synthesized_blocks=None,
    enable_sfx=True
):
    """
    Mixes the Teacher's narration WAV, Background Soundtrack WAV, and optional
    Studio Transition & Bullet SFX directly in floating-point PCM without FFmpeg's
    amix 1/2 volume attenuation!
    - Preserves 100% of the Teacher's boosted broadcast loudness.
    - Scales the background music linearly by the user's bg_volume slider (0.0 .. 1.0).
    - Applies gentle voice-activated ducking so the music never masks syllables.
    """
    voice_np, sr = sf.read(voice_wav_path, dtype='float32')
    if voice_np.ndim > 1:
        voice_np = np.mean(voice_np, axis=1)

    n_voice = len(voice_np)
    sfx_np = generate_studio_sfx_track(synthesized_blocks or [], n_voice, sr=sr) if enable_sfx else np.zeros(n_voice, dtype=np.float32)

    if not bg_track_path or not os.path.exists(bg_track_path) or float(bg_volume) <= 0.001:
        combined = 0.97 * np.tanh((voice_np + sfx_np) / 0.94)
        stereo = np.column_stack([combined, combined])
        sf.write(output_wav_path, stereo.astype(np.float32), sr, subtype='PCM_16')
        return output_wav_path

    # Ensure bg_track is 24kHz mono WAV
    bg_wav_norm = f"{output_wav_path}.bgnorm.wav"
    normalize_audio_to_wav(bg_track_path, bg_wav_norm)
    bg_np, _ = sf.read(bg_wav_norm, dtype='float32')
    if os.path.exists(bg_wav_norm):
        os.remove(bg_wav_norm)
    if bg_np.ndim > 1:
        bg_np = np.mean(bg_np, axis=1)

    if len(bg_np) < n_voice:
        repeats = int(np.ceil(n_voice / float(max(1, len(bg_np)))))
        bg_np = np.tile(bg_np, repeats)[:n_voice]
    else:
        bg_np = bg_np[:n_voice]

    # Smooth 1.5s fade-in and 2.5s fade-out on the background music
    fade_in = min(int(1.5 * sr), n_voice // 4)
    fade_out = min(int(2.5 * sr), n_voice // 4)
    if fade_in > 0:
        bg_np[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out > 0:
        bg_np[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)

    # Gentle sidechain ducking envelope: duck background music by 28% during active syllables
    win = max(1, int(0.08 * sr))
    voice_env = np.convolve(np.abs(voice_np), np.ones(win, dtype=np.float32) / float(win), mode='same')
    duck_factor = 1.0 - 0.28 * np.clip(voice_env / 0.18, 0.0, 1.0)

    # User's bg_volume slider (0.0 .. 1.0, default 0.25)
    bg_gain = float(max(0.0, min(1.5, bg_volume))) * 0.52
    mixed_mono = voice_np + (bg_np * bg_gain * duck_factor) + sfx_np

    # Transparent studio ceiling limiter so combined peaks never clip
    mixed_limited = 0.97 * np.tanh(mixed_mono / 0.94)
    stereo = np.column_stack([mixed_limited, mixed_limited])
    sf.write(output_wav_path, stereo.astype(np.float32), sr, subtype='PCM_16')
    return output_wav_path


def _extract_cf_account_id(explicit_account_id, r2_endpoint):
    if explicit_account_id and str(explicit_account_id).strip():
        return str(explicit_account_id).strip()
    if r2_endpoint:
        m = re.search(r'https?://([a-f0-9]{32})\.r2\.cloudflarestorage\.com', str(r2_endpoint), re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


def _extract_clean_search_keywords(raw_text, max_words=4):
    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "into", "your", "about",
        "lesson", "key", "concept", "understanding", "introduction", "overview",
        "clean", "educational", "illustration", "showing", "real", "world", "example",
        "diagram", "visual", "part", "section", "module", "summary", "how", "what", "why",
        "practical", "application", "case", "study"
    }
    words = re.findall(r'[A-Za-z]{3,}', str(raw_text or ""))
    keywords = []
    for w in words:
        wl = w.lower()
        if wl not in stopwords and wl not in [k.lower() for k in keywords]:
            keywords.append(w)
        if len(keywords) >= max_words:
            break
    return " ".join(keywords)


def _is_specific_subject_requiring_ai_video(lesson_title, slide_title, pexels_query, block=None):
    """
    Detects whether a lesson block depicts a specific religious/biblical scene, a scientific/chemistry
    process or apparatus, specific regional geography/mountains (e.g. Rwenzori, Elgon, Virunga, tectonic
    formations), specialized agriculture/livestock feed formulation, historical/documentary reconstructions,
    or when the LLM explicitly set `requires_ai_video = True`.
    """
    if isinstance(block, dict):
        req_flag = block.get("requires_ai_video")
        if req_flag is True or str(req_flag).strip().lower() in ("true", "1", "yes"):
            return True

    combined = f"{lesson_title} {slide_title} {pexels_query} {(block or {}).get('image_prompt', '')}".lower()
    specific_subject_keywords = (
        # Biblical / Religious / Miracles / Historical
        "jesus", "christ", "miracle", "walking on water", "walk on water", "sea of galilee",
        "biblical", "bible", "gospel", "moses", "resurrection", "crucifixion", "apostle",
        "disciple", "parable", "scripture", "prophet", "loaves", "blind", "healed", "red sea",
        "ark", "angel", "heaven", "salvation",
        # Chemistry / Physics / Biology Processes & Lab Apparatus
        "distillation", "condenser", "fractionating", "fractional", "titration", "electrolysis",
        "chromatography", "bunsen", "beaker", "flask", "evaporation", "precipitate",
        "mitosis", "meiosis", "photosynthesis", "chloroplast", "mitochondria", "dna replication",
        "ribosome", "enzyme", "krebs", "osmosis", "diffusion", "atomic", "electron", "covalent",
        "ionic", "periodic table", "chemical reaction", "hydrocarbon", "polymer", "magnetic field",
        "refraction", "diffraction", "thermodynamics", "circuit diagram",
        # Regional Geography / Mountains / Earth Science / Tectonics
        "rwenzori", "elgon", "muhabura", "virunga", "moroto", "kadam", "kilimanjaro",
        "rift valley", "horst", "block mountain", "volcanic plug", "caldera", "tectonic",
        "magma", "glaciation", "stratovolcano", "faulting",
        # Specialized Agriculture / Animal Feeds / Nutrition Formulation
        "feed formulation", "person square", "pearson square", "silage fermentation",
        "premix", "mycotoxin", "rumen", "digestive tract", "hydroponic fodder"
    )
    return any(kw in combined for kw in specific_subject_keywords)


def fetch_pexels_video_clips_for_block(
    pexels_api_key,
    pexels_query,
    slide_title,
    lesson_title,
    notes,
    workspace,
    block_idx,
    used_video_ids,
    count=2,
    strict_subject=False
):
    """
    Queries the Pexels Videos API (https://api.pexels.com/videos/search) for best-matching
    HD 16:9 landscape video clips (.mp4) for the current lesson block.
    Avoids reusing video IDs across blocks and filters out generic unrelated clips when
    `strict_subject` is True so AI Motion Video Clips can take over for specific scenes.
    """
    if not pexels_api_key or not str(pexels_api_key).strip():
        return []

    headers = {
        "Authorization": str(pexels_api_key).strip(),
        "User-Agent": HTTP_HEADERS["User-Agent"]
    }

    note_kw = _extract_clean_search_keywords(notes[0] if notes else "", max_words=3)
    candidate_queries = []
    raw_candidates = [
        pexels_query,
        f"{slide_title} {lesson_title}",
        f"{slide_title} {note_kw}",
        lesson_title,
        slide_title,
    ]
    for raw_q in raw_candidates:
        cleaned = _extract_clean_search_keywords(raw_q, max_words=4)
        if cleaned and cleaned.lower() not in [q.lower() for q in candidate_queries]:
            candidate_queries.append(cleaned)

    subject_tokens = [
        w.lower() for w in re.findall(r'[A-Za-z]{4,}', f"{pexels_query} {slide_title} {lesson_title}")
        if w.lower() not in ("lesson", "video", "introduction", "overview", "summary", "concept", "understanding")
    ]

    downloaded_paths = []
    api_url = "https://api.pexels.com/videos/search"

    for q in candidate_queries:
        if len(downloaded_paths) >= count:
            break
        params = {
            "query": q,
            "orientation": "landscape",
            "size": "medium",
            "per_page": 12
        }
        try:
            resp = requests.get(api_url, headers=headers, params=params, timeout=12)
            if resp.status_code != 200:
                print(f"Pexels Videos API HTTP {resp.status_code} for '{q}': {resp.text[:160]}")
                continue
            videos = (resp.json() or {}).get("videos") or []
            for v in videos:
                if len(downloaded_paths) >= count:
                    break
                vid_id = v.get("id")
                if not vid_id or vid_id in used_video_ids:
                    continue
                dur = float(v.get("duration") or 0)
                if dur < 3.0 or dur > 90.0:
                    continue

                # Verify URL slug relevance when strict subject matching is needed
                v_url_slug = str(v.get("url") or "").lower()
                if strict_subject and subject_tokens:
                    matches = sum(1 for tok in subject_tokens[:5] if tok in v_url_slug)
                    if matches == 0:
                        continue

                video_files = v.get("video_files") or []
                best_link = None
                best_score = -1e9
                for vf in video_files:
                    link = vf.get("link") or ""
                    ftype = (vf.get("file_type") or "").lower()
                    vw = int(vf.get("width") or 0)
                    vh = int(vf.get("height") or 0)
                    if not link or ("mp4" not in ftype and ".mp4" not in link.lower()):
                        continue
                    if vw < vh or vw < 640:
                        continue
                    score = -abs(vw - 1280)
                    if 1280 <= vw <= 1920:
                        score += 1000
                    if score > best_score:
                        best_score = score
                        best_link = link

                if not best_link:
                    continue

                out_mp4 = f"{workspace}/pexels_{block_idx}_{len(downloaded_paths) + 1}.mp4"
                print(f"Downloading Pexels HD Video Clip (id={vid_id}, query='{q}') -> {out_mp4}...")
                v_resp = requests.get(best_link, headers=HTTP_HEADERS, stream=True, timeout=25)
                if v_resp.status_code == 200:
                    with open(out_mp4, "wb") as f:
                        for chunk in v_resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                    if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 15000:
                        used_video_ids.add(vid_id)
                        downloaded_paths.append(out_mp4)
        except Exception as px_err:
            print(f"Pexels video search note for '{q}': {px_err}")
            continue

    return downloaded_paths


def synthesize_ai_subject_video_clip(
    cf_images,
    output_mp4,
    subject_text="",
    duration_sec=8.0,
    fps=24,
    reverse_motion=False,
    block_seed=0
):
    """
    Synthesizes a custom 24fps 1280x720 HD AI Motion Video Clip (.mp4) directly from the
    block's subject-accurate AI illustrations across diverse spheres:
      - Geography / Mountains / Nature (e.g., Mountains in Uganda, Rwenzori, Elgon, volcanoes):
        high-altitude mist/cloud drift + panoramic ridge sweep
      - Agriculture / Animal Feeds / Farming / Biology:
        warm golden sunbeams + organic pollen/dust drift + macro-to-wide reveal
      - Documentary / History / Heritage:
        slow cinema push-in + subtle archival vignette & film grain
      - Water / Sea / Miracle waves:
        animated lower-frame water ripple displacement + light shimmer
      - Chemistry / Distillation / Lab process:
        rising vapor/bubble particles + thermal wave
    Uses 6 distinct non-predictable camera trajectories seeded by `block_seed`.
    """
    valid_imgs = []
    for p in (cf_images or []):
        if p and os.path.exists(p):
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is not None and im.size > 0:
                valid_imgs.append(cv2.resize(im, (1280, 720), interpolation=cv2.INTER_LANCZOS4))
    if not valid_imgs:
        return None
    if reverse_motion and len(valid_imgs) > 1:
        valid_imgs = list(reversed(valid_imgs))

    target_w, target_h = 1280, 720
    total_frames = max(48, int(round(float(duration_sec) * fps)))
    n_scenes = len(valid_imgs)
    subj_lower = str(subject_text or "").lower()

    is_water_scene = any(k in subj_lower for k in ("water", "sea", "wave", "lake", "ocean", "river", "jesus", "miracle", "galilee", "boat", "storm", "fish", "irrigation"))
    is_chem_scene = any(k in subj_lower for k in ("distillation", "chemistry", "flask", "condenser", "boiling", "vapor", "steam", "beaker", "bunsen", "reaction", "liquid", "laboratory", "electrolysis", "titration"))
    is_mountain_geo = any(k in subj_lower for k in ("mountain", "rwenzori", "elgon", "virunga", "muhabura", "volcano", "peak", "glacier", "altitude", "valley", "rift", "tectonic", "terrain", "geography", "uganda", "highland", "hill", "forest"))
    is_agri_scene = any(k in subj_lower for k in ("agriculture", "animal feed", "livestock", "cattle", "poultry", "dairy", "crop", "maize", "soya", "pasture", "silage", "farm", "harvest", "soil", "nutrition", "fodder", "grazing"))
    is_doc_history = any(k in subj_lower for k in ("documentary", "history", "historical", "ancient", "civilization", "empire", "heritage", "colonial", "kingdom", "biography", "archive", "century"))

    # Precompute coordinate grids for water ripple or atmospheric mist
    grid_x, grid_y = np.meshgrid(np.arange(target_w, dtype=np.float32), np.arange(target_h, dtype=np.float32))

    rng = np.random.RandomState((42 + int(block_seed) * 19) if not reverse_motion else (137 + int(block_seed) * 23))
    num_particles = 32
    p_x = rng.uniform(0.08, 0.92, size=num_particles)
    p_y = rng.uniform(0.12, 0.92, size=num_particles)
    p_speed = rng.uniform(0.05, 0.15, size=num_particles)
    p_radius = rng.randint(3, 9, size=num_particles)

    temp_raw_mp4 = output_mp4 + ".raw.mp4"
    writer = cv2.VideoWriter(
        temp_raw_mp4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (target_w, target_h)
    )
    if not writer.isOpened():
        return None

    def _render_scene_camera(base_img, scene_idx, local_u):
        ease_u = 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, local_u)))
        mode = (scene_idx + int(block_seed) + (2 if reverse_motion else 0)) % 6
        if mode == 0:
            # Documentary slow push-in toward upper-center focal point
            zoom = 1.03 + 0.13 * ease_u
            cx = 0.48 + 0.05 * ease_u
            cy = 0.46 - 0.04 * ease_u
        elif mode == 1:
            # Wide pull-back reveal from detail to landscape panorama
            zoom = 1.17 - 0.12 * ease_u
            cx = 0.53 - 0.06 * ease_u
            cy = 0.48 + 0.04 * ease_u
        elif mode == 2:
            # Panoramic horizontal ridge/horizon tracking shot
            zoom = 1.12
            cx = 0.43 + 0.14 * ease_u
            cy = 0.49 + 0.02 * math.sin(ease_u * math.pi)
        elif mode == 3:
            # Ascending crane sweep (low-to-high elevation reveal, ideal for mountains & architecture)
            zoom = 1.08 + 0.06 * ease_u
            cx = 0.50 + 0.03 * math.cos(ease_u * math.pi)
            cy = 0.56 - 0.12 * ease_u
        elif mode == 4:
            # Diagonal discovery drift (top-left to bottom-right)
            zoom = 1.05 + 0.10 * ease_u
            cx = 0.45 + 0.10 * ease_u
            cy = 0.44 + 0.09 * ease_u
        else:
            # Smooth orbital arc
            zoom = 1.07 + 0.08 * math.sin(ease_u * math.pi)
            cx = 0.50 + 0.06 * math.sin(ease_u * math.pi * 1.2)
            cy = 0.50 - 0.05 * math.cos(ease_u * math.pi)

        crop_w = int(round(target_w / zoom))
        crop_h = int(round(target_h / zoom))
        x0 = max(0, min(target_w - crop_w, int(round(cx * target_w - crop_w * 0.5))))
        y0 = max(0, min(target_h - crop_h, int(round(cy * target_h - crop_h * 0.5))))
        cropped = base_img[y0:y0 + crop_h, x0:x0 + crop_w]
        return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Precompute subtle documentary vignette mask if documentary/history/mountain sphere
    vignette_mask = None
    if is_doc_history or is_mountain_geo:
        nx = (grid_x - target_w * 0.5) / (target_w * 0.5)
        ny = (grid_y - target_h * 0.5) / (target_h * 0.5)
        rad_sq = nx * nx + ny * ny
        vignette_mask = np.clip(1.0 - 0.22 * np.clip(rad_sq - 0.35, 0.0, 1.5), 0.72, 1.0)[..., None].astype(np.float32)

    for f_idx in range(total_frames):
        t_sec = f_idx / float(fps)
        pos = (f_idx / float(max(1, total_frames - 1))) * n_scenes
        s_idx = min(n_scenes - 1, int(pos))
        local_u = pos - s_idx

        frame = _render_scene_camera(valid_imgs[s_idx], s_idx, local_u)
        # Smooth cross-dissolve over the last 22% of each sub-scene
        if s_idx < n_scenes - 1 and local_u > 0.78:
            alpha = (local_u - 0.78) / 0.22
            next_frame = _render_scene_camera(valid_imgs[s_idx + 1], s_idx + 1, 0.0)
            frame = cv2.addWeighted(frame, 1.0 - alpha, next_frame, alpha, 0)

        # Sphere Motion Layer 1: Animated Water Waves
        if is_water_scene:
            water_top = int(target_h * 0.56)
            sub_gx = grid_x[water_top:, :]
            sub_gy = grid_y[water_top:, :]
            depth_factor = (sub_gy - water_top) / float(max(1, target_h - water_top))
            dx = (np.sin(sub_gy * 0.11 - t_sec * 4.8) * 5.5 + np.cos(sub_gx * 0.04 + t_sec * 3.2) * 2.5) * depth_factor
            dy = np.cos(sub_gx * 0.06 + sub_gy * 0.08 - t_sec * 4.0) * 2.8 * depth_factor
            map_x = np.clip(sub_gx + dx, 0, target_w - 1).astype(np.float32)
            map_y = np.clip(sub_gy + dy, water_top, target_h - 1).astype(np.float32)
            frame[water_top:, :] = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Sphere Motion Layer 2: Drifting High-Altitude Mountain Mist / Clouds (for Mountains/Geography)
        overlay = frame.copy()
        if is_mountain_geo:
            for m_i in range(4):
                cloud_x = int(((0.22 * m_i + t_sec * (0.035 + 0.01 * m_i)) % 1.25 - 0.12) * target_w)
                cloud_y = int((0.24 + 0.11 * m_i + 0.02 * math.sin(t_sec + m_i)) * target_h)
                cv2.ellipse(overlay, (cloud_x, cloud_y), (220, 48), 0, 0, 360, (245, 248, 255), -1, cv2.LINE_AA)
            frame = cv2.addWeighted(frame, 0.90, overlay, 0.10, 0)
            overlay = frame.copy()

        # Sphere Motion Layer 3: Particles (Chem vapor / Agri golden sunlight motes / Atmospheric dust)
        for p_i in range(num_particles):
            if is_mountain_geo or is_agri_scene:
                # Gently drifting horizontal/diagonal organic breeze particles
                cur_x = (p_x[p_i] + t_sec * p_speed[p_i] * 0.45) % 0.92 + 0.04
                cur_y = p_y[p_i] + 0.02 * math.sin(t_sec * 1.8 + p_i)
            else:
                cur_y = (p_y[p_i] - t_sec * p_speed[p_i]) % 0.85 + 0.08
                cur_x = p_x[p_i] + 0.015 * math.sin(t_sec * 2.4 + p_i)
            px = int(cur_x * target_w)
            py = int(cur_y * target_h)
            rad = int(p_radius[p_i])
            if is_chem_scene:
                cv2.circle(overlay, (px, py), rad, (245, 235, 210), -1, cv2.LINE_AA)
                cv2.circle(overlay, (px, py), rad + 2, (255, 255, 255), 1, cv2.LINE_AA)
            elif is_agri_scene:
                # Warm golden harvest sun-motes (BGR: 130, 215, 255)
                cv2.circle(overlay, (px, py), max(2, rad - 1), (130, 215, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(overlay, (px, py), max(2, rad - 2), (180, 235, 255), -1, cv2.LINE_AA)
        frame = cv2.addWeighted(frame, 0.86, overlay, 0.14, 0)

        if vignette_mask is not None:
            frame = (frame.astype(np.float32) * vignette_mask).astype(np.uint8)

        writer.write(frame)

    writer.release()
    try:
        run_ffmpeg(
            f"ffmpeg -y -i {temp_raw_mp4} -c:v libx264 -preset veryfast -pix_fmt yuv420p {output_mp4}",
            desc="Encode synthesized AI Subject Motion Video Clip"
        )
        if os.path.exists(temp_raw_mp4):
            os.remove(temp_raw_mp4)
    except Exception:
        if os.path.exists(temp_raw_mp4):
            shutil.move(temp_raw_mp4, output_mp4)

    return output_mp4 if (os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 5000) else None


def _save_and_frame_image_bytes(img_bytes, output_png, title_text="", subtitle_text=""):
    """
    Decodes raw image bytes (PNG/JPEG/WebP), crops/fits cleanly to 1280x720 (16:9),
    and saves as PNG. Returns True if valid.
    """
    try:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return False
        ih, iw = img.shape[:2]
        if ih < 64 or iw < 64:
            return False

        target_w, target_h = 1280, 720
        scale = max(target_w / float(iw), target_h / float(ih))
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        x0 = max(0, (nw - target_w) // 2)
        y0 = max(0, (nh - target_h) // 2)
        cropped = resized[y0:y0 + target_h, x0:x0 + target_w].copy()

        cv2.imwrite(output_png, cropped)
        return os.path.exists(output_png) and os.path.getsize(output_png) > 1000
    except Exception as e:
        print(f"Image decode warning: {e}")
        return False


def _fetch_pollinations_ai_illustration(prompt_text, title_text, lesson_title, output_png, variant=0):
    """
    Free, zero-key AI Flux image generator fallback (Pollinations.ai) so specific biblical,
    historical, or chemistry distillation prompts are always custom-rendered with 100% subject
    accuracy even when Cloudflare Workers AI credentials are omitted or rate-limited.
    """
    try:
        full_prompt = (
            f"Detailed educational illustration and cinematic high resolution visual showing: {prompt_text}. "
            f"Subject: {lesson_title} - {title_text}. Accurate anatomy and apparatus, vibrant studio lighting, 16:9 widescreen"
        )
        encoded = urllib.parse.quote(full_prompt[:450])
        seed = 1000 + int(variant) * 37
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1280&height=720&seed={seed}&nologo=true&model=flux"
        r = requests.get(url, headers=HTTP_HEADERS, timeout=22)
        if r.status_code == 200 and len(r.content) > 4000:
            if _save_and_frame_image_bytes(r.content, output_png, title_text=title_text):
                return output_png
    except Exception as pol_err:
        print(f"Pollinations AI fallback note: {pol_err}")
    return None


def _fetch_pexels_photo_illustration(prompt_text, title_text, lesson_title, pexels_api_key, output_png, variant=0):
    """
    Fetches a high-resolution 16:9 landscape photograph from the Pexels Photos API
    when Pexels API Key is available.
    """
    if not pexels_api_key or not str(pexels_api_key).strip():
        return None
    headers = {
        "Authorization": str(pexels_api_key).strip(),
        "User-Agent": HTTP_HEADERS["User-Agent"]
    }
    for raw_q in [f"{title_text} {lesson_title}", prompt_text, lesson_title]:
        q = _extract_clean_search_keywords(raw_q, max_words=4)
        if not q:
            continue
        try:
            r = requests.get(
                "https://api.pexels.com/v1/search",
                headers=headers,
                params={"query": q, "orientation": "landscape", "per_page": 10},
                timeout=10
            )
            if r.status_code != 200:
                continue
            photos = (r.json() or {}).get("photos") or []
            if not photos:
                continue
            chosen = photos[variant % len(photos)]
            src = (chosen.get("src") or {})
            img_url = src.get("large2x") or src.get("large") or src.get("landscape") or src.get("original")
            if not img_url:
                continue
            img_resp = requests.get(img_url, headers=HTTP_HEADERS, timeout=12)
            if img_resp.status_code == 200 and len(img_resp.content) > 2000:
                if _save_and_frame_image_bytes(img_resp.content, output_png, title_text=title_text):
                    return output_png
        except Exception as e:
            print(f"Pexels photo search note: {e}")
            continue
    return None


def _fetch_wikimedia_commons_illustration(prompt_text, title_text, lesson_title, output_png, variant=0):
    """
    Fetches a real, high-resolution educational photograph or scientific diagram
    from Wikimedia Commons API (no API key required) matching the lesson topic.
    """
    combined_queries = [
        f"{title_text} {lesson_title}",
        f"{lesson_title}",
        f"{prompt_text}",
    ]

    api_url = "https://commons.wikimedia.org/w/api.php"
    for raw_q in combined_queries:
        search_q = _extract_clean_search_keywords(raw_q, max_words=5)
        if not search_q:
            continue

        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {search_q}",
            "gsrnamespace": 6,
            "gsrlimit": 10,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 1280,
            "format": "json"
        }
        try:
            r = requests.get(api_url, params=params, headers=HTTP_HEADERS, timeout=10)
            if r.status_code != 200:
                continue
            pages = (r.json().get("query") or {}).get("pages") or {}
            candidates = []
            for p in pages.values():
                ii = (p.get("imageinfo") or [{}])[0]
                thumb = ii.get("thumburl") or ii.get("url")
                if thumb and any(thumb.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                    candidates.append(thumb)
            if not candidates:
                continue

            chosen_url = candidates[variant % len(candidates)]
            print(f"Downloading real educational visual from Wikimedia Commons for '{search_q}': {chosen_url}")
            img_resp = requests.get(chosen_url, headers=HTTP_HEADERS, timeout=12)
            if img_resp.status_code == 200 and len(img_resp.content) > 2000:
                if _save_and_frame_image_bytes(img_resp.content, output_png, title_text=title_text):
                    return output_png
        except Exception as wm_err:
            print(f"Wikimedia search note ({wm_err})")
            continue
    return None


def generate_cloudflare_ai_image(prompt_text, title_text, cf_account_id, cf_api_token, output_png, variant=0, lesson_title="", pexels_api_key=""):
    """
    5-Tier Subject-Accurate Contextual Image Generator:
      1. Cloudflare Workers AI Model Cascade (Flux-1-Schnell -> SDXL-Lightning -> SDXL-Base)
      2. Pollinations.ai Flux Custom Subject Generator (ensures exact scenes like Jesus walking
         on water or chemistry distillation apparatus are custom-illustrated even without CF keys)
      3. Pexels High-Res Landscape Photography (when Pexels API Key is configured)
      4. Real Wikimedia Commons High-Res Educational Photography/Diagrams
      5. Procedural Multi-Node Studio Vector Infographic
    """
    clean_prompt = str(prompt_text or title_text or "Educational lesson concept").strip()
    angle_desc = [
        "wide cinematic establishing view, clear subject focus",
        "detailed close-up action and step-by-step process view",
        "educational diagram and real-world demonstration perspective"
    ][int(variant) % 3]

    if cf_account_id and cf_api_token:
        cf_models = [
            ("@cf/black-forest-labs/flux-1-schnell", {"prompt": f"Subject-accurate educational illustration and cinematic high-resolution visual showing: {clean_prompt}. Topic: {lesson_title} - {title_text}. Perspective: {angle_desc}. Vibrant studio lighting, 16:9 widescreen.", "steps": 4}),
            ("@cf/bytedance/stable-diffusion-xl-lightning", {"prompt": f"Detailed educational visual of {clean_prompt}, {title_text}, {angle_desc}, high resolution, 16:9"}),
            ("@cf/stabilityai/stable-diffusion-xl-base-1.0", {"prompt": f"Educational textbook illustration of {clean_prompt}, {title_text}, {angle_desc}, clear visual, 16:9"}),
        ]
        headers = {
            "Authorization": f"Bearer {cf_api_token}",
            "Content-Type": "application/json"
        }
        for model_slug, payload_body in cf_models:
            url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/run/{model_slug}"
            try:
                print(f"Generating Cloudflare Workers AI ({model_slug}) illustration for '{clean_prompt[:60]}...'")
                resp = requests.post(url, headers=headers, json=payload_body, timeout=30)
                if resp.status_code == 200:
                    ctype = resp.headers.get("Content-Type", "")
                    raw_bytes = None
                    if "application/json" in ctype:
                        payload = resp.json()
                        b64_img = (payload.get("result") or {}).get("image")
                        if b64_img:
                            raw_bytes = base64.b64decode(b64_img)
                    elif len(resp.content) > 1000:
                        raw_bytes = resp.content

                    if raw_bytes and _save_and_frame_image_bytes(raw_bytes, output_png, title_text=title_text):
                        return output_png
                else:
                    print(f"Cloudflare Workers AI ({model_slug}) returned HTTP {resp.status_code}: {resp.text[:180]}")
            except Exception as cf_err:
                print(f"Cloudflare Workers AI ({model_slug}) note: {cf_err}")

    # Tier 2: Pollinations.ai Flux Custom Subject Generator
    pol_path = _fetch_pollinations_ai_illustration(
        prompt_text=f"{clean_prompt}, {angle_desc}",
        title_text=title_text,
        lesson_title=lesson_title,
        output_png=output_png,
        variant=variant
    )
    if pol_path and os.path.exists(pol_path):
        return pol_path

    # Tier 3: Pexels High-Res Photography (if Pexels key is available)
    if pexels_api_key:
        px_path = _fetch_pexels_photo_illustration(
            prompt_text=clean_prompt,
            title_text=title_text,
            lesson_title=lesson_title,
            pexels_api_key=pexels_api_key,
            output_png=output_png,
            variant=variant
        )
        if px_path and os.path.exists(px_path):
            return px_path

    # Tier 4: Real high-resolution educational photo/diagram from Wikimedia Commons
    wm_path = _fetch_wikimedia_commons_illustration(
        prompt_text=clean_prompt,
        title_text=title_text,
        lesson_title=lesson_title,
        output_png=output_png,
        variant=variant
    )
    if wm_path and os.path.exists(wm_path):
        return wm_path

    # Tier 5: Rich vector node diagram fallback
    return generate_fallback_illustration(clean_prompt, title_text, output_png, variant=variant)


def _format_ass_timestamp(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60.0
    return f"{h}:{m:02d}:{s:05.2f}"


def _split_script_into_lyric_phrases(spoken_text, max_words_per_phrase=8):
    clean = re.sub(r'\[pause\s+[0-9\.]+\s*s?\]', ' ', str(spoken_text or ''), flags=re.IGNORECASE)
    clean = re.sub(r'\s+', ' ', clean).strip()
    if not clean:
        return []

    raw_clauses = re.split(r'(?<=[\.\,\;\:\!\?])\s+', clean)
    phrases = []
    for clause in raw_clauses:
        words = clause.split()
        if not words:
            continue
        for i in range(0, len(words), max_words_per_phrase):
            chunk = " ".join(words[i:i + max_words_per_phrase]).strip()
            if chunk:
                phrases.append(chunk)
    return phrases


def _format_youtube_chapter_ts(seconds):
    seconds = max(0, int(round(float(seconds))))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_youtube_seo_package(lesson_title, synthesized_blocks, presenter_name="", channel_name=""):
    """
    Computes exact MM:SS YouTube Chapter Timestamps from the actual rendered block
    audio durations, plus a ready-to-paste YouTube SEO Description & Comma-Separated Tags.
    Always begins with 00:00 as required by YouTube automatic chapters.
    """
    chapters = []
    keywords_bag = []

    for idx, item in enumerate(synthesized_blocks):
        ts_str = "00:00" if idx == 0 else _format_youtube_chapter_ts(item.get("start_sec", 0.0))
        stitle = str(item.get("slide_title") or f"Section {idx + 1}").strip()
        chapters.append(f"{ts_str} - {stitle}")
        for w in re.findall(r'[A-Za-z]{4,}', stitle):
            wl = w.lower()
            if wl not in keywords_bag and wl not in ("section", "module", "slide", "lesson", "introduction", "conclusion"):
                keywords_bag.append(wl)

    chapters_text = "\n".join(chapters)
    ch_line = f"Presented by {presenter_name}" if presenter_name else (f"Published by {channel_name}" if channel_name else "Complete Interactive Video Masterclass")
    summary_bullets = []
    for item in synthesized_blocks[:6]:
        notes = item.get("notes") or []
        if notes:
            summary_bullets.append(f"• {notes[0]}")

    tags_list = [lesson_title, "Video Lesson", "Online Course", "Educational Masterclass"]
    if channel_name:
        tags_list.append(channel_name)
    for kw in keywords_bag[:12]:
        tags_list.append(kw.title())
    tags_csv = ", ".join(tags_list[:16])

    full_description = (
        f"🎓 {lesson_title.upper()}\n"
        f"{ch_line}\n\n"
        f"📌 KEY TAKEAWAYS IN THIS LESSON:\n"
        + ("\n".join(summary_bullets) if summary_bullets else f"• Comprehensive walkthrough of {lesson_title}")
        + "\n\n⏱️ YOUTUBE CHAPTER TIMESTAMPS:\n"
        + chapters_text
        + f"\n\n🔖 TAGS:\n{tags_csv}\n"
    )

    return {
        "title": lesson_title,
        "chapters_text": chapters_text,
        "full_description": full_description,
        "tags": tags_csv,
    }


def _build_active_karaoke_line_ass(safe_text, phrase_duration_sec):
    """
    Builds a word-by-word active-highlight ASS Karaoke line using `{\\kf<centiseconds>}` tags.
    In ASS karaoke:
      - `PrimaryColour` is the ACTIVE / SPOKEN highlight color (Vibrant Gold/Cyan)
      - `SecondaryColour` is the UNSPOKEN upcoming word color (Crisp White)
    Each word smoothly sweeps from Crisp White to Vibrant Gold/Cyan at the exact
    centisecond the teacher speaks it!
    """
    words = [w for w in safe_text.split() if w.strip()]
    if not words:
        return safe_text

    total_cs = max(20, int(round(phrase_duration_sec * 100.0)))
    char_weights = [max(2, len(re.sub(r'[^A-Za-z0-9]', '', w)) + (2 if w.endswith((',', '.', ';', '!', '?')) else 0)) for w in words]
    sum_w = float(max(1, sum(char_weights)))

    k_tokens = []
    allocated = 0
    for idx, w in enumerate(words):
        if idx == len(words) - 1:
            cs = max(8, total_cs - allocated)
        else:
            cs = max(8, int(round(total_cs * (char_weights[idx] / sum_w))))
            allocated += cs
        k_tokens.append(f"{{\\kf{cs}}}{w}")

    if len(k_tokens) > 6:
        mid = len(k_tokens) // 2
        return " ".join(k_tokens[:mid]) + "\\N" + " ".join(k_tokens[mid:])
    return " ".join(k_tokens)


def _format_srt_timestamp(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000.0))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def create_synced_srt_subtitles(synthesized_blocks, output_srt_path):
    """
    Generates a standard YouTube-ready SubRip (.srt) Closed Caption file across
    all blocks so viewers can toggle native YouTube [CC] captions on/off and
    YouTube indexes 100% of the spoken lesson keywords for SEO.
    """
    srt_entries = []
    cue_idx = 1
    for item in synthesized_blocks:
        blk_start = float(item.get("start_sec", 0.0))
        blk_dur = max(1.0, float(item.get("duration_sec", 20.0)))
        text = str(item.get("text") or "").strip()
        phrases = _split_script_into_lyric_phrases(text, max_words_per_phrase=8)
        if not phrases:
            continue
        weights = [max(1, len(p)) for p in phrases]
        total_w = float(sum(weights))
        curr_t = blk_start
        for i, ph in enumerate(phrases):
            dur = blk_dur * (weights[i] / total_w)
            st = curr_t
            et = (blk_start + blk_dur) if i == len(phrases) - 1 else (curr_t + dur)
            curr_t = et
            clean_ph = ph.replace("\n", " ").strip()
            if clean_ph:
                srt_entries.append(
                    f"{cue_idx}\n{_format_srt_timestamp(st)} --> {_format_srt_timestamp(et)}\n{clean_ph}\n\n"
                )
                cue_idx += 1

    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write("".join(srt_entries))
    return output_srt_path


def create_synced_lyrics_ass(
    spoken_text,
    audio_wav_path,
    output_sub_path,
    block_idx=0,
    total_blocks=1,
    aspect_ratio="16:9",
    caption_mode="full_karaoke"
):
    """
    Generates word-by-word Active Karaoke Highlighted captions (.ass) according to `caption_mode`:
      - 'full_karaoke' (Default): Displays active word-by-word karaoke captions throughout the entire
        video, anchored cleanly in the bottom safe zone so they never collide with slide headers.
      - 'smart_broadcast': Eliminates visual clutter by suppressing burned-in lyrics during PowerPoint
        Slide / Chart views and only showing captions during Studio / B-Roll segments.
      - 'slides_only': Suppresses all burned-in subtitles (relies on slides + .srt Closed Captions).
    Returns (output_sub_path, has_dialogue_events).
    """
    speech_onset_sec = 0.0
    try:
        wav_data, sr = sf.read(audio_wav_path, dtype='float32')
        if wav_data.ndim > 1:
            wav_data = np.mean(wav_data, axis=1)
        total_duration = max(1.0, len(wav_data) / float(max(1, sr)))
        abs_w = np.abs(wav_data)
        peak_w = float(np.max(abs_w)) if len(abs_w) > 0 else 0.0
        if peak_w > 0.02:
            active_idx = np.where(abs_w > (peak_w * 0.08))[0]
            if len(active_idx) > 0:
                speech_onset_sec = min(0.65, float(active_idx[0]) / float(sr))
    except Exception:
        total_duration = 25.0

    is_vertical = str(aspect_ratio or "16:9").strip() == "9:16"
    mode_clean = str(caption_mode or "full_karaoke").strip().lower()
    res_x, res_y = (1080, 1920) if is_vertical else (1920, 1080)
    phrases = _split_script_into_lyric_phrases(spoken_text, max_words_per_phrase=6 if is_vertical else 7)

    margin_v = 115 if is_vertical else 44
    ass_header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {res_x}\n"
        f"PlayResY: {res_y}\n"
        "WrapStyle: 1\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: LyricBottom,Liberation Sans,54,&H0015CCFA,&H00FFFFFF,&H00080C18,&HB4080E1A,-1,0,0,0,100,100,0.6,0,1,4.8,2.6,2,72,72,{margin_v},1\n"
        f"Style: LyricAlt,Liberation Sans,54,&H00F8BD38,&H00FFFFFF,&H00080C18,&HB4080E1A,-1,0,0,0,100,100,0.6,0,1,4.8,2.6,2,72,72,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    if not phrases or (mode_clean == "slides_only" and not is_vertical):
        with open(output_sub_path, "w", encoding="utf-8") as f:
            f.write(ass_header)
        return output_sub_path, False

    weights = [max(1, len(p)) for p in phrases]
    total_weight = float(sum(weights))
    usable_duration = max(0.5, total_duration - speech_onset_sec)

    events = []
    curr_t = speech_onset_sec
    for idx, phrase in enumerate(phrases):
        dur = usable_duration * (weights[idx] / total_weight)
        start_t = curr_t
        end_t = total_duration if idx == len(phrases) - 1 else (curr_t + dur)
        curr_t = end_t

        # In Smart Broadcast Mode on 16:9, suppress burned-in captions during PowerPoint Slide windows
        # so the viewer reads the slide bullets/charts without competing bottom lyrics!
        if mode_clean == "smart_broadcast" and not is_vertical:
            mid_frac = ((start_t + end_t) * 0.5) / max(1.0, total_duration)
            if block_idx == 0:
                is_slide_window = (mid_frac >= 0.78)
            elif block_idx == total_blocks - 1:
                is_slide_window = (mid_frac < 0.22)
            else:
                is_slide_window = (0.0 <= mid_frac < 0.28) or (0.56 <= mid_frac < 0.80)
            if is_slide_window:
                continue

        style_name = "LyricBottom" if (idx % 2 == 0) else "LyricAlt"
        safe_text = phrase.replace("{", "(").replace("}", ")").replace("\n", " ").strip()
        karaoke_text = _build_active_karaoke_line_ass(safe_text, max(0.3, end_t - start_t))

        events.append(
            f"Dialogue: 0,{_format_ass_timestamp(start_t)},{_format_ass_timestamp(end_t)},{style_name},,0,0,0,,{{\\fad(120,110)}}{karaoke_text}\n"
        )

    with open(output_sub_path, "w", encoding="utf-8") as f:
        f.write(ass_header + "".join(events))
    return output_sub_path, (len(events) > 0)


def burn_presentation_slides(input_video, subtitle_file, final_output):
    cmd = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-vf', f"subtitles={subtitle_file}",
        '-c:a', 'copy',
        '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
        final_output
    ]
    run_ffmpeg(cmd, desc="Burn synchronized active-karaoke captions")


def publish_video_to_youtube_cloud(
    video_path,
    thumbnail_path,
    srt_path,
    youtube_seo,
    youtube_oauth,
    privacy_status="public"
):
    """
    Streams the finished .mp4 video, 1280x720 .jpg Thumbnail, and .srt Closed Captions
    directly from the RunPod GPU worker over 10Gbps cloud fiber to YouTube Data API v3
    without requiring the user to download any files locally!
    Supports scheduled publishing via `youtube_oauth['publish_at']` (ISO 8601 UTC).
    """
    if not isinstance(youtube_oauth, dict):
        return None
    access_token = (youtube_oauth.get("access_token") or "").strip()
    refresh_token = (youtube_oauth.get("refresh_token") or "").strip()
    client_id = (youtube_oauth.get("client_id") or "").strip()
    client_secret = (youtube_oauth.get("client_secret") or "").strip()

    if not access_token and (refresh_token and client_id and client_secret):
        try:
            tok_resp = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=15
            )
            if tok_resp.status_code == 200:
                access_token = (tok_resp.json() or {}).get("access_token", "")
            else:
                print(f"YouTube OAuth refresh error HTTP {tok_resp.status_code}: {tok_resp.text[:200]}")
        except Exception as auth_e:
            print(f"YouTube OAuth token refresh exception: {auth_e}")

    if not access_token or not os.path.exists(video_path):
        return None

    title_str = str((youtube_seo or {}).get("title") or "AI Video Lesson").strip()[:98]
    desc_str = str((youtube_seo or {}).get("full_description") or "").strip()[:4900]
    raw_tags = str((youtube_seo or {}).get("tags") or "").split(",")
    tags_list = [t.strip() for t in raw_tags if t.strip()][:15]
    priv = str(privacy_status or youtube_oauth.get("privacy_status") or "public").strip().lower()
    publish_at = str(youtube_oauth.get("publish_at") or "").strip()

    if publish_at or priv == "scheduled":
        # YouTube Data API v3 requires privacyStatus='private' when status.publishAt is set
        priv = "private"
    elif priv not in ("public", "unlisted", "private"):
        priv = "public"

    status_obj = {
        "privacyStatus": priv,
        "selfDeclaredMadeForKids": False,
    }
    if publish_at:
        status_obj["publishAt"] = publish_at

    meta_body = {
        "snippet": {
            "title": title_str,
            "description": desc_str,
            "tags": tags_list,
            "categoryId": "27",  # Education
        },
        "status": status_obj,
    }

    try:
        file_size = os.path.getsize(video_path)
        init_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Length": str(file_size),
            "X-Upload-Content-Type": "video/mp4",
        }
        init_resp = requests.post(
            "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
            headers=init_headers,
            data=json.dumps(meta_body).encode("utf-8"),
            timeout=25
        )
        if init_resp.status_code not in (200, 201):
            print(f"YouTube resumable init failed HTTP {init_resp.status_code}: {init_resp.text[:250]}")
            return {"error": f"YouTube upload init HTTP {init_resp.status_code}: {init_resp.text[:180]}"}

        upload_url = init_resp.headers.get("Location")
        if not upload_url:
            return {"error": "YouTube API did not return a resumable Location header."}

        with open(video_path, "rb") as vf:
            up_resp = requests.put(
                upload_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "video/mp4",
                    "Content-Length": str(file_size),
                },
                data=vf,
                timeout=600
            )
        if up_resp.status_code not in (200, 201):
            print(f"YouTube video upload failed HTTP {up_resp.status_code}: {up_resp.text[:250]}")
            return {"error": f"YouTube video transfer HTTP {up_resp.status_code}"}

        vid_json = up_resp.json() or {}
        video_id = vid_json.get("id")
        if not video_id:
            return {"error": "YouTube upload completed without a video ID."}

        print(f"Successfully uploaded video to YouTube! Video ID: {video_id} (publish_at={publish_at or 'immediate'})")

        # 2. Upload Custom 1280x720 Thumbnail (.jpg)
        thumb_uploaded = False
        if thumbnail_path and os.path.exists(thumbnail_path):
            try:
                with open(thumbnail_path, "rb") as tf:
                    th_resp = requests.post(
                        f"https://www.googleapis.com/upload/youtube/v3/thumbnails/set?videoId={video_id}&uploadType=media",
                        headers={
                            "Authorization": f"Bearer {access_token}",
                            "Content-Type": "image/jpeg",
                        },
                        data=tf.read(),
                        timeout=45
                    )
                thumb_uploaded = (th_resp.status_code in (200, 201))
                if not thumb_uploaded:
                    print(f"YouTube thumbnail set note HTTP {th_resp.status_code}: {th_resp.text[:180]}")
            except Exception as th_e:
                print(f"YouTube thumbnail upload note: {th_e}")

        # 3. Upload .srt Closed Captions Track for YouTube [CC] & SEO
        captions_uploaded = False
        if srt_path and os.path.exists(srt_path) and os.path.getsize(srt_path) > 10:
            try:
                boundary = f"===============ailg_srt_{uuid.uuid4().hex}=="
                cap_meta = json.dumps({
                    "snippet": {
                        "videoId": video_id,
                        "language": "en",
                        "name": "English (Official Lesson Captions)",
                        "isDraft": False
                    }
                })
                with open(srt_path, "r", encoding="utf-8", errors="ignore") as sf_in:
                    srt_body = sf_in.read()
                multipart_body = (
                    f"--{boundary}\r\n"
                    "Content-Type: application/json; charset=UTF-8\r\n\r\n"
                    f"{cap_meta}\r\n"
                    f"--{boundary}\r\n"
                    "Content-Type: application/octet-stream\r\n\r\n"
                    f"{srt_body}\r\n"
                    f"--{boundary}--\r\n"
                ).encode("utf-8")
                cap_resp = requests.post(
                    "https://www.googleapis.com/upload/youtube/v3/captions?uploadType=multipart&part=snippet",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": f"multipart/related; boundary={boundary}",
                    },
                    data=multipart_body,
                    timeout=45
                )
                captions_uploaded = (cap_resp.status_code in (200, 201))
            except Exception as cap_e:
                print(f"YouTube captions upload note: {cap_e}")

        return {
            "video_id": video_id,
            "watch_url": f"https://youtu.be/{video_id}",
            "studio_url": f"https://studio.youtube.com/video/{video_id}/edit",
            "privacy_status": "scheduled" if publish_at else priv,
            "publish_at": publish_at,
            "thumbnail_set": thumb_uploaded,
            "captions_set": captions_uploaded,
        }
    except Exception as yt_err:
        print(f"YouTube direct publish exception: {yt_err}")
        return {"error": str(yt_err)}


import threading


def report_live_progress(job, s3_client, bucket_name, job_id, step, percent, time_left_sec=15, extra=None):
    """
    Dual-channel real-time progress reporter:
      1. Calls runpod.serverless.progress_update(job, payload)
      2. Asynchronously writes progress/{job_id}.json to Cloudflare R2 so WordPress
         Frontend & Backend dashboards receive sub-second progress updates even if
         RunPod's API gateway caches /status responses.
    """
    pct = max(5, min(100, int(percent)))
    rem = max(0, int(time_left_sec))
    payload = {
        "step": str(step),
        "percent": pct,
        "time_left_sec": rem,
        "updated_at": int(time.time())
    }
    if isinstance(extra, dict):
        payload.update(extra)

    print(f"[LIVE PROGRESS {pct}% | ~{rem}s left] {step}", flush=True)
    try:
        runpod.serverless.progress_update(job, payload)
    except Exception as rp_err:
        print(f"RunPod progress_update note: {rp_err}")

    if s3_client and bucket_name and job_id:
        def _push_r2():
            try:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=f"progress/{job_id}.json",
                    Body=json.dumps(payload).encode("utf-8"),
                    ContentType="application/json"
                )
            except Exception as s3_err:
                print(f"R2 progress push note: {s3_err}")

        t = threading.Thread(target=_push_r2, daemon=True)
        t.start()


def handler(job):
    _hot_sync_lipsync_engine_from_github()
    job_input = job.get('input', {})
    job_id = job.get('id', str(uuid.uuid4()))

    timeline_data = job_input.get('timeline_data') or {}
    lesson_title = (
        job_input.get('lesson_title')
        or timeline_data.get('lesson_title')
        or "Interactive Video Lesson"
    ).strip()
    visual_blueprint = (job_input.get('visual_blueprint') or 'standard_30s').strip()
    visual_template = (job_input.get('visual_template') or 'broadcast_studio').strip()
    presenter_name = (job_input.get('presenter_name') or timeline_data.get('presenter_name') or '').strip()
    channel_name = (job_input.get('channel_name') or timeline_data.get('channel_name') or '').strip()
    aspect_ratio = (job_input.get('aspect_ratio') or '16:9').strip()
    caption_mode = (job_input.get('caption_mode') or 'full_karaoke').strip()
    enable_sfx = (str(job_input.get('enable_sfx', 'yes')).strip().lower() != 'no')
    logo_url = (job_input.get('logo_url') or '').strip()
    logo_position = (job_input.get('logo_position') or 'top_right').strip()
    master_audio_url = (job_input.get('master_audio_url') or timeline_data.get('master_audio_url') or '').strip()
    youtube_auto_publish = (str(job_input.get('youtube_auto_publish', 'no')).strip().lower() == 'yes')
    youtube_oauth = job_input.get('youtube_oauth') if isinstance(job_input.get('youtube_oauth'), dict) else None

    voice_ref_url = job_input.get('voice_ref_url')
    avatar_url = job_input.get('avatar_url')
    language_mode = job_input.get('language_mode', 'omnivoice_en')
    explicit_engine = job_input.get('tts_engine')
    explicit_edge_voice = job_input.get('edge_voice')
    bg_music = job_input.get('bg_music', 'none')
    try: bg_volume = float(job_input.get('bg_volume', 0.25))
    except: bg_volume = 0.25
    global_whisper = (job_input.get('whisper_mode', 'no') == 'yes')
    voice_style = (job_input.get('voice_style') or ('whisper' if global_whisper else 'normal')).strip().lower()
    try: global_speed = float(job_input.get('speaking_speed', 1.0))
    except: global_speed = 1.0
    try: global_pitch = float(job_input.get('voice_pitch', 1.0))
    except: global_pitch = 1.0
    try: global_volume = float(job_input.get('voice_volume', 1.3))
    except: global_volume = 1.3
    output_format = job_input.get('output_format', 'video')

    tts_engine, target_lang, edge_voice = resolve_engine_and_language(
        language_mode,
        explicit_engine=explicit_engine,
        explicit_edge_voice=explicit_edge_voice,
        voice_style=voice_style
    )
    print(
        f"Selected TTS Engine: {tts_engine} | Language: {target_lang} | Edge Voice: {edge_voice} | "
        f"Voice Style: {voice_style} | Voice Volume: {global_volume:.2f}x | BG Music: {bg_music} ({bg_volume:.2f}) | "
        f"Blueprint: {visual_blueprint} | Studio Template: {visual_template} | Aspect: {aspect_ratio} | "
        f"Captions: {caption_mode} | SFX: {enable_sfx} | MasterAudio: {bool(master_audio_url)}"
    )

    # Resolve Cloudflare R2, Workers AI, and Pexels credentials
    r2_access_key = (job_input.get('r2_access_key') or DEFAULT_R2_ACCESS_KEY or "").strip()
    r2_secret_key = (job_input.get('r2_secret_key') or DEFAULT_R2_SECRET_KEY or "").strip()
    r2_endpoint = (job_input.get('r2_endpoint') or DEFAULT_R2_ENDPOINT_URL or "").strip().rstrip('/')
    bucket_name = (job_input.get('r2_bucket') or DEFAULT_BUCKET_NAME or "video-making").strip()

    cf_ai_token = (job_input.get('cf_ai_token') or DEFAULT_CF_AI_TOKEN or "").strip()
    cf_account_id = _extract_cf_account_id(job_input.get('cf_account_id') or DEFAULT_CF_ACCOUNT_ID, r2_endpoint)
    pexels_api_key = (job_input.get('pexels_api_key') or DEFAULT_PEXELS_API_KEY or "").strip()

    s3_client = build_s3_client(r2_access_key, r2_secret_key, r2_endpoint)
    if not s3_client:
        raise Exception("Cloudflare R2 credentials (access_key, secret_key, endpoint) are missing in both job payload and RunPod environment variables.")

    workspace = f"/tmp/{job_id}"
    os.makedirs(workspace, exist_ok=True)

    try:
        blocks_list = timeline_data.get('timeline') if timeline_data and 'timeline' in timeline_data else []
        total_blocks = max(1, len(blocks_list))
        start_time = time.time()

        report_live_progress(
            job, s3_client, bucket_name, job_id,
            step=f"GPU Worker active: Initializing {'Audio-to-Video Slicer' if master_audio_url else tts_engine.upper()} & downloading media...",
            percent=8,
            time_left_sec=int(total_blocks * (8 if output_format == 'audio' else 22))
        )

        raw_avatar_path = ""
        if output_format == 'video' and avatar_url:
            # Preserve file extension (.mp4/.webm/.jpg/.png) so video avatar loops are recognized!
            av_ext = os.path.splitext(str(avatar_url).split('?')[0])[1].lower()
            if av_ext not in ('.mp4', '.webm', '.mov', '.mkv', '.avi', '.jpg', '.jpeg', '.png', '.webp'):
                av_ext = '.jpg'
            raw_avatar_path = download_file(avatar_url, f"{workspace}/avatar_orig{av_ext}", s3_client, r2_endpoint, bucket_name)

        raw_logo_path = None
        if output_format == 'video' and logo_url:
            try:
                lg_ext = os.path.splitext(str(logo_url).split('?')[0])[1].lower()
                if lg_ext not in ('.png', '.webp', '.jpg', '.jpeg', '.gif'):
                    lg_ext = '.png'
                raw_logo_path = download_file(logo_url, f"{workspace}/brand_logo{lg_ext}", s3_client, r2_endpoint, bucket_name)
            except Exception as lg_err:
                print(f"Brand logo download note: {lg_err}")
                raw_logo_path = None

        synthesized_blocks = []
        total_video_duration_sec = 0.0
        tts_phase_span = 76 if output_format == 'audio' else 28

        # =====================================================================
        # PASS 1A: If `master_audio_url` is provided (Audio-to-Video Mode),
        # slice the user's uploaded audio lesson directly across the blocks
        # instead of running TTS synthesis!
        # =====================================================================
        if master_audio_url:
            report_live_progress(
                job, s3_client, bucket_name, job_id,
                step="Downloading & normalizing uploaded Audio Lesson for video conversion...",
                percent=14,
                time_left_sec=int(total_blocks * 18)
            )
            raw_master_audio = download_file(master_audio_url, f"{workspace}/uploaded_lesson_raw", s3_client, r2_endpoint, bucket_name)
            norm_master_wav = f"{workspace}/uploaded_lesson_24k.wav"
            normalize_audio_to_wav(raw_master_audio, norm_master_wav)
            if os.path.exists(raw_master_audio):
                try: os.remove(raw_master_audio)
                except Exception: pass
            boost_and_normalize_speech_wav(norm_master_wav, volume_val=global_volume)

            master_np, master_sr = sf.read(norm_master_wav, dtype='float32')
            if os.path.exists(norm_master_wav):
                try: os.remove(norm_master_wav)
                except Exception: pass
            if master_np.ndim > 1:
                master_np = np.mean(master_np, axis=1)
            total_master_samples = len(master_np)
            total_master_sec = max(2.0, total_master_samples / float(max(1, master_sr)))

            # Compute proportional weights from each block's spoken_script length
            block_weights = []
            for b in blocks_list:
                txt_len = len(str(b.get('spoken_script') or '').split())
                block_weights.append(max(5, txt_len))
            sum_bw = float(max(1, sum(block_weights)))

            curr_sample = 0
            for idx_counter, block in enumerate(blocks_list):
                idx = block.get('block_index', idx_counter)
                text = str(block.get('spoken_script') or '').strip()
                notes = [str(n).strip() for n in (block.get('slide_notes') or []) if str(n).strip()]
                slide_title = (block.get('slide_title') or (notes[0][:48] if notes else f"Key Concept #{idx_counter + 1}")).strip()

                if idx_counter == len(blocks_list) - 1:
                    end_sample = total_master_samples
                else:
                    seg_samples = int(round(total_master_samples * (block_weights[idx_counter] / sum_bw)))
                    end_sample = min(total_master_samples, curr_sample + max(master_sr, seg_samples))

                slice_np = master_np[curr_sample:end_sample]
                if len(slice_np) < master_sr:
                    slice_np = np.pad(slice_np, (0, master_sr - len(slice_np)))
                curr_sample = end_sample

                output_wav = f"{workspace}/audio_{idx}.wav"
                sf.write(output_wav, slice_np.astype(np.float32), master_sr, subtype='PCM_16')
                blk_dur = max(1.0, len(slice_np) / float(master_sr))

                # If spoken_script is empty or placeholder, auto-transcribe the slice using Whisper on GPU
                if len(text.split()) < 3:
                    transcribed = _auto_transcribe_reference_audio(output_wav, max_words=120)
                    if transcribed:
                        text = transcribed
                        if not notes:
                            sents = [s.strip() for s in re.split(r'[\.\!\?]+', transcribed) if len(s.strip()) > 8]
                            notes = sents[:3] if sents else [transcribed[:65]]

                synthesized_blocks.append({
                    "idx_counter": idx_counter,
                    "idx": idx,
                    "block": block,
                    "text": text,
                    "notes": notes,
                    "slide_title": slide_title,
                    "audio_wav": output_wav,
                    "start_sec": total_video_duration_sec,
                    "duration_sec": blk_dur,
                })
                total_video_duration_sec += blk_dur

            del master_np
            gc.collect()
            print(f"Uploaded Audio Lesson ({total_master_sec:.1f}s) sliced into {len(synthesized_blocks)} video modules.")

        else:
            # =====================================================================
            # PASS 1B: Standard TTS Narration Synthesis across all blocks
            # =====================================================================
            voice_ref_text = job_input.get('voice_ref_text', '').strip()
            ref_voice_path = None
            omnivoice_prompt = None
            moss_ref_codes = None

            if tts_engine in ("omnivoice", "moss_tts"):
                if voice_ref_url:
                    raw_voice_path = download_file(voice_ref_url, f"{workspace}/ref_voice_orig", s3_client, r2_endpoint, bucket_name)
                    ref_voice_path = f"{workspace}/ref_voice.wav"
                    print("Preparing clean reference audio (smart silence-valley trim + 400ms silence pad to prevent word bleed)...")
                    prepare_clean_reference_voice_wav(raw_voice_path, ref_voice_path)
                    if os.path.exists(raw_voice_path):
                        try: os.remove(raw_voice_path)
                        except Exception: pass

                    report_live_progress(
                        job, s3_client, bucket_name, job_id,
                        step=f"Encoding Teacher Voice Clone Profile with {tts_engine.upper()}...",
                        percent=11,
                        time_left_sec=int(total_blocks * (7 if output_format == 'audio' else 18))
                    )

                    if tts_engine == "omnivoice":
                        model = get_omnivoice_model()
                        print("Creating reusable OmniVoice voice clone prompt...")
                        try:
                            if voice_ref_text:
                                clean_rt = voice_ref_text.strip()
                                if clean_rt and clean_rt[-1] not in '.!?':
                                    clean_rt += '.'
                                omnivoice_prompt = model.create_voice_clone_prompt(ref_audio=ref_voice_path, ref_text=clean_rt + " ")
                            else:
                                omnivoice_prompt = model.create_voice_clone_prompt(ref_audio=ref_voice_path)
                            omnivoice_prompt = sanitize_omnivoice_prompt_object(omnivoice_prompt)
                        except Exception as prompt_err:
                            print(f"OmniVoice prompt pre-encoding note ({prompt_err}); will pass ref_audio directly during generation.")
                            omnivoice_prompt = None
                    elif tts_engine == "moss_tts":
                        processor, _ = get_moss_tts_engine()
                        print("Encoding reference audio into reusable MOSS-TTS RVQ codes...")
                        try:
                            moss_ref_codes = processor.encode_audios_from_path([ref_voice_path])
                        except Exception as moss_err:
                            print(f"MOSS-TTS pre-encoding note ({moss_err}); will pass reference path directly.")
                            moss_ref_codes = None
                else:
                    print(f"No reference audio provided for {tts_engine}; using default/auto voice generation.")

            for idx_counter, block in enumerate(blocks_list):
                idx = block.get('block_index', idx_counter)
                text = block.get('spoken_script', '')
                notes = [str(n).strip() for n in (block.get('slide_notes') or []) if str(n).strip()]
                slide_title = (block.get('slide_title') or (notes[0][:48] if notes else f"Key Concept #{idx_counter + 1}")).strip()

                percent_start = int(12 + (idx_counter / max(1, total_blocks)) * tts_phase_span)
                report_live_progress(
                    job, s3_client, bucket_name, job_id,
                    step=f"Block {idx_counter+1}/{total_blocks}: Synthesizing {tts_engine.upper()} narration ({voice_style} style, {int(global_volume*100)}% boost)...",
                    percent=percent_start,
                    time_left_sec=max(8, int((total_blocks - idx_counter) * (6 if output_format == 'audio' else 17)))
                )

                output_wav = f"{workspace}/audio_{idx}.wav"
                delivery = (block.get('delivery_style') or 'normal').strip().lower()
                eff_style = delivery if delivery != 'normal' else voice_style
                blk_whisper = True if eff_style == 'whisper' else global_whisper
                if eff_style == 'slow':
                    blk_speed = global_speed * 0.84
                elif eff_style == 'fast':
                    blk_speed = global_speed * 1.18
                elif eff_style == 'documentary':
                    blk_speed = global_speed * 0.95 if abs(global_speed - 1.0) < 0.03 else global_speed
                else:
                    blk_speed = global_speed

                synthesize_block_audio_with_pauses(
                    text=text,
                    output_wav=output_wav,
                    tts_engine=tts_engine,
                    target_lang=target_lang,
                    edge_voice=edge_voice,
                    omnivoice_prompt=omnivoice_prompt,
                    moss_ref_codes=moss_ref_codes,
                    ref_voice_path=ref_voice_path,
                    voice_ref_text=voice_ref_text,
                    workspace=workspace,
                    whisper_mode=blk_whisper,
                    speed_val=blk_speed,
                    pitch_val=global_pitch,
                    volume_val=global_volume,
                    voice_style=eff_style
                )

                try:
                    blk_dur = max(1.0, float(sf.info(output_wav).duration))
                except Exception:
                    blk_dur = 30.0

                synthesized_blocks.append({
                    "idx_counter": idx_counter,
                    "idx": idx,
                    "block": block,
                    "text": text,
                    "notes": notes,
                    "slide_title": slide_title,
                    "audio_wav": output_wav,
                    "start_sec": total_video_duration_sec,
                    "duration_sec": blk_dur,
                })
                total_video_duration_sec += blk_dur

                percent_after_tts = int(12 + ((idx_counter + 1) / max(1, total_blocks)) * tts_phase_span)
                report_live_progress(
                    job, s3_client, bucket_name, job_id,
                    step=f"Block {idx_counter+1}/{total_blocks}: Narration audio ready ({blk_dur:.1f}s)...",
                    percent=percent_after_tts,
                    time_left_sec=max(6, int((total_blocks - idx_counter) * (5 if output_format == 'audio' else 15)))
                )

            print(f"All {len(synthesized_blocks)} audio blocks synthesized. Total lesson duration: {total_video_duration_sec:.2f}s")

        # Build YouTube Chapters, SEO Description & Tags package from exact block timestamps
        youtube_seo = build_youtube_seo_package(
            lesson_title=lesson_title,
            synthesized_blocks=synthesized_blocks,
            presenter_name=presenter_name,
            channel_name=channel_name
        )

        # Free TTS models from GPU VRAM before video lip-sync rendering
        if output_format == 'video':
            global _OMNIVOICE_MODEL, _MOSS_PROCESSOR, _MOSS_MODEL
            _OMNIVOICE_MODEL = None
            _MOSS_PROCESSOR = None
            _MOSS_MODEL = None
            _free_gpu_memory()

        # =====================================================================
        # PASS 2: Render Dynamic Multi-Scene Video Blocks (or collect audio)
        # =====================================================================
        clip_files = []
        used_pexels_video_ids = set()
        hero_thumbnail_bg_img = None
        render_start_time = time.time()
        vid_phase_start = 40
        vid_phase_span = 50.0  # 40% -> 90%

        for item in synthesized_blocks:
            idx_counter = item["idx_counter"]
            idx = item["idx"]
            block = item["block"]
            text = item["text"]
            notes = item["notes"]
            slide_title = item["slide_title"]
            output_wav = item["audio_wav"]
            blk_start_sec = item["start_sec"]
            chart_data = block.get("chart_data") if isinstance(block.get("chart_data"), dict) else None

            if output_format == 'audio':
                clip_files.append(output_wav)
                continue

            elapsed_render = time.time() - render_start_time
            sec_per_vid_blk = elapsed_render / idx_counter if idx_counter > 0 else 12.0
            time_left_blk = max(6, int((total_blocks - idx_counter) * sec_per_vid_blk))

            blk_base_pct = vid_phase_start + (idx_counter / float(max(1, total_blocks))) * vid_phase_span
            blk_span_pct = vid_phase_span / float(max(1, total_blocks))

            report_live_progress(
                job, s3_client, bucket_name, job_id,
                step=f"Block {idx_counter+1}/{total_blocks}: Fetching HD video clips & generating AI visuals in parallel...",
                percent=int(blk_base_pct + blk_span_pct * 0.08),
                time_left_sec=time_left_blk
            )

            pexels_query = (block.get('pexels_query') or f"{slide_title} {lesson_title}").strip()
            needs_ai_subject_video = _is_specific_subject_requiring_ai_video(
                lesson_title=lesson_title,
                slide_title=slide_title,
                pexels_query=pexels_query,
                block=block
            )

            base_img_prompt = (block.get('image_prompt') or f"{lesson_title} - {slide_title}: {notes[0] if notes else slide_title}").strip()
            prompt_1 = base_img_prompt
            prompt_2 = f"{base_img_prompt} — step-by-step action demonstration: {notes[1] if len(notes) > 1 else slide_title}".strip()
            prompt_3 = f"{base_img_prompt} — close-up detailed outcome: {notes[2] if len(notes) > 2 else (notes[-1] if notes else slide_title)}".strip()
            px_key_for_images = "" if needs_ai_subject_video else pexels_api_key

            # On long videos (>= 8 blocks), generate 2 distinct AI images concurrently alongside Pexels video search
            # (unless needs_ai_subject_video is True, where all 3 images are generated in parallel).
            # This cuts network I/O wait time per block by 65%!
            need_third_img = needs_ai_subject_video or (total_blocks < 8)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as media_pool:
                fut_pexels = media_pool.submit(
                    fetch_pexels_video_clips_for_block,
                    pexels_api_key=pexels_api_key,
                    pexels_query=pexels_query,
                    slide_title=slide_title,
                    lesson_title=lesson_title,
                    notes=notes,
                    workspace=workspace,
                    block_idx=idx,
                    used_video_ids=used_pexels_video_ids,
                    count=2,
                    strict_subject=needs_ai_subject_video
                )
                fut_img1 = media_pool.submit(
                    generate_cloudflare_ai_image,
                    prompt_text=prompt_1,
                    title_text=slide_title,
                    cf_account_id=cf_account_id,
                    cf_api_token=cf_ai_token,
                    output_png=f"{workspace}/cf_{idx}_1.png",
                    variant=idx_counter * 3,
                    lesson_title=lesson_title,
                    pexels_api_key=px_key_for_images
                )
                fut_img2 = media_pool.submit(
                    generate_cloudflare_ai_image,
                    prompt_text=prompt_2,
                    title_text=slide_title,
                    cf_account_id=cf_account_id,
                    cf_api_token=cf_ai_token,
                    output_png=f"{workspace}/cf_{idx}_2.png",
                    variant=idx_counter * 3 + 1,
                    lesson_title=lesson_title,
                    pexels_api_key=px_key_for_images
                )
                fut_img3 = media_pool.submit(
                    generate_cloudflare_ai_image,
                    prompt_text=prompt_3,
                    title_text=slide_title,
                    cf_account_id=cf_account_id,
                    cf_api_token=cf_ai_token,
                    output_png=f"{workspace}/cf_{idx}_3.png",
                    variant=idx_counter * 3 + 2,
                    lesson_title=lesson_title,
                    pexels_api_key=px_key_for_images
                ) if need_third_img else None

                pexels_clips = fut_pexels.result()
                cf_img_1 = fut_img1.result()
                cf_img_2 = fut_img2.result()
                cf_img_3 = fut_img3.result() if fut_img3 is not None else (cf_img_2 or cf_img_1)

            if hero_thumbnail_bg_img is None and cf_img_1 and os.path.exists(cf_img_1):
                hero_thumbnail_bg_img = f"{workspace}/hero_thumb_bg.png"
                try:
                    shutil.copyfile(cf_img_1, hero_thumbnail_bg_img)
                except Exception:
                    hero_thumbnail_bg_img = cf_img_1

            # 3. Synthesize custom 24fps HD AI Subject Motion Video Clips whenever the topic is a specific
            # subject scene or when Pexels cannot provide 2 matching HD video clips
            synthesized_ai_clips = []
            if needs_ai_subject_video or len(pexels_clips) < 2:
                subject_desc = f"{lesson_title} {slide_title} {base_img_prompt}"
                ai_clip_1 = synthesize_ai_subject_video_clip(
                    cf_images=[cf_img_1, cf_img_2, cf_img_3],
                    output_mp4=f"{workspace}/ai_subject_{idx}_1.mp4",
                    subject_text=subject_desc,
                    duration_sec=7.5 if total_blocks >= 8 else 8.0,
                    fps=24,
                    reverse_motion=False,
                    block_seed=idx_counter
                )
                ai_clip_2 = None
                if needs_ai_subject_video or len(pexels_clips) == 0:
                    ai_clip_2 = synthesize_ai_subject_video_clip(
                        cf_images=[cf_img_2, cf_img_3, cf_img_1],
                        output_mp4=f"{workspace}/ai_subject_{idx}_2.mp4",
                        subject_text=subject_desc,
                        duration_sec=7.5 if total_blocks >= 8 else 8.0,
                        fps=24,
                        reverse_motion=True,
                        block_seed=idx_counter + 1
                    )
                synthesized_ai_clips = [c for c in (ai_clip_1, ai_clip_2) if c and os.path.exists(c)]
                if needs_ai_subject_video and synthesized_ai_clips:
                    pexels_clips = (synthesized_ai_clips + pexels_clips)[:2]
                elif synthesized_ai_clips:
                    pexels_clips = (pexels_clips + synthesized_ai_clips)[:2]

            report_live_progress(
                job, s3_client, bucket_name, job_id,
                step=f"Block {idx_counter+1}/{total_blocks}: Running Wav2Lip-GAN neural lip-sync & multi-scene compositor...",
                percent=int(blk_base_pct + blk_span_pct * 0.28),
                time_left_sec=max(5, int(time_left_blk * 0.72))
            )

            def _make_block_progress_cb(b_idx_human, b_base, b_span, b_rem):
                def _cb(sub_frac, sub_msg):
                    sub_pct = int(b_base + b_span * (0.28 + float(sub_frac) * 0.68))
                    sub_rem = max(4, int(b_rem * (1.0 - float(sub_frac) * 0.75)))
                    report_live_progress(
                        job, s3_client, bucket_name, job_id,
                        step=f"Block {b_idx_human}/{total_blocks}: {sub_msg}",
                        percent=sub_pct,
                        time_left_sec=sub_rem
                    )
                return _cb

            raw_block_mp4 = f"{workspace}/block_raw_{idx}.mp4"
            render_dynamic_lesson_block(
                raw_avatar_path=raw_avatar_path,
                audio_wav_path=output_wav,
                output_mp4_path=raw_block_mp4,
                lesson_title=lesson_title,
                slide_title=slide_title,
                slide_notes=notes,
                cf_images=[cf_img_1, cf_img_2, cf_img_3],
                pexels_video_clips=pexels_clips,
                block_idx=idx_counter,
                total_blocks=total_blocks,
                block_start_sec=blk_start_sec,
                total_video_duration_sec=total_video_duration_sec,
                visual_blueprint=visual_blueprint,
                visual_template=visual_template,
                chart_data=chart_data,
                presenter_name=presenter_name,
                channel_name=channel_name,
                aspect_ratio=aspect_ratio,
                logo_path=raw_logo_path,
                logo_position=logo_position,
                fps=24,
                device=device,
                run_ffmpeg_fn=run_ffmpeg,
                progress_cb=_make_block_progress_cb(idx_counter + 1, blk_base_pct, blk_span_pct, time_left_blk)
            )

            # Burn captions according to Smart Broadcast Mode (skips FFmpeg burn-in when no events are needed!)
            lyrics_ass = f"{workspace}/lyrics_{idx}.ass"
            _, has_ass_events = create_synced_lyrics_ass(
                text,
                output_wav,
                lyrics_ass,
                block_idx=idx_counter,
                total_blocks=total_blocks,
                aspect_ratio=aspect_ratio,
                caption_mode=caption_mode
            )
            final_chunk = f"{workspace}/final_chunk_{idx}.mp4"
            if has_ass_events:
                burn_presentation_slides(raw_block_mp4, lyrics_ass, final_chunk)
                if os.path.exists(raw_block_mp4):
                    try: os.remove(raw_block_mp4)
                    except Exception: pass
            else:
                shutil.move(raw_block_mp4, final_chunk)

            # Immediately purge per-block temporary media files so /tmp never fills up on long 10-minute videos!
            for temp_f in [lyrics_ass, cf_img_1, cf_img_2, cf_img_3] + list(pexels_clips or []) + list(synthesized_ai_clips or []):
                if temp_f and temp_f != hero_thumbnail_bg_img and os.path.exists(temp_f):
                    try:
                        os.remove(temp_f)
                    except Exception:
                        pass
            gc.collect()

            clip_files.append(final_chunk)

        report_live_progress(
            job, s3_client, bucket_name, job_id,
            step="Mixing Teacher Voice, Studio Transition SFX & Background Soundtrack...",
            percent=92,
            time_left_sec=8
        )

        concat_list = f"{workspace}/concat.txt"
        with open(concat_list, "w") as f:
            for clip in clip_files:
                f.write(f"file '{clip}'\n")

        bg_track = get_background_soundtrack(bg_music, workspace)

        # Build concatenated teacher voice WAV so we can mix without FFmpeg's amix volume halving
        voice_concat_list = f"{workspace}/voice_concat.txt"
        with open(voice_concat_list, "w") as f:
            for item in synthesized_blocks:
                f.write(f"file '{item['audio_wav']}'\n")
        temp_voice_wav = f"{workspace}/temp_voice_master.wav"
        run_ffmpeg(f"ffmpeg -y -f concat -safe 0 -i {voice_concat_list} -c:a pcm_s16le {temp_voice_wav}", desc="Concat teacher voice WAV blocks")

        # Free individual block WAV files after concatenating temp_voice_wav
        for item in synthesized_blocks:
            w_p = item.get('audio_wav')
            if w_p and os.path.exists(w_p) and output_format != 'audio':
                try: os.remove(w_p)
                except Exception: pass

        mixed_master_wav = f"{workspace}/mixed_master.wav"
        mix_voice_and_background_wav(
            voice_wav_path=temp_voice_wav,
            bg_track_path=bg_track,
            output_wav_path=mixed_master_wav,
            bg_volume=bg_volume,
            synthesized_blocks=synthesized_blocks,
            enable_sfx=enable_sfx
        )
        if os.path.exists(temp_voice_wav):
            try: os.remove(temp_voice_wav)
            except Exception: pass

        # Build clean, human-readable filename from Lesson Title
        clean_slug = re.sub(r'[^A-Za-z0-9]+', '-', lesson_title).strip('-')[:65] or "AI-Lesson"
        ext = "mp3" if output_format == 'audio' else "mp4"
        download_filename = f"{clean_slug}-Lesson.{ext}"

        if output_format == 'audio':
            master_file = f"{workspace}/master_lesson.mp3"
            run_ffmpeg(
                f"ffmpeg -y -i {mixed_master_wav} -c:a libmp3lame -b:a 320k -ar 44100 -ac 2 {master_file}",
                desc="Encode mixed master audio to 320kbps MP3"
            )
            r2_key = f"outputs/{job_id}/{download_filename}"
            content_type = "audio/mpeg"
        else:
            master_file = f"{workspace}/master_lesson.mp4"
            temp_video = f"{workspace}/temp_video.mp4"
            run_ffmpeg(f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c copy {temp_video}", desc="Concat video blocks")
            # Remove individual block MP4 chunks right after temp_video is assembled to save /tmp space
            for ch_f in clip_files:
                if ch_f and os.path.exists(ch_f):
                    try: os.remove(ch_f)
                    except Exception: pass
            run_ffmpeg(
                f"ffmpeg -y -i {temp_video} -i {mixed_master_wav} -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -ar 44100 -ac 2 -shortest {master_file}",
                desc="Mux video with non-attenuated broadcast master audio"
            )
            for tmp_v in (temp_video, mixed_master_wav):
                if os.path.exists(tmp_v):
                    try: os.remove(tmp_v)
                    except Exception: pass
            r2_key = f"outputs/{job_id}/{download_filename}"
            content_type = "video/mp4"

        # Generate & Upload YouTube Closed Captions (.srt)
        srt_url = ""
        srt_filename = f"{clean_slug}-Captions.srt"
        srt_local = f"{workspace}/youtube_captions.srt"
        try:
            create_synced_srt_subtitles(synthesized_blocks, srt_local)
            if os.path.exists(srt_local) and os.path.getsize(srt_local) > 4:
                srt_r2_key = f"outputs/{job_id}/{srt_filename}"
                srt_disp = f'attachment; filename="{srt_filename}"'
                s3_client.upload_file(
                    srt_local,
                    bucket_name,
                    srt_r2_key,
                    ExtraArgs={
                        'ContentType': 'text/plain; charset=utf-8',
                        'ContentDisposition': srt_disp
                    }
                )
                srt_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': bucket_name,
                        'Key': srt_r2_key,
                        'ResponseContentDisposition': srt_disp,
                        'ResponseContentType': 'text/plain; charset=utf-8'
                    },
                    ExpiresIn=604800
                )
        except Exception as srt_err:
            print(f"SRT closed captions upload note: {srt_err}")

        # Generate & Upload Unique 1280x720 High-CTR YouTube Thumbnail (.jpg)
        thumbnail_url = ""
        thumbnail_filename = f"{clean_slug}-Thumbnail.jpg"
        thumb_local = f"{workspace}/youtube_thumbnail.jpg"
        try:
            generate_youtube_thumbnail(
                raw_avatar_path=raw_avatar_path,
                cf_image_path=hero_thumbnail_bg_img,
                lesson_title=lesson_title,
                presenter_name=presenter_name,
                channel_name=channel_name,
                visual_template=visual_template,
                output_jpg_path=thumb_local,
                logo_path=raw_logo_path,
                variant_seed=job_id
            )
            if os.path.exists(thumb_local):
                thumb_r2_key = f"outputs/{job_id}/{thumbnail_filename}"
                thumb_disp = f'attachment; filename="{thumbnail_filename}"'
                s3_client.upload_file(
                    thumb_local,
                    bucket_name,
                    thumb_r2_key,
                    ExtraArgs={
                        'ContentType': 'image/jpeg',
                        'ContentDisposition': thumb_disp
                    }
                )
                thumbnail_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': bucket_name,
                        'Key': thumb_r2_key,
                        'ResponseContentDisposition': thumb_disp,
                        'ResponseContentType': 'image/jpeg'
                    },
                    ExpiresIn=604800
                )
        except Exception as th_err:
            print(f"YouTube thumbnail generation/upload note: {th_err}")

        report_live_progress(
            job, s3_client, bucket_name, job_id,
            step=f"Uploading {download_filename} & YouTube SEO Package to Cloudflare R2...",
            percent=96,
            time_left_sec=4
        )

        content_disp = f'attachment; filename="{download_filename}"'
        print(f"Uploading {master_file} to R2 bucket {bucket_name} as {r2_key} (filename={download_filename})...")
        s3_client.upload_file(
            master_file,
            bucket_name,
            r2_key,
            ExtraArgs={
                'ContentType': content_type,
                'ContentDisposition': content_disp
            }
        )

        try:
            download_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': bucket_name,
                    'Key': r2_key,
                    'ResponseContentDisposition': content_disp,
                    'ResponseContentType': content_type
                },
                ExpiresIn=604800  # 7 days valid
            )
        except Exception as e:
            print(f"Presigned URL generation failed ({e}), falling back to endpoint URL...")
            download_url = f"{r2_endpoint}/{bucket_name}/{r2_key}"

        # Optional Direct Cloud-to-YouTube Publishing (when requested & authorized)
        youtube_publish_result = None
        if output_format == 'video' and youtube_auto_publish and youtube_oauth:
            report_live_progress(
                job, s3_client, bucket_name, job_id,
                step="Streaming Video, Thumbnail & Captions directly from Cloud to YouTube...",
                percent=98,
                time_left_sec=3
            )
            youtube_publish_result = publish_video_to_youtube_cloud(
                video_path=master_file,
                thumbnail_path=thumb_local if os.path.exists(thumb_local) else None,
                srt_path=srt_local if os.path.exists(srt_local) else None,
                youtube_seo=youtube_seo,
                youtube_oauth=youtube_oauth
            )

        report_live_progress(
            job, s3_client, bucket_name, job_id,
            step="Completed! Media, YouTube Thumbnail, Captions (.srt) & Chapters ready.",
            percent=100,
            time_left_sec=0,
            extra={
                "download_url": download_url,
                "filename": download_filename,
                "thumbnail_url": thumbnail_url,
                "thumbnail_filename": thumbnail_filename,
                "srt_url": srt_url,
                "srt_filename": srt_filename,
                "youtube_seo": youtube_seo,
                "youtube_publish": youtube_publish_result,
                "status": "COMPLETED"
            }
        )

        print(f"Job completed successfully! Returning output with download_url: {download_url} ({download_filename})")
        return {
            "status": "success",
            "type": output_format,
            "filename": download_filename,
            "download_url": download_url,
            "thumbnail_url": thumbnail_url,
            "thumbnail_filename": thumbnail_filename,
            "srt_url": srt_url,
            "srt_filename": srt_filename,
            "youtube_seo": youtube_seo,
            "youtube_publish": youtube_publish_result
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

runpod.serverless.start({"handler": handler})

