import os
import re
import gc
import shutil
import uuid
import json
import base64
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
from lipsync_engine import (
    render_dynamic_lesson_block,
    render_lipsync_teacher_block,
    render_slides_block,
    render_broll_motion_block,
    generate_fallback_illustration,
)

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

def resolve_engine_and_language(language_mode, explicit_engine=None, explicit_edge_voice=None):
    """
    Maps language_mode into:
      tts_engine: 'omnivoice' | 'moss_tts' | 'edge_tts'
      target_lang: 'English' | 'Swahili'
      edge_voice: Microsoft Edge TTS ShortName
    """
    mode = (language_mode or "omnivoice_en").strip().lower()

    if explicit_engine in ("omnivoice", "moss_tts", "edge_tts"):
        engine = explicit_engine
    elif mode.startswith("moss"):
        engine = "moss_tts"
    elif mode.startswith("edge") or mode in ("swahili", "east_african_english", "sunbird_english", "luganda"):
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
        else:
            edge_voice = "en-US-AriaNeural"

    return engine, target_lang, edge_voice

async def _edge_tts_save_async(text, voice, out_mp3):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_mp3)

def generate_edge_tts_audio(text, output_wav, edge_voice):
    raw_mp3 = f"{output_wav}.edge.mp3"

    def _run_edge_in_new_thread():
        asyncio.run(_edge_tts_save_async(text, edge_voice, raw_mp3))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(_run_edge_in_new_thread).result()

    normalize_audio_to_wav(raw_mp3, output_wav)
    if os.path.exists(raw_mp3):
        os.remove(raw_mp3)

def generate_omnivoice_audio(text, output_wav, voice_clone_prompt=None, ref_voice_path=None, voice_ref_text=""):
    model = get_omnivoice_model()
    raw_wav = f"{output_wav}.omni.wav"

    kwargs = {"text": text}
    if voice_clone_prompt is not None:
        kwargs["voice_clone_prompt"] = voice_clone_prompt
    elif ref_voice_path and os.path.exists(ref_voice_path):
        kwargs["ref_audio"] = ref_voice_path
        if voice_ref_text:
            kwargs["ref_text"] = voice_ref_text

    audio_list = model.generate(**kwargs)
    audio_data = audio_list[0]
    if hasattr(audio_data, "detach"):
        audio_data = audio_data.detach().cpu().numpy()
    sf.write(raw_wav, audio_data, 24000)
    normalize_audio_to_wav(raw_wav, output_wav)
    if os.path.exists(raw_wav):
        os.remove(raw_wav)

def generate_moss_tts_audio(text, output_wav, moss_ref_codes=None, ref_voice_path=None, target_lang="English"):
    processor, model = get_moss_tts_engine()
    moss_device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_wav = f"{output_wav}.moss.wav"

    reference_arg = None
    if moss_ref_codes is not None:
        reference_arg = moss_ref_codes
    elif ref_voice_path and os.path.exists(ref_voice_path):
        reference_arg = [ref_voice_path]

    user_msg = processor.build_user_message(
        text=text,
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
    if not text or not text.strip():
        run_ffmpeg(
            f"ffmpeg -y -f lavfi -i anullsrc=r=24000:cl=mono:d=0.2 -c:a pcm_s16le {output_wav}",
            desc="Generate short silence segment"
        )
        return

    if tts_engine == "edge_tts":
        generate_edge_tts_audio(text, output_wav, edge_voice)
    elif tts_engine == "moss_tts":
        generate_moss_tts_audio(
            text,
            output_wav,
            moss_ref_codes=moss_ref_codes,
            ref_voice_path=ref_voice_path,
            target_lang=target_lang
        )
    else:
        generate_omnivoice_audio(
            text,
            output_wav,
            voice_clone_prompt=omnivoice_prompt,
            ref_voice_path=ref_voice_path,
            voice_ref_text=voice_ref_text
        )

def apply_vocal_modulation(audio_path, whisper_mode, speed_val, pitch_val, volume_val, workspace):
    if not whisper_mode and abs(speed_val - 1.0) < 0.01 and abs(pitch_val - 1.0) < 0.01 and abs(volume_val - 1.0) < 0.01:
        return
    filters = []
    if whisper_mode:
        filters.append("highpass=f=300,treble=g=10:f=3000,compand=attacks=0:decays=0.1:points=-90/-90|-40/-30|-10/-20|0/-20:gain=5")
    if abs(pitch_val - 1.0) >= 0.01:
        new_rate = int(24000 * pitch_val)
        tempo_comp = 1.0 / pitch_val
        filters.append(f"asetrate={new_rate},atempo={tempo_comp:.4f},aresample=24000")
    if abs(speed_val - 1.0) >= 0.01:
        filters.append(f"atempo={speed_val:.4f}")
    if abs(volume_val - 1.0) >= 0.01:
        filters.append(f"volume={volume_val:.4f}")
    if not filters:
        return
    filter_str = ",".join(filters)
    mod_temp = f"{workspace}/mod_{uuid.uuid4().hex[:6]}.wav"
    print(f"Applying vocal acoustic modulation ({filter_str}) to {audio_path}...")
    run_ffmpeg(
        f'ffmpeg -y -i {audio_path} -af "{filter_str}" -ac 1 -ar 24000 -c:a pcm_s16le {mod_temp}',
        desc="Vocal modulation filter"
    )
    shutil.copyfile(mod_temp, audio_path)

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
    volume_val=1.0
):
    parts = re.split(r'\[pause\s+([0-9\.]+)\s*s?\]', text, flags=re.IGNORECASE)
    if len(parts) <= 1:
        _synthesize_single_segment(
            text.strip(), output_wav, tts_engine, target_lang, edge_voice,
            omnivoice_prompt, moss_ref_codes, ref_voice_path, voice_ref_text
        )
        apply_vocal_modulation(output_wav, whisper_mode, speed_val, pitch_val, volume_val, workspace)
        return

    temp_clips = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            try:
                duration = float(part)
            except ValueError:
                duration = 1.0
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
        apply_vocal_modulation(output_wav, whisper_mode, speed_val, pitch_val, volume_val, workspace)
        return

    seg_list = f"{workspace}/seg_concat_{uuid.uuid4().hex[:6]}.txt"
    with open(seg_list, "w") as f:
        for c in temp_clips:
            f.write(f"file '{c}'\n")
    run_ffmpeg(
        f"ffmpeg -y -f concat -safe 0 -i {seg_list} -c:a pcm_s16le {output_wav}",
        desc="Segment pause concatenation"
    )
    apply_vocal_modulation(output_wav, whisper_mode, speed_val, pitch_val, volume_val, workspace)


def get_background_soundtrack(bg_preset, workspace):
    """
    Synthesizes a rich, musical polyphonic studio background track (chord progressions,
    plucked arpeggios, warm bassline, and gentle rhythm pulse) at a clearly audible
    studio level (-15 dB RMS) so background music properly works across all lessons.
    """
    if not bg_preset or bg_preset == 'none':
        return None
    asset_path = f"/app/assets/{bg_preset}.mp3"
    if not os.path.exists(asset_path):
        asset_path = f"assets/{bg_preset}.mp3"
    if os.path.exists(asset_path):
        return asset_path

    synth_bg = f"{workspace}/bg_synth_{bg_preset}.wav"
    print(f"Synthesizing polyphonic musical soundtrack for mood '{bg_preset}'...")

    sr = 24000
    duration_sec = 64.0  # Seamless 16-bar loop at 120 BPM (or 8-bar loop at 60 BPM)
    n_samples = int(sr * duration_sec)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    if bg_preset == 'lofi_study':
        # Warm Rhodes Jazz progression: Fmaj7 -> Em7 -> Dm7 -> Cmaj7 (4s per chord)
        chords = [
            (174.61, [220.00, 261.63, 329.63, 392.00]),  # Fmaj7
            (164.81, [196.00, 246.94, 293.66, 392.00]),  # Em7
            (146.83, [174.61, 220.00, 261.63, 329.63]),  # Dm7
            (130.81, [196.00, 246.94, 261.63, 329.63]),  # Cmaj7
        ]
        beat_hz = 1.5  # 90 BPM relaxed lo-fi pulse
    elif bg_preset == 'african_acoustic':
        # Uplifting African Acoustic / Kalimba & Guitar progression: G -> C -> D -> G
        chords = [
            (196.00, [246.94, 293.66, 392.00, 493.88]),  # G major
            (130.81, [196.00, 261.63, 329.63, 523.25]),  # C major add9
            (146.83, [220.00, 293.66, 369.99, 587.33]),  # D major
            (196.00, [246.94, 293.66, 392.00, 587.33]),  # G major
        ]
        beat_hz = 2.0  # 120 BPM acoustic groove
    else:
        # 'corporate_tech' or default: Inspiring Ambient Pluck & Pad: Am -> F -> C -> G
        chords = [
            (110.00, [220.00, 261.63, 329.63, 440.00]),  # Am
            (87.31,  [174.61, 220.00, 261.63, 349.23]),  # F
            (130.81, [196.00, 261.63, 329.63, 523.25]),  # C
            (98.00,  [196.00, 246.94, 293.66, 392.00]),  # G
        ]
        beat_hz = 1.75  # 105 BPM modern corporate pulse

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

        # Smooth chord pad envelope
        pad_env = np.sin(np.pi * lt / chord_len_sec) ** 0.45
        chord_sig = 0.28 * np.sin(2.0 * np.pi * bass_f * lt) + 0.10 * np.sin(2.0 * np.pi * (bass_f * 2.0) * lt)
        for n_f in notes:
            # Warm electric piano / acoustic harmonics (fundamental + 2nd + gentle 3rd harmonic)
            chord_sig += (
                0.16 * np.sin(2.0 * np.pi * n_f * lt)
                + 0.05 * np.sin(2.0 * np.pi * (n_f * 2.0) * lt)
                + 0.02 * np.sin(2.0 * np.pi * (n_f * 3.0) * lt)
            )
        chord_sig *= pad_env

        # Melodic plucked arpeggio (8 notes per 4-second bar)
        arp_notes = notes + notes[::-1]
        note_dur = chord_len_sec / float(len(arp_notes))
        note_samps = int(note_dur * sr)
        for a_idx, a_freq in enumerate(arp_notes):
            as_i = a_idx * note_samps
            ae_i = min(seg_len, as_i + note_samps)
            if ae_i <= as_i:
                continue
            nt = np.linspace(0.0, (ae_i - as_i) / float(sr), ae_i - as_i, endpoint=False, dtype=np.float32)
            # Crisp pluck decay envelope
            pluck_env = np.exp(-nt * 6.5) * (1.0 - np.exp(-nt * 120.0))
            pluck_wave = (
                0.24 * np.sin(2.0 * np.pi * a_freq * nt)
                + 0.10 * np.sin(2.0 * np.pi * (a_freq * 2.0) * nt)
                + 0.04 * np.sin(2.0 * np.pi * (a_freq * 3.0) * nt)
            ) * pluck_env
            chord_sig[as_i:ae_i] += pluck_wave

        mix[s_i:e_i] += chord_sig

    # Add gentle soft kick/pulse beat so the track feels alive and rhythmic
    beat_phase = (t * beat_hz) % 1.0
    kick_env = np.exp(-beat_phase * 14.0)
    kick_freq = 55.0 + 65.0 * np.exp(-beat_phase * 28.0)
    kick_wave = 0.18 * np.sin(2.0 * np.pi * kick_freq * (beat_phase / beat_hz)) * kick_env
    mix += kick_wave

    # Normalize to a clean, warm studio backing level (peak 0.55, ~ -13 dB RMS)
    peak = float(np.max(np.abs(mix))) + 1e-6
    mix = (mix / peak) * 0.55

    sf.write(synth_bg, mix, sr, subtype='PCM_16')
    return synth_bg


def _extract_cf_account_id(explicit_account_id, r2_endpoint):
    if explicit_account_id and str(explicit_account_id).strip():
        return str(explicit_account_id).strip()
    if r2_endpoint:
        m = re.search(r'https?://([a-f0-9]{32})\.r2\.cloudflarestorage\.com', str(r2_endpoint), re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


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


def _fetch_wikimedia_commons_illustration(prompt_text, title_text, lesson_title, output_png, variant=0):
    """
    Fetches a real, high-resolution educational photograph or scientific diagram
    from Wikimedia Commons API (no API key required) matching the lesson topic.
    """
    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "into", "your", "about",
        "lesson", "key", "concept", "understanding", "introduction", "overview",
        "clean", "educational", "illustration", "showing", "real", "world", "example",
        "diagram", "visual", "part", "section", "module", "summary", "how", "what", "why"
    }
    combined_queries = [
        f"{title_text} {lesson_title}",
        f"{lesson_title}",
        f"{prompt_text}",
    ]

    api_url = "https://commons.wikimedia.org/w/api.php"
    for raw_q in combined_queries:
        words = re.findall(r'[A-Za-z]{3,}', str(raw_q))
        keywords = []
        for w in words:
            wl = w.lower()
            if wl not in stopwords and wl not in [k.lower() for k in keywords]:
                keywords.append(w)
            if len(keywords) >= 5:
                break
        if not keywords:
            continue

        search_q = " ".join(keywords)
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

            # Pick a distinct image based on variant index
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


def generate_cloudflare_ai_image(prompt_text, title_text, cf_account_id, cf_api_token, output_png, variant=0, lesson_title=""):
    """
    3-Tier Contextual Image Generator:
      1. Cloudflare Workers AI Model Cascade (Flux-1-Schnell -> SDXL-Lightning -> SDXL-Base)
      2. Real Wikimedia Commons High-Res Educational Photography/Diagrams (when CF token is not yet set or fails)
      3. Procedural Multi-Node Studio Vector Infographic
    """
    clean_prompt = str(prompt_text or title_text or "Educational lesson concept").strip()
    if cf_account_id and cf_api_token:
        cf_models = [
            ("@cf/black-forest-labs/flux-1-schnell", {"prompt": f"Clean educational textbook illustration and professional visual photograph showing: {clean_prompt}. Context: {lesson_title} - {title_text}. High resolution, vibrant studio lighting, clear subject focus, 16:9 widescreen.", "steps": 4}),
            ("@cf/bytedance/stable-diffusion-xl-lightning", {"prompt": f"Educational photography and visual diagram of {clean_prompt}, {title_text}, high resolution, 16:9"}),
            ("@cf/stabilityai/stable-diffusion-xl-base-1.0", {"prompt": f"Educational textbook illustration of {clean_prompt}, {title_text}, clear visual, 16:9"}),
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

    # Tier 2: Real high-resolution educational photo/diagram from Wikimedia Commons
    wm_path = _fetch_wikimedia_commons_illustration(
        prompt_text=clean_prompt,
        title_text=title_text,
        lesson_title=lesson_title,
        output_png=output_png,
        variant=variant
    )
    if wm_path and os.path.exists(wm_path):
        return wm_path

    # Tier 3: Rich vector node diagram fallback
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


def _highlight_lyric_keywords_ass(safe_text, phrase_idx=0):
    """
    Adds dynamic broadcast color highlights (Gold & Sky Cyan) to important content words
    inside an .ass subtitle line so the typography looks captivating and modern.
    """
    words = safe_text.split()
    if not words:
        return safe_text

    # Pick up to 2 longest content words (>= 6 letters) to highlight in Gold or Cyan
    accent_color = "&H0038BDF8&" if (phrase_idx % 2 == 0) else "&H0000D4FF&"
    scored = []
    for idx, w in enumerate(words):
        clean_w = re.sub(r'[^A-Za-z]', '', w)
        if len(clean_w) >= 6:
            scored.append((len(clean_w), idx))
    scored.sort(reverse=True)
    highlight_indices = {idx for _, idx in scored[:2]}

    out_words = []
    for idx, w in enumerate(words):
        if idx in highlight_indices:
            out_words.append(f"{{\\c{accent_color}}}{w}{{\\c&H00FFFFFF&}}")
        else:
            out_words.append(w)

    # Wrap into 2 balanced lines if more than 6 words
    if len(out_words) > 6:
        mid = len(out_words) // 2
        return " ".join(out_words[:mid]) + "\\N" + " ".join(out_words[mid:])
    return " ".join(out_words)


def create_synced_lyrics_ass(spoken_text, audio_wav_path, output_sub_path, block_idx=0):
    """
    Generates captivating, dynamic phrase-by-phrase synchronized on-screen lyrics (.ass)
    that alternate positions across Bottom, Right, Top, and Left screen zones after
    every 2 phrases, with enhanced Liberation Sans Bold typography and keyword color accents.
    """
    try:
        total_duration = max(1.0, float(sf.info(audio_wav_path).duration))
    except Exception:
        total_duration = 25.0

    phrases = _split_script_into_lyric_phrases(spoken_text, max_words_per_phrase=8)
    ass_header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "WrapStyle: 1\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: LyricBottom,Liberation Sans,42,&H00FFFFFF,&H0000D4FF,&H000F172A,&H96080E1A,-1,0,0,0,100,100,0.5,0,1,3.5,2,2,80,80,38,1\n"
        "Style: LyricTop,Liberation Sans,40,&H00FFFFFF,&H0038BDF8,&H000F172A,&H96080E1A,-1,0,0,0,100,100,0.5,0,1,3.5,2,8,80,80,82,1\n"
        "Style: LyricLeft,Liberation Sans,39,&H00FFFFFF,&H0000D4FF,&H000F172A,&H96080E1A,-1,0,0,0,100,100,0.5,0,1,3.5,2,1,68,580,128,1\n"
        "Style: LyricRight,Liberation Sans,39,&H00FFFFFF,&H0038BDF8,&H000F172A,&H96080E1A,-1,0,0,0,100,100,0.5,0,1,3.5,2,3,580,68,128,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    if not phrases:
        with open(output_sub_path, "w", encoding="utf-8") as f:
            f.write(ass_header)
        return output_sub_path

    # Cycle lyric position every 2 phrases: Bottom -> Right -> Top -> Left
    style_cycle = ["LyricBottom", "LyricRight", "LyricTop", "LyricLeft"]

    weights = [max(1, len(p)) for p in phrases]
    total_weight = float(sum(weights))

    events = []
    curr_t = 0.0
    for idx, phrase in enumerate(phrases):
        dur = total_duration * (weights[idx] / total_weight)
        start_t = curr_t
        end_t = total_duration if idx == len(phrases) - 1 else (curr_t + dur)
        curr_t = end_t

        style_name = style_cycle[((idx // 2) + block_idx) % len(style_cycle)]
        safe_text = phrase.replace("{", "(").replace("}", ")").replace("\n", " ").strip()
        styled_text = _highlight_lyric_keywords_ass(safe_text, phrase_idx=idx)

        events.append(
            f"Dialogue: 0,{_format_ass_timestamp(start_t)},{_format_ass_timestamp(end_t)},{style_name},,0,0,0,,{{\\fad(150,130)}}{styled_text}\n"
        )

    with open(output_sub_path, "w", encoding="utf-8") as f:
        f.write(ass_header + "".join(events))
    return output_sub_path


def burn_presentation_slides(input_video, subtitle_file, final_output):
    cmd = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-vf', f"subtitles={subtitle_file}",
        '-c:a', 'copy',
        '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
        final_output
    ]
    run_ffmpeg(cmd, desc="Burn synchronized on-screen lyrics")


def handler(job):
    job_input = job.get('input', {})
    job_id = job.get('id', str(uuid.uuid4()))

    timeline_data = job_input.get('timeline_data') or {}
    lesson_title = (
        job_input.get('lesson_title')
        or timeline_data.get('lesson_title')
        or "Interactive Video Lesson"
    ).strip()
    visual_blueprint = (job_input.get('visual_blueprint') or 'standard_30s').strip()

    voice_ref_url = job_input.get('voice_ref_url')
    avatar_url = job_input.get('avatar_url')
    language_mode = job_input.get('language_mode', 'omnivoice_en')
    explicit_engine = job_input.get('tts_engine')
    explicit_edge_voice = job_input.get('edge_voice')
    bg_music = job_input.get('bg_music', 'none')
    global_whisper = (job_input.get('whisper_mode', 'no') == 'yes')
    try: global_speed = float(job_input.get('speaking_speed', 1.0))
    except: global_speed = 1.0
    try: global_pitch = float(job_input.get('voice_pitch', 1.0))
    except: global_pitch = 1.0
    try: global_volume = float(job_input.get('voice_volume', 1.0))
    except: global_volume = 1.0
    output_format = job_input.get('output_format', 'video')

    tts_engine, target_lang, edge_voice = resolve_engine_and_language(
        language_mode,
        explicit_engine=explicit_engine,
        explicit_edge_voice=explicit_edge_voice
    )
    print(f"Selected TTS Engine: {tts_engine} | Language: {target_lang} | Edge Voice: {edge_voice} | Blueprint: {visual_blueprint}")

    # Resolve Cloudflare R2 & Workers AI credentials
    r2_access_key = (job_input.get('r2_access_key') or DEFAULT_R2_ACCESS_KEY or "").strip()
    r2_secret_key = (job_input.get('r2_secret_key') or DEFAULT_R2_SECRET_KEY or "").strip()
    r2_endpoint = (job_input.get('r2_endpoint') or DEFAULT_R2_ENDPOINT_URL or "").strip().rstrip('/')
    bucket_name = (job_input.get('r2_bucket') or DEFAULT_BUCKET_NAME or "video-making").strip()

    cf_ai_token = (job_input.get('cf_ai_token') or DEFAULT_CF_AI_TOKEN or "").strip()
    cf_account_id = _extract_cf_account_id(job_input.get('cf_account_id') or DEFAULT_CF_ACCOUNT_ID, r2_endpoint)

    s3_client = build_s3_client(r2_access_key, r2_secret_key, r2_endpoint)
    if not s3_client:
        raise Exception("Cloudflare R2 credentials (access_key, secret_key, endpoint) are missing in both job payload and RunPod environment variables.")

    workspace = f"/tmp/{job_id}"
    os.makedirs(workspace, exist_ok=True)

    try:
        blocks_list = timeline_data.get('timeline') if timeline_data and 'timeline' in timeline_data else []
        total_blocks = max(1, len(blocks_list))
        start_time = time.time()

        runpod.serverless.progress_update(job, {
            "step": f"Initializing {tts_engine.upper()} & preparing {'audio' if output_format == 'audio' else 'Broadcast Studio'} assets...",
            "percent": 5,
            "time_left_sec": int(total_blocks * (8 if output_format == 'audio' else 28))
        })

        voice_ref_text = job_input.get('voice_ref_text', '').strip()
        ref_voice_path = None
        omnivoice_prompt = None
        moss_ref_codes = None

        if tts_engine in ("omnivoice", "moss_tts"):
            if voice_ref_url:
                raw_voice_path = download_file(voice_ref_url, f"{workspace}/ref_voice_orig", s3_client, r2_endpoint, bucket_name)
                ref_voice_path = f"{workspace}/ref_voice.wav"
                print("Normalizing reference audio to 24kHz 16-bit PCM WAV (trimmed to 12s max for optimal cloning)...")
                normalize_audio_to_wav(raw_voice_path, ref_voice_path, max_duration_sec=12 if not voice_ref_text else None)

                if tts_engine == "omnivoice":
                    model = get_omnivoice_model()
                    print("Creating reusable OmniVoice voice clone prompt...")
                    try:
                        if voice_ref_text:
                            omnivoice_prompt = model.create_voice_clone_prompt(ref_audio=ref_voice_path, ref_text=voice_ref_text)
                        else:
                            omnivoice_prompt = model.create_voice_clone_prompt(ref_audio=ref_voice_path)
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

        raw_avatar_path = ""
        if output_format == 'video' and avatar_url:
            raw_avatar_path = download_file(avatar_url, f"{workspace}/avatar_orig", s3_client, r2_endpoint, bucket_name)

        # =====================================================================
        # PASS 1: Synthesize narration audio for all blocks first so we know
        # the exact total video duration (for rescaling the 10-min rules to ANY
        # duration) and can free TTS VRAM before running Wav2Lip-GAN + S3FD!
        # =====================================================================
        synthesized_blocks = []
        total_video_duration_sec = 0.0

        for idx_counter, block in enumerate(blocks_list):
            idx = block.get('block_index', idx_counter)
            text = block.get('spoken_script', '')
            notes = [str(n).strip() for n in (block.get('slide_notes') or []) if str(n).strip()]
            slide_title = (block.get('slide_title') or (notes[0][:48] if notes else f"Key Concept #{idx_counter + 1}")).strip()

            percent_done = int(8 + (idx_counter / max(1, total_blocks)) * 32)
            runpod.serverless.progress_update(job, {
                "step": f"Block {idx_counter+1}/{total_blocks}: Synthesizing {tts_engine.upper()} narration audio...",
                "percent": percent_done,
                "time_left_sec": max(10, int((total_blocks - idx_counter) * (7 if output_format == 'audio' else 22)))
            })

            output_wav = f"{workspace}/audio_{idx}.wav"
            delivery = block.get('delivery_style', 'normal')
            blk_whisper = True if delivery == 'whisper' else global_whisper
            blk_speed = global_speed * (0.8 if delivery == 'slow' else (1.2 if delivery == 'fast' else 1.0))

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
                volume_val=global_volume
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

        print(f"All {len(synthesized_blocks)} audio blocks synthesized. Total lesson duration: {total_video_duration_sec:.2f}s")

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
        render_start_time = time.time()

        for item in synthesized_blocks:
            idx_counter = item["idx_counter"]
            idx = item["idx"]
            block = item["block"]
            text = item["text"]
            notes = item["notes"]
            slide_title = item["slide_title"]
            output_wav = item["audio_wav"]
            blk_start_sec = item["start_sec"]

            if output_format == 'audio':
                clip_files.append(output_wav)
                continue

            elapsed_render = time.time() - render_start_time
            sec_per_vid_blk = elapsed_render / idx_counter if idx_counter > 0 else 16.0
            time_left = max(8, int((total_blocks - idx_counter) * sec_per_vid_blk))
            percent_done = int(42 + (idx_counter / max(1, total_blocks)) * 46)

            runpod.serverless.progress_update(job, {
                "step": f"Block {idx_counter+1}/{total_blocks}: Rendering Lip-Sync Teacher, Slides, B-Roll Clips & Visuals...",
                "percent": percent_done,
                "time_left_sec": time_left
            })

            # Generate 3 distinct contextual visuals per block so Slides and B-Roll Clips never repeat the same image
            prompt_1 = (block.get('image_prompt') or f"{lesson_title} - {slide_title}: {notes[0] if notes else slide_title}").strip()
            prompt_2 = f"{lesson_title} - {slide_title} practical application: {notes[1] if len(notes) > 1 else slide_title}".strip()
            prompt_3 = f"{lesson_title} - {slide_title} real-world case study: {notes[2] if len(notes) > 2 else (notes[-1] if notes else slide_title)}".strip()

            cf_img_1 = generate_cloudflare_ai_image(
                prompt_text=prompt_1,
                title_text=slide_title,
                cf_account_id=cf_account_id,
                cf_api_token=cf_ai_token,
                output_png=f"{workspace}/cf_{idx}_1.png",
                variant=idx_counter * 3,
                lesson_title=lesson_title
            )
            cf_img_2 = generate_cloudflare_ai_image(
                prompt_text=prompt_2,
                title_text=slide_title,
                cf_account_id=cf_account_id,
                cf_api_token=cf_ai_token,
                output_png=f"{workspace}/cf_{idx}_2.png",
                variant=idx_counter * 3 + 1,
                lesson_title=lesson_title
            )
            cf_img_3 = generate_cloudflare_ai_image(
                prompt_text=prompt_3,
                title_text=slide_title,
                cf_account_id=cf_account_id,
                cf_api_token=cf_ai_token,
                output_png=f"{workspace}/cf_{idx}_3.png",
                variant=idx_counter * 3 + 2,
                lesson_title=lesson_title
            )

            raw_block_mp4 = f"{workspace}/block_raw_{idx}.mp4"
            render_dynamic_lesson_block(
                raw_avatar_path=raw_avatar_path,
                audio_wav_path=output_wav,
                output_mp4_path=raw_block_mp4,
                lesson_title=lesson_title,
                slide_title=slide_title,
                slide_notes=notes,
                cf_images=[cf_img_1, cf_img_2, cf_img_3],
                block_idx=idx_counter,
                total_blocks=total_blocks,
                block_start_sec=blk_start_sec,
                total_video_duration_sec=total_video_duration_sec,
                visual_blueprint=visual_blueprint,
                fps=24,
                device=device,
                run_ffmpeg_fn=run_ffmpeg
            )

            # Burn dynamic alternating Top/Bottom/Left/Right synchronized lyrics
            lyrics_ass = f"{workspace}/lyrics_{idx}.ass"
            create_synced_lyrics_ass(text, output_wav, lyrics_ass, block_idx=idx_counter)
            final_chunk = f"{workspace}/final_chunk_{idx}.mp4"
            burn_presentation_slides(raw_block_mp4, lyrics_ass, final_chunk)

            clip_files.append(final_chunk)

        runpod.serverless.progress_update(job, {
            "step": "Mixing studio soundtrack & uploading master media to Cloudflare R2...",
            "percent": 92,
            "time_left_sec": 12
        })

        concat_list = f"{workspace}/concat.txt"
        with open(concat_list, "w") as f:
            for clip in clip_files:
                f.write(f"file '{clip}'\n")

        bg_track = get_background_soundtrack(bg_music, workspace)

        if output_format == 'audio':
            master_file = f"{workspace}/master_lesson.mp3"
            if bg_track:
                temp_voice = f"{workspace}/temp_voice.wav"
                run_ffmpeg(f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c:a pcm_s16le {temp_voice}", desc="Concat audio blocks")
                run_ffmpeg(
                    f'ffmpeg -y -i {temp_voice} -stream_loop -1 -i {bg_track} -filter_complex "[0:a]volume=1.15[v];[1:a]volume=0.35[bg];[v][bg]amix=inputs=2:duration=first:dropout_transition=2[out]" -map "[out]" -c:a libmp3lame -b:a 320k -ac 2 {master_file}',
                    desc="Mix audio with polyphonic background soundtrack"
                )
            else:
                run_ffmpeg(f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c:a libmp3lame -b:a 320k -ac 2 {master_file}", desc="Concat audio to MP3")
            r2_key = f"outputs/{job_id}_master.mp3"
            content_type = "audio/mpeg"
        else:
            master_file = f"{workspace}/master_lesson.mp4"
            if bg_track:
                temp_video = f"{workspace}/temp_video.mp4"
                run_ffmpeg(f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c copy {temp_video}", desc="Concat video blocks")
                run_ffmpeg(
                    f'ffmpeg -y -i {temp_video} -stream_loop -1 -i {bg_track} -filter_complex "[0:a]volume=1.15[v];[1:a]volume=0.35[bg];[v][bg]amix=inputs=2:duration=first:dropout_transition=2[out]" -map 0:v -map "[out]" -c:v copy -c:a aac -b:a 192k -ac 2 {master_file}',
                    desc="Mix video with polyphonic background soundtrack"
                )
            else:
                run_ffmpeg(f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c copy {master_file}", desc="Concat video blocks to MP4")
            r2_key = f"outputs/{job_id}_master.mp4"
            content_type = "video/mp4"

        print(f"Uploading {master_file} to R2 bucket {bucket_name} as {r2_key}...")
        s3_client.upload_file(
            master_file,
            bucket_name,
            r2_key,
            ExtraArgs={'ContentType': content_type}
        )

        try:
            download_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': r2_key},
                ExpiresIn=604800  # 7 days valid
            )
        except Exception as e:
            print(f"Presigned URL generation failed ({e}), falling back to endpoint URL...")
            download_url = f"{r2_endpoint}/{bucket_name}/{r2_key}"

        print(f"Job completed successfully! Returning output with download_url: {download_url}")
        return {
            "status": "success",
            "type": output_format,
            "download_url": download_url
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

runpod.serverless.start({"handler": handler})
