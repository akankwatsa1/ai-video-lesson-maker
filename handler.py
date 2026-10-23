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
    if not bg_preset or bg_preset == 'none':
        return None
    asset_path = f"/app/assets/{bg_preset}.mp3"
    if not os.path.exists(asset_path):
        asset_path = f"assets/{bg_preset}.mp3"
    if os.path.exists(asset_path):
        return asset_path

    synth_bg = f"{workspace}/bg_synth_{bg_preset}.wav"
    if bg_preset == 'lofi_study':
        freqs = "0.05*sin(2*PI*174*t)+0.04*sin(2*PI*220*t)+0.03*sin(2*PI*261.63*t)"
    elif bg_preset == 'african_acoustic':
        freqs = "0.06*sin(2*PI*196*t)+0.05*sin(2*PI*246.94*t)+0.04*sin(2*PI*293.66*t)"
    else:
        freqs = "0.05*sin(2*PI*220*t)+0.04*sin(2*PI*277.18*t)+0.04*sin(2*PI*329.63*t)"
    print(f"Generating ambient studio soundscape for mood '{bg_preset}'...")
    run_ffmpeg(
        f'ffmpeg -y -f lavfi -i aevalsrc="{freqs}":s=24000:d=600 -af "lowpass=f=800,volume=0.25" {synth_bg}',
        desc="Background ambient soundtrack synthesis"
    )
    return synth_bg

def _extract_cf_account_id(explicit_account_id, r2_endpoint):
    if explicit_account_id and str(explicit_account_id).strip():
        return str(explicit_account_id).strip()
    if r2_endpoint:
        m = re.search(r'https?://([a-f0-9]{32})\.r2\.cloudflarestorage\.com', str(r2_endpoint), re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


def generate_cloudflare_ai_image(prompt_text, title_text, cf_account_id, cf_api_token, output_png, variant=0):
    """
    Calls Cloudflare Workers AI (@cf/black-forest-labs/flux-1-schnell) to generate
    a contextual 16:9 illustration for the current lesson point.
    Automatically falls back to a crisp procedural infographic card if the token is
    not yet configured or if the API request fails.
    """
    clean_prompt = str(prompt_text or title_text or "Educational lesson concept").strip()
    if cf_account_id and cf_api_token:
        url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"
        headers = {
            "Authorization": f"Bearer {cf_api_token}",
            "Content-Type": "application/json"
        }
        full_prompt = (
            f"Clean educational textbook illustration and professional visual diagram showing: {clean_prompt}. "
            f"Lesson context: {title_text}. High resolution, vibrant studio colors, clear subject focus, no gibberish text, 16:9 widescreen."
        )
        try:
            print(f"Generating Cloudflare Workers AI illustration for '{clean_prompt[:60]}...'")
            resp = requests.post(url, headers=headers, json={"prompt": full_prompt, "steps": 4}, timeout=35)
            if resp.status_code == 200:
                ctype = resp.headers.get("Content-Type", "")
                if "application/json" in ctype:
                    payload = resp.json()
                    b64_img = (payload.get("result") or {}).get("image")
                    if b64_img:
                        with open(output_png, "wb") as f:
                            f.write(base64.b64decode(b64_img))
                elif len(resp.content) > 1000:
                    with open(output_png, "wb") as f:
                        f.write(resp.content)

                if os.path.exists(output_png) and os.path.getsize(output_png) > 500:
                    return output_png
            else:
                print(f"Cloudflare Workers AI returned HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as cf_err:
            print(f"Cloudflare Workers AI request warning ({cf_err}); using studio infographic fallback.")

    return generate_fallback_illustration(clean_prompt, title_text, output_png, variant=variant)


def _format_ass_timestamp(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60.0
    return f"{h}:{m:02d}:{s:05.2f}"


def _split_script_into_lyric_phrases(spoken_text, max_words_per_phrase=9):
    clean = re.sub(r'\[pause\s+[0-9\.]+\s*s?\]', ' ', str(spoken_text or ''), flags=re.IGNORECASE)
    clean = re.sub(r'\s+', ' ', clean).strip()
    if not clean:
        return []

    # First split on natural clause/sentence boundaries
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


def create_synced_lyrics_ass(spoken_text, audio_wav_path, output_sub_path):
    """
    Generates phrase-by-phrase synchronized on-screen lyrics (.ass) positioned inside
    the bottom lower-third lyric bar, timed proportionally to the exact audio duration.
    """
    try:
        total_duration = max(1.0, float(sf.info(audio_wav_path).duration))
    except Exception:
        total_duration = 25.0

    phrases = _split_script_into_lyric_phrases(spoken_text, max_words_per_phrase=9)
    ass_header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: LyricStyle,DejaVu Sans,36,&H00FFFFFF,&H0038BDF8,&H000F172A,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,90,90,46,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    if not phrases:
        with open(output_sub_path, "w", encoding="utf-8") as f:
            f.write(ass_header)
        return output_sub_path

    weights = [max(1, len(p)) for p in phrases]
    total_weight = float(sum(weights))

    events = []
    curr_t = 0.0
    for idx, phrase in enumerate(phrases):
        dur = total_duration * (weights[idx] / total_weight)
        start_t = curr_t
        end_t = total_duration if idx == len(phrases) - 1 else (curr_t + dur)
        curr_t = end_t

        safe_text = phrase.replace("{", "(").replace("}", ")").replace("\n", " ").strip()
        # Wrap long phrase cleanly into 2 balanced lines if > 54 chars
        if len(safe_text) > 54:
            words = safe_text.split()
            mid = len(words) // 2
            safe_text = " ".join(words[:mid]) + "\\N" + " ".join(words[mid:])

        events.append(
            f"Dialogue: 0,{_format_ass_timestamp(start_t)},{_format_ass_timestamp(end_t)},LyricStyle,,0,0,0,,{safe_text}\n"
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


def resolve_block_scene_type(block, idx_counter, total_blocks, visual_blueprint="standard_30s"):
    """
    Determines the visual layout for a 30-second block:
      - 'teacher_lipsync': AI Lip-Synced Teacher in Studio Frame + Right PowerPoint Card
      - 'slides': Full-Screen PowerPoint Presentation Slide Deck + Interchanging AI Illustrations
      - 'b_roll': Contextual 30s B-Roll Motion Clip (Ken Burns pan/zoom across 2 AI visuals)
    """
    raw_flag = (block.get("flag_type") or "teacher").strip().lower()
    if raw_flag in ("teacher_lipsync", "slides", "b_roll"):
        return raw_flag

    # Automatic Cost-Saving Pacing Blueprint when legacy 'teacher' flag is present:
    if idx_counter == 0 or idx_counter == total_blocks - 1:
        return "teacher_lipsync"

    if visual_blueprint == "90s_avatar":
        # 90s Avatar per 120s (3 out of 4 blocks lip-synced teacher, 1 block B-roll/Slides)
        if idx_counter % 4 == 2:
            return "b_roll"
        elif idx_counter % 4 == 1:
            return "slides"
        return "teacher_lipsync"

    # Default 'standard_30s': First & Last 30s Lip-Sync Teacher; 30s B-Roll per 120s; rest PowerPoint Slides
    if idx_counter % 4 == 2:
        return "b_roll"
    return "slides"


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
        total_blocks = len(timeline_data['timeline']) if timeline_data and 'timeline' in timeline_data else 1
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

        clip_files = []

        for idx_counter, block in enumerate(timeline_data['timeline']):
            idx = block.get('block_index', idx_counter)
            text = block.get('spoken_script', '')
            notes = [str(n).strip() for n in (block.get('slide_notes') or []) if str(n).strip()]
            slide_title = (block.get('slide_title') or (notes[0][:48] if notes else f"Key Concept #{idx_counter + 1}")).strip()
            scene_type = resolve_block_scene_type(block, idx_counter, total_blocks, visual_blueprint=visual_blueprint)

            elapsed = time.time() - start_time
            sec_per_block = elapsed / idx_counter if idx_counter > 0 else (8.0 if output_format == 'audio' else 25.0)
            remaining_blocks = total_blocks - idx_counter
            time_left = max(5, int(remaining_blocks * sec_per_block))
            percent_done = int(10 + (idx_counter / max(1, total_blocks)) * 75)

            runpod.serverless.progress_update(job, {
                "step": f"Block {idx_counter+1}/{total_blocks} ({scene_type.upper()}): Synthesizing {tts_engine.upper()} voice & studio visuals...",
                "percent": percent_done,
                "time_left_sec": time_left
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

            if output_format == 'audio':
                clip_files.append(output_wav)
                continue

            # Generate contextual illustrations via Cloudflare Workers AI (or crisp infographic fallback)
            base_prompt = (block.get('image_prompt') or f"{lesson_title} - {slide_title}: {'; '.join(notes[:2])}").strip()
            second_prompt = f"{lesson_title} - {slide_title} real-world example: {'; '.join(notes[1:3] if len(notes) > 1 else notes)}".strip()

            cf_img_1 = generate_cloudflare_ai_image(
                prompt_text=base_prompt,
                title_text=slide_title,
                cf_account_id=cf_account_id,
                cf_api_token=cf_ai_token,
                output_png=f"{workspace}/cf_{idx}_1.png",
                variant=idx_counter * 2
            )

            raw_block_mp4 = f"{workspace}/block_raw_{idx}.mp4"

            if scene_type == "teacher_lipsync":
                # For short 1-2 block videos, include a mid-block full-screen PowerPoint slide cutaway on the last block
                # so even a 1-minute test video showcases Lip-Sync Teacher + Full-Screen Slide + Synced Lyrics!
                use_cutaway = (total_blocks <= 2 and idx_counter == total_blocks - 1)
                cf_img_2 = None
                if use_cutaway:
                    cf_img_2 = generate_cloudflare_ai_image(
                        prompt_text=second_prompt,
                        title_text=slide_title,
                        cf_account_id=cf_account_id,
                        cf_api_token=cf_ai_token,
                        output_png=f"{workspace}/cf_{idx}_2.png",
                        variant=idx_counter * 2 + 1
                    )
                render_lipsync_teacher_block(
                    raw_avatar_path=raw_avatar_path,
                    audio_wav_path=output_wav,
                    output_mp4_path=raw_block_mp4,
                    lesson_title=lesson_title,
                    slide_title=slide_title,
                    slide_notes=notes,
                    cf_image_path=cf_img_1,
                    cf_image_2=cf_img_2,
                    block_idx=idx_counter,
                    total_blocks=total_blocks,
                    fps=24,
                    device=device,
                    run_ffmpeg_fn=run_ffmpeg,
                    include_mid_cutaway=use_cutaway
                )
            elif scene_type == "b_roll":
                cf_img_2 = generate_cloudflare_ai_image(
                    prompt_text=second_prompt,
                    title_text=slide_title,
                    cf_account_id=cf_account_id,
                    cf_api_token=cf_ai_token,
                    output_png=f"{workspace}/cf_{idx}_2.png",
                    variant=idx_counter * 2 + 1
                )
                render_broll_motion_block(
                    lesson_title=lesson_title,
                    slide_title=slide_title,
                    slide_notes=notes,
                    cf_image_1=cf_img_1,
                    cf_image_2=cf_img_2,
                    raw_avatar_path=raw_avatar_path,
                    audio_wav_path=output_wav,
                    output_mp4_path=raw_block_mp4,
                    block_idx=idx_counter,
                    total_blocks=total_blocks,
                    fps=24,
                    run_ffmpeg_fn=run_ffmpeg
                )
            else:
                # 'slides' - Full-screen PowerPoint Slide Deck with 2 interchanging visuals & progressive bullets
                cf_img_2 = generate_cloudflare_ai_image(
                    prompt_text=second_prompt,
                    title_text=slide_title,
                    cf_account_id=cf_account_id,
                    cf_api_token=cf_ai_token,
                    output_png=f"{workspace}/cf_{idx}_2.png",
                    variant=idx_counter * 2 + 1
                )
                render_slides_block(
                    lesson_title=lesson_title,
                    slide_title=slide_title,
                    slide_notes=notes,
                    cf_image_1=cf_img_1,
                    cf_image_2=cf_img_2,
                    raw_avatar_path=raw_avatar_path,
                    audio_wav_path=output_wav,
                    output_mp4_path=raw_block_mp4,
                    block_idx=idx_counter,
                    total_blocks=total_blocks,
                    fps=24,
                    run_ffmpeg_fn=run_ffmpeg
                )

            # Burn synchronized phrase-by-phrase on-screen lyrics at the bottom of the screen
            lyrics_ass = f"{workspace}/lyrics_{idx}.ass"
            create_synced_lyrics_ass(text, output_wav, lyrics_ass)
            final_chunk = f"{workspace}/final_chunk_{idx}.mp4"
            burn_presentation_slides(raw_block_mp4, lyrics_ass, final_chunk)

            clip_files.append(final_chunk)

        runpod.serverless.progress_update(job, {
            "step": "Stitching master media and uploading to Cloudflare R2...",
            "percent": 90,
            "time_left_sec": 15
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
                run_ffmpeg(f'ffmpeg -y -i {temp_voice} -stream_loop -1 -i {bg_track} -filter_complex "[1:a]volume=0.12[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[out]" -map "[out]" -c:a libmp3lame -b:a 320k -ac 2 {master_file}', desc="Mix audio with background music")
            else:
                run_ffmpeg(f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c:a libmp3lame -b:a 320k -ac 2 {master_file}", desc="Concat audio to MP3")
            r2_key = f"outputs/{job_id}_master.mp3"
            content_type = "audio/mpeg"
        else:
            master_file = f"{workspace}/master_lesson.mp4"
            if bg_track:
                temp_video = f"{workspace}/temp_video.mp4"
                run_ffmpeg(f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c copy {temp_video}", desc="Concat video blocks")
                run_ffmpeg(f'ffmpeg -y -i {temp_video} -stream_loop -1 -i {bg_track} -filter_complex "[1:a]volume=0.12[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[out]" -map 0:v -map "[out]" -c:v copy -c:a aac -b:a 192k -ac 2 {master_file}', desc="Mix video with background music")
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
