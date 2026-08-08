import os
import re
import shutil
import uuid
import json
import runpod
import torch
import requests
import subprocess
import boto3
from botocore.config import Config
import time
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"

R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "video-making")

s3_client = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    region_name='auto',
    config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
)

print(f"Loading Qwen3-TTS (1.7B Base) Multilingual Voice Model on {device}...")
try:
    tts_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
except Exception as e:
    print(f"Loading model with bfloat16 failed ({e}), retrying with standard precision...")
    tts_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=device
    )

print("Qwen3-TTS Model loaded successfully!")

def download_file(url, target_path):
    if not url:
        return ""
    print(f"Attempting to download {url} to {target_path}...")
    
    downloaded = False
    if R2_ENDPOINT_URL and (R2_ENDPOINT_URL in url or BUCKET_NAME in url):
        try:
            parts = url.split(f"{BUCKET_NAME}/", 1)
            if len(parts) == 2:
                object_key = parts[1].split("?")[0]
                print(f"Detected R2 S3 Object Key: '{object_key}'. Downloading via S3 client...")
                s3_client.download_file(BUCKET_NAME, object_key, target_path)
                downloaded = True
        except Exception as s3_err:
            print(f"S3 direct download failed: {s3_err}. Falling back to standard HTTP get...")
            downloaded = False

    if not downloaded:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            raise Exception(f"HTTP Download Error: Received status {response.status_code} for {url}")
            
    if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
        raise Exception(f"Downloaded file from {url} is empty or failed to save at {target_path}.")
        
    return target_path

def generate_sunbird_audio(text, output_path, sunbird_key, language="sw"):
    headers = {
        "Authorization": f"Bearer {sunbird_key}",
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    lang_map = {
        "sw": ("swa", "salt_swa_0001"),
        "lg": ("lug", "salt_lug_0001"),
        "en": ("eng", "salt_eng_0001"),
        "swahili": ("swa", "salt_swa_0001"),
        "luganda": ("lug", "salt_lug_0001"),
        "east_african_english": ("eng", "salt_eng_0001"),
        "sunbird_english": ("eng", "salt_eng_0001")
    }
    iso_lang, voice_tag = lang_map.get(language, ("eng", "salt_eng_0001"))
    payload = {
        "text": text,
        "model": "orpheus-3b-tts",
        "voice": voice_tag,
        "language": iso_lang
    }
    response = requests.post("https://api.sunbird.ai/tasks/audio/speech", json=payload, headers=headers)
    if response.status_code == 200:
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            data = response.json()
            audio_url = None
            if "output" in data and isinstance(data["output"], dict) and "audio_url" in data["output"]:
                audio_url = data["output"]["audio_url"]
            elif "audio_url" in data:
                audio_url = data["audio_url"]
            elif "url" in data:
                audio_url = data["url"]
            
            if audio_url:
                audio_resp = requests.get(audio_url)
                if audio_resp.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(audio_resp.content)
                    return
                else:
                    raise Exception(f"Failed to download Sunbird audio from signed URL: HTTP {audio_resp.status_code}")
            elif "audio_content" in data or "base64" in data:
                import base64
                b64_str = data.get("audio_content") or data.get("base64")
                with open(output_path, 'wb') as f:
                    f.write(base64.b64decode(b64_str))
                return
            else:
                raise Exception(f"Unexpected JSON format from Sunbird API: {data}")
        else:
            with open(output_path, 'wb') as f:
                f.write(response.content)
    else:
        raise Exception(f"Sunbird API failed with status {response.status_code}: {response.text}")

def _synthesize_single_segment(text, output_wav, language_mode, tts_model, voice_clone_prompt, sunbird_key):
    if not text or not text.strip():
        subprocess.run(f"ffmpeg -y -f lavfi -i anullsrc=r=24000:cl=stereo:d=0.2 -c:a pcm_s16le {output_wav}", shell=True, check=True)
        return

    if language_mode == 'english_cloned' and voice_clone_prompt is not None:
        wavs, sr = tts_model.generate_voice_clone(
            text=text,
            language="English",
            voice_clone_prompt=voice_clone_prompt
        )
        audio_data = wavs[0]
        if hasattr(audio_data, 'cpu'):
            audio_data = audio_data.cpu().numpy()
        sf.write(output_wav, audio_data, sr)
    elif language_mode in ['swahili', 'luganda', 'east_african_english', 'sunbird_english'] and sunbird_key:
        lang_code = {"swahili": "sw", "luganda": "lg", "east_african_english": "en", "sunbird_english": "en"}[language_mode]
        generate_sunbird_audio(text, output_wav, sunbird_key, language=lang_code)
    else:
        if voice_clone_prompt is not None:
            wavs, sr = tts_model.generate_voice_clone(
                text=text,
                language="English",
                voice_clone_prompt=voice_clone_prompt
            )
            audio_data = wavs[0]
            if hasattr(audio_data, 'cpu'):
                audio_data = audio_data.cpu().numpy()
            sf.write(output_wav, audio_data, sr)
        elif sunbird_key and language_mode in ['east_african_english', 'sunbird_english']:
            generate_sunbird_audio(text, output_wav, sunbird_key, language="en")
        else:
            raise Exception("Missing valid voice reference sample for Qwen3-TTS cloning or Sunbird AI credentials.")

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
    subprocess.run(f"ffmpeg -y -i {audio_path} -af \"{filter_str}\" -c:a pcm_s16le {mod_temp}", shell=True, check=True)
    shutil.copyfile(mod_temp, audio_path)

def synthesize_block_audio_with_pauses(text, output_wav, language_mode, tts_model, voice_clone_prompt, sunbird_key, workspace, whisper_mode=False, speed_val=1.0, pitch_val=1.0, volume_val=1.0):
    parts = re.split(r'\[pause\s+([0-9\.]+)\s*s?\]', text, flags=re.IGNORECASE)
    if len(parts) <= 1:
        _synthesize_single_segment(text.strip(), output_wav, language_mode, tts_model, voice_clone_prompt, sunbird_key)
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
            subprocess.run(f"ffmpeg -y -f lavfi -i anullsrc=r=24000:cl=stereo:d={duration} -c:a pcm_s16le {silence_wav}", shell=True, check=True)
            temp_clips.append(silence_wav)
        else:
            clean_text = part.strip()
            if clean_text:
                seg_wav = f"{workspace}/seg_{idx}_{uuid.uuid4().hex[:6]}.wav"
                _synthesize_single_segment(clean_text, seg_wav, language_mode, tts_model, voice_clone_prompt, sunbird_key)
                temp_clips.append(seg_wav)

    if not temp_clips:
        _synthesize_single_segment(".", output_wav, language_mode, tts_model, voice_clone_prompt, sunbird_key)
        return
    elif len(temp_clips) == 1:
        shutil.copyfile(temp_clips[0], output_wav)
        return

    seg_list = f"{workspace}/seg_concat_{uuid.uuid4().hex[:6]}.txt"
    with open(seg_list, "w") as f:
        for c in temp_clips:
            f.write(f"file '{c}'\n")
    subprocess.run(f"ffmpeg -y -f concat -safe 0 -i {seg_list} -c:a pcm_s16le {output_wav}", shell=True, check=True)
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
    subprocess.run(f"ffmpeg -y -f lavfi -i aevalsrc=\"{freqs}\":s=24000:d=600 -af \"lowpass=f=800,volume=0.25\" {synth_bg}", shell=True, check=True)
    return synth_bg

def create_presentation_slide_file(slide_notes, output_sub_path):
    ass_template = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, BackColour, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: SlideStyle,Helvetica,42,&H00FFFFFF,&H80000000,0,3,6,100,100,50,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    
    formatted_bullets = ""
    for note in slide_notes:
        formatted_bullets += f"  {note}\\N"
        
    event_line = f"Dialogue: 0,0:00:00.00,0:00:30.00,SlideStyle,,0,0,0,,{formatted_bullets}\n"
    
    with open(output_sub_path, "w", encoding="utf-8") as f:
        f.write(ass_template + event_line)

def burn_presentation_slides(input_video, subtitle_file, final_output):
    cmd = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-vf', f"subtitles={subtitle_file}",
        '-c:a', 'copy',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        final_output
    ]
    subprocess.run(cmd, check=True)

def create_broll_cutaway_overlay(teacher_video, broll_video, output_path):
    cmd = [
        'ffmpeg', '-y',
        '-i', teacher_video,
        '-i', broll_video,
        '-filter_complex', 
        "[1:v]setpts=PTS-STARTPTS[broll];[0:v][broll]overlay=enable='between(t,5,25)':x=0:y=0[outv]",
        '-map', '[outv]', 
        '-map', '0:a',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-c:a', 'copy',
        output_path
    ]
    subprocess.run(cmd, check=True)

def handler(job):
    job_input = job['input']
    
    timeline_data = job_input.get('timeline_data')
    voice_ref_url = job_input.get('voice_ref_url')
    avatar_url = job_input.get('avatar_url')
    language_mode = job_input.get('language_mode')
    sunbird_key = job_input.get('sunbird_key')
    bg_music = job_input.get('bg_music', 'none')
    global_whisper = (job_input.get('whisper_mode', 'no') == 'yes')
    try: global_speed = float(job_input.get('speaking_speed', 1.0))
    except: global_speed = 1.0
    try: global_pitch = float(job_input.get('voice_pitch', 1.0))
    except: global_pitch = 1.0
    try: global_volume = float(job_input.get('voice_volume', 1.0))
    except: global_volume = 1.0
    output_format = job_input.get('output_format', 'video')
    
    job_id = job['id']
    
    workspace = f"/tmp/{job_id}"
    os.makedirs(workspace, exist_ok=True)
    
    total_blocks = len(timeline_data['timeline']) if timeline_data and 'timeline' in timeline_data else 1
    start_time = time.time()
    
    runpod.serverless.progress_update(job, {
        "step": f"Downloading media assets & preparing {'audio' if output_format == 'audio' else 'video'} rendering...",
        "percent": 5,
        "time_left_sec": int(total_blocks * (8 if output_format == 'audio' else 35))
    })
    
    voice_ref_text = job_input.get('voice_ref_text', '').strip()
    voice_clone_prompt = None
    if voice_ref_url:
        raw_voice_path = download_file(voice_ref_url, f"{workspace}/ref_voice_orig")
        ref_voice_path = f"{workspace}/ref_voice.wav"
        print("Normalizing reference audio to 24kHz 16-bit PCM WAV for Qwen3-TTS...")
        subprocess.run([
            'ffmpeg', '-y', '-i', raw_voice_path,
            '-ac', '1', '-ar', '24000', '-sample_fmt', 's16',
            ref_voice_path
        ], check=True)
        
        print("Extracting speaker embeddings & creating reusable Qwen3-TTS voice clone prompt...")
        use_x_vector = not bool(voice_ref_text)
        voice_clone_prompt = tts_model.create_voice_clone_prompt(
            ref_audio=ref_voice_path,
            ref_text=voice_ref_text if voice_ref_text else "",
            x_vector_only_mode=use_x_vector
        )
        
    avatar_img_path = ""
    if output_format == 'video' and avatar_url:
        avatar_img_path = download_file(avatar_url, f"{workspace}/avatar.jpg")
    
    clip_files = []

    for idx_counter, block in enumerate(timeline_data['timeline']):
        idx = block.get('block_index', idx_counter)
        text = block['spoken_script']
        flag = block.get('flag_type', 'teacher')
        notes = block.get('slide_notes', [])
        
        elapsed = time.time() - start_time
        sec_per_block = elapsed / idx_counter if idx_counter > 0 else (8.0 if output_format == 'audio' else 30.0)
        remaining_blocks = total_blocks - idx_counter
        time_left = max(5, int(remaining_blocks * sec_per_block))
        percent_done = int(10 + (idx_counter / max(1, total_blocks)) * 75)
        
        runpod.serverless.progress_update(job, {
            "step": f"Rendering Block {idx_counter+1} of {total_blocks}: Generating {'voiceover audio' if output_format == 'audio' else 'voice & video animation'}...",
            "percent": percent_done,
            "time_left_sec": time_left
        })
        
        output_wav = f"{workspace}/audio_{idx}.wav"
        
        delivery = block.get('delivery_style', 'normal')
        blk_whisper = True if delivery == 'whisper' else global_whisper
        blk_speed = global_speed * (0.8 if delivery == 'slow' else (1.2 if delivery == 'fast' else 1.0))
        synthesize_block_audio_with_pauses(
            text, output_wav, language_mode, tts_model, voice_clone_prompt, sunbird_key, workspace,
            whisper_mode=blk_whisper, speed_val=blk_speed, pitch_val=global_pitch, volume_val=global_volume
        )
        
        if output_format == 'audio':
            clip_files.append(output_wav)
            continue
            
        # Video Rendering Mode
        teacher_raw_mp4 = f"{workspace}/teacher_raw_{idx}.mp4"
        subprocess.run(f"ffmpeg -y -loop 1 -i {avatar_img_path} -i {output_wav} -c:v libx264 -c:a aac -b:a 320k -ac 2 -shortest {teacher_raw_mp4}", shell=True, check=True)

        if flag == 'b_roll':
            broll_raw_mp4 = f"{workspace}/broll_raw_{idx}.mp4"
            subprocess.run(f"ffmpeg -y -f lavfi -i color=c=blue:s=1920x1080:d=20 -c:v libx264 {broll_raw_mp4}", shell=True, check=True)
            
            teacher_overlaid_mp4 = f"{workspace}/teacher_overlaid_{idx}.mp4"
            create_broll_cutaway_overlay(teacher_raw_mp4, broll_raw_mp4, teacher_overlaid_mp4)
            teacher_raw_mp4 = teacher_overlaid_mp4 

        ass_path = f"{workspace}/slides_{idx}.ass"
        create_presentation_slide_file(notes, ass_path)
        
        final_chunk = f"{workspace}/final_chunk_{idx}.mp4"
        burn_presentation_slides(teacher_raw_mp4, ass_path, final_chunk)
        
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
            subprocess.run(f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c:a pcm_s16le {temp_voice}", shell=True, check=True)
            subprocess.run(f"ffmpeg -y -i {temp_voice} -stream_loop -1 -i {bg_track} -filter_complex \"[1:a]volume=0.12[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[out]\" -map \"[out]\" -c:a libmp3lame -b:a 320k -ac 2 {master_file}", shell=True, check=True)
        else:
            subprocess.run(f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c:a libmp3lame -b:a 320k -ac 2 {master_file}", shell=True, check=True)
        r2_key = f"outputs/{job_id}_master.mp3"
    else:
        master_file = f"{workspace}/master_lesson.mp4"
        if bg_track:
            temp_video = f"{workspace}/temp_video.mp4"
            subprocess.run(f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c copy {temp_video}", shell=True, check=True)
            subprocess.run(f"ffmpeg -y -i {temp_video} -stream_loop -1 -i {bg_track} -filter_complex \"[1:a]volume=0.12[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[out]\" -map 0:v -map \"[out]\" -c:v copy -c:a aac -b:a 320k -ac 2 {master_file}", shell=True, check=True)
        else:
            subprocess.run(f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c copy {master_file}", shell=True, check=True)
        r2_key = f"outputs/{job_id}_master.mp4"
    
    print(f"Uploading {master_file} to R2 bucket {BUCKET_NAME} as {r2_key}...")
    s3_client.upload_file(master_file, BUCKET_NAME, r2_key)
    
    try:
        download_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': r2_key},
            ExpiresIn=604800 # 7 days valid
        )
    except Exception as e:
        download_url = f"{R2_ENDPOINT_URL}/{BUCKET_NAME}/{r2_key}"
    
    print(f"Job completed successfully! Returning output with download_url: {download_url}")
    return {
        "status": "success",
        "type": output_format,
        "download_url": download_url
    }

runpod.serverless.start({"handler": handler})


