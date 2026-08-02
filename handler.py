import os

# Auto-agree to Coqui TTS Terms of Service to prevent interactive prompt crash
os.environ['COQUI_TOS_AGREED'] = '1'

import json
import runpod
import torch
import requests
import subprocess
import boto3
import time
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"

R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "video-making")

s3_client = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY
)

print("Loading XTTS v2 Multilingual Voice Model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def download_file(url, target_path):
    if not url:
        return ""
    print(f"Attempting to download {url} to {target_path}...")
    
    # Check if this URL is inside our R2 S3 bucket endpoint
    downloaded = False
    if R2_ENDPOINT_URL and (R2_ENDPOINT_URL in url or BUCKET_NAME in url):
        try:
            # Extract object key after bucket name
            parts = url.split(f"{BUCKET_NAME}/", 1)
            if len(parts) == 2:
                object_key = parts[1].split("?")[0] # strip query params if any
                print(f"Detected R2 S3 Object Key: '{object_key}'. Downloading via S3 client...")
                s3_client.download_file(BUCKET_NAME, object_key, target_path)
                downloaded = True
        except Exception as s3_err:
            print(f"S3 direct download failed: {s3_err}. Falling back to standard HTTP get...")
            downloaded = False

    # Fallback to direct HTTP request
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
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "language": language 
    }
    response = requests.post("https://api.sunbird.ai/tasks/tts", json=payload, headers=headers)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Sunbird API failed with status {response.status_code}: {response.text}")

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
        formatted_bullets += f"• {note}\\N"
        
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
    
    ref_voice_path = ""
    if voice_ref_url:
        raw_voice_path = download_file(voice_ref_url, f"{workspace}/ref_voice_orig")
        ref_voice_path = f"{workspace}/ref_voice.wav"
        # Always convert reference audio (whether MP3, M4A, OGG, WAV) into pure 22050Hz 16-bit PCM WAV for Coqui XTTS
        print("Normalizing reference audio to 22.05kHz 16-bit PCM WAV...")
        subprocess.run([
            'ffmpeg', '-y', '-i', raw_voice_path,
            '-ac', '1', '-ar', '22050', '-sample_fmt', 's16',
            ref_voice_path
        ], check=True)
        
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
        
        if language_mode == 'english_cloned' and ref_voice_path and os.path.exists(ref_voice_path):
            tts.tts_to_file(text=text, speaker_wav=ref_voice_path, language="en", file_path=output_wav)
        elif language_mode == 'swahili' and sunbird_key:
            generate_sunbird_audio(text, output_wav, sunbird_key, language="sw")
        elif language_mode == 'luganda' and sunbird_key:
            generate_sunbird_audio(text, output_wav, sunbird_key, language="lg")
        else:
            if ref_voice_path and os.path.exists(ref_voice_path):
                tts.tts_to_file(text=text, speaker_wav=ref_voice_path, language="en", file_path=output_wav)
            else:
                raise Exception("Missing valid voice reference sample for English cloned TTS or Sunbird credentials.")
        
        if output_format == 'audio':
            clip_files.append(output_wav)
            continue
            
        # Video Rendering Mode
        teacher_raw_mp4 = f"{workspace}/teacher_raw_{idx}.mp4"
        subprocess.run(f"ffmpeg -y -loop 1 -i {avatar_img_path} -i {output_wav} -c:v libx264 -tune stillimage -c:a copy -shortest {teacher_raw_mp4}", shell=True, check=True)

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
            
    if output_format == 'audio':
        master_file = f"{workspace}/master_lesson.mp3"
        subprocess.run(f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c:a libmp3lame -q:a 2 {master_file}", shell=True, check=True)
        r2_key = f"outputs/{job_id}_master.mp3"
    else:
        master_file = f"{workspace}/master_lesson.mp4"
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
    
    runpod.serverless.progress_update(job, {
        "step": "Generation complete!",
        "percent": 100,
        "time_left_sec": 0
    })
    
    return {
        "status": "success",
        "type": output_format,
        "download_url": download_url
    }

runpod.serverless.start({"handler": handler})
