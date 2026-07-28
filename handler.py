import os
import json
import runpod
import torch
import requests
import subprocess
import boto3
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
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return target_path

def generate_sunbird_audio(text, output_path, sunbird_key):
    headers = {
        "Authorization": f"Bearer {sunbird_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "language": "sw" 
    }
    response = requests.post("https://api.sunbird.ai/tasks/tts", json=payload, headers=headers)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception("Sunbird API failed")

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
    job_id = job['id']
    
    workspace = f"/tmp/{job_id}"
    os.makedirs(workspace, exist_ok=True)
    
    ref_voice_path = download_file(voice_ref_url, f"{workspace}/ref_voice.wav")
    avatar_img_path = download_file(avatar_url, f"{workspace}/avatar.jpg")
    
    clip_files = []

    for block in timeline_data['timeline']:
        idx = block.get('block_index', len(clip_files))
        text = block['spoken_script']
        flag = block['flag_type']
        notes = block['slide_notes']
        
        output_wav = f"{workspace}/audio_{idx}.wav"
        
        if language_mode == 'english_cloned':
            tts.tts_to_file(text=text, speaker_wav=ref_voice_path, language="en", file_path=output_wav)
        else:
            generate_sunbird_audio(text, output_wav, sunbird_key)
            
        teacher_raw_mp4 = f"{workspace}/teacher_raw_{idx}.mp4"
        subprocess.run(f"ffmpeg -y -loop 1 -i {avatar_img_path} -i {output_wav} -c:v libx264 -tune stillimage -c:a copy -shortest {teacher_raw_mp4}", shell=True)

        if flag == 'b_roll':
            broll_raw_mp4 = f"{workspace}/broll_raw_{idx}.mp4"
            subprocess.run(f"ffmpeg -y -f lavfi -i color=c=blue:s=1920x1080:d=20 -c:v libx264 {broll_raw_mp4}", shell=True)
            
            teacher_overlaid_mp4 = f"{workspace}/teacher_overlaid_{idx}.mp4"
            create_broll_cutaway_overlay(teacher_raw_mp4, broll_raw_mp4, teacher_overlaid_mp4)
            teacher_raw_mp4 = teacher_overlaid_mp4 

        ass_path = f"{workspace}/slides_{idx}.ass"
        create_presentation_slide_file(notes, ass_path)
        
        final_chunk = f"{workspace}/final_chunk_{idx}.mp4"
        burn_presentation_slides(teacher_raw_mp4, ass_path, final_chunk)
        
        clip_files.append(final_chunk)

    concat_list = f"{workspace}/concat.txt"
    with open(concat_list, "w") as f:
        for clip in clip_files:
            f.write(f"file '{clip}'\n")
            
    master_mp4 = f"{workspace}/master_lesson.mp4"
    subprocess.run(f"ffmpeg -y -f concat -safe 0 -i {concat_list} -c copy {master_mp4}", shell=True, check=True)
    
    r2_key = f"outputs/{job_id}_master.mp4"
    s3_client.upload_file(master_mp4, BUCKET_NAME, r2_key)
    
    return {
        "status": "success",
        "type": "video",
        "download_url": f"{R2_ENDPOINT_URL}/{BUCKET_NAME}/{r2_key}"
    }

runpod.serverless.start({"handler": handler})
