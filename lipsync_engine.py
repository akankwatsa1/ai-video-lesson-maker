import os
import math
import cv2
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import librosa
from scipy import signal
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ==============================================================================
# 1. WAV2LIP NEURAL ARCHITECTURE & MEL-SPECTROGRAM PROCESSOR
# ==============================================================================

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Wav2Lip(nn.Module):
    def __init__(self):
        super().__init__()
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),
            nn.Sequential(
                Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
            ),
        ])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        )

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),
            nn.Sequential(
                Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)
            ),
        ])

        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, audio_sequences, face_sequences):
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences)
        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            x = torch.cat((x, feats[-1]), dim=1)
            feats.pop()

        x = self.output_block(x)
        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x
        return outputs


_MEL_BASIS = None
_WAV2LIP_MODEL = None
WAV2LIP_CHECKPOINT_PATHS = [
    "/app/checkpoints/wav2lip_gan.pth",
    "/app/wav2lip_gan.pth",
]

def _compute_melspectrogram(wav_path):
    global _MEL_BASIS
    wav, _ = librosa.load(wav_path, sr=16000)
    # Add tiny epsilon noise to prevent NaN on pure digital silence segments
    wav = wav + np.random.normal(0, 1e-6, size=wav.shape).astype(np.float32)
    preemp = signal.lfilter([1.0, -0.97], [1.0], wav)
    D = librosa.stft(y=preemp, n_fft=800, hop_length=200, win_length=800)
    if _MEL_BASIS is None:
        _MEL_BASIS = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
    S = np.dot(_MEL_BASIS, np.abs(D))
    min_level = np.exp(-100.0 / 20.0 * np.log(10.0))
    S_db = 20.0 * np.log10(np.maximum(min_level, S)) - 20.0
    norm = np.clip((2.0 * 4.0) * ((S_db - (-100.0)) / 100.0) - 4.0, -4.0, 4.0)
    return norm, wav

def get_wav2lip_model(device="cuda"):
    global _WAV2LIP_MODEL
    if _WAV2LIP_MODEL is not None:
        return _WAV2LIP_MODEL

    ckpt_path = None
    for p in WAV2LIP_CHECKPOINT_PATHS:
        if os.path.exists(p) and os.path.getsize(p) > 10_000_000:
            ckpt_path = p
            break

    if not ckpt_path:
        try:
            from huggingface_hub import hf_hub_download
            print("Downloading wav2lip_gan.pth checkpoint from HuggingFace...")
            ckpt_path = hf_hub_download(
                repo_id="camenduru/Wav2Lip",
                filename="checkpoints/wav2lip_gan.pth",
                local_dir="/app"
            )
        except Exception as e:
            print(f"Wav2Lip checkpoint download note ({e}); using fallback jaw/mouth articulation.")
            return None

    try:
        model = Wav2Lip()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_sd)
        model = model.to(device).eval()
        _WAV2LIP_MODEL = model
        print("Wav2Lip-GAN lip-sync model loaded successfully!")
        return _WAV2LIP_MODEL
    except Exception as e:
        print(f"Wav2Lip load warning ({e}); falling back to procedural lip articulation.")
        return None


# ==============================================================================
# 2. SMART PORTRAIT CROPPING & FACE DETECTION FOR PHONE PHOTOS
# ==============================================================================

def detect_face_bbox(img_bgr):
    """
    Detects the primary face in a BGR image and returns (x1, y1, x2, y2)
    padded appropriately for Wav2Lip mouth/chin synthesis.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(48, 48))

    if len(faces) == 0:
        alt_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
        if os.path.exists(alt_path):
            detector_alt = cv2.CascadeClassifier(alt_path)
            faces = detector_alt.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(48, 48))

    if len(faces) > 0:
        # Pick largest face
        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        # Pad downward slightly to capture chin/jaw movement cleanly
        x1 = max(0, int(fx - fw * 0.05))
        x2 = min(w, int(fx + fw * 1.05))
        y1 = max(0, int(fy))
        y2 = min(h, int(fy + fh * 1.12))
        return (x1, y1, x2, y2), True

    # Fallback central portrait face region if stylized/profile face
    cx, cy = w // 2, int(h * 0.42)
    fw, fh = int(min(w, h) * 0.42), int(min(w, h) * 0.46)
    return (max(0, cx - fw // 2), max(0, cy - fh // 2), min(w, cx + fw // 2), min(h, cy + fh // 2)), False


def prepare_presenter_portrait(raw_avatar_path, target_w=680, target_h=820):
    """
    Takes any phone photo (vertical, square, or horizontal), detects the teacher's face,
    and smartly crops/scales it into a clean, natural presenter portrait (680x820) with no ugly black bars.
    """
    if not raw_avatar_path or not os.path.exists(raw_avatar_path):
        canvas = np.full((target_h, target_w, 3), (45, 55, 75), dtype=np.uint8)
        return canvas, (target_w // 4, target_h // 4, 3 * target_w // 4, 3 * target_h // 4), False

    img = cv2.imread(raw_avatar_path)
    if img is None:
        pil_img = Image.open(raw_avatar_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    (fx1, fy1, fx2, fy2), found = detect_face_bbox(img)
    face_cx = (fx1 + fx2) // 2
    face_cy = (fy1 + fy2) // 2

    target_aspect = target_w / float(target_h)
    img_aspect = w / float(h)

    if img_aspect > target_aspect:
        # Image is wider than presenter card: crop width centered on face_cx
        crop_h = h
        crop_w = int(h * target_aspect)
        x0 = max(0, min(w - crop_w, face_cx - crop_w // 2))
        y0 = 0
    else:
        # Image is taller (e.g. vertical phone photo): crop height keeping face in upper-middle third
        crop_w = w
        crop_h = int(w / target_aspect)
        x0 = 0
        desired_y0 = face_cy - int(crop_h * 0.38)
        y0 = max(0, min(h - crop_h, desired_y0))

    cropped = img[y0:y0 + crop_h, x0:x0 + crop_w]
    presenter_bgr = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # Re-detect face bbox on the final 680x820 presenter portrait
    bbox, found_final = detect_face_bbox(presenter_bgr)
    return presenter_bgr, bbox, found_final


# ==============================================================================
# 3. STUDIO COMPOSITING, POWERPOINT SLIDE CARDS & LIP-SYNC RENDERER
# ==============================================================================

def _load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _wrap_text(draw, text, font, max_width):
    words = str(text).split()
    lines = []
    current = []
    for w in words:
        test = " ".join(current + [w])
        bbox = draw.textbbox((0, 0), test, font=font)
        if (bbox[2] - bbox[0]) <= max_width or not current:
            current.append(w)
        else:
            lines.append(" ".join(current))
            current = [w]
    if current:
        lines.append(" ".join(current))
    return lines


def build_studio_Composite_background(
    raw_avatar_path,
    lesson_title,
    slide_title,
    slide_notes,
    cf_image_path=None,
    block_idx=0,
    total_blocks=1
):
    """
    Creates a 1920x1080 YouTube Broadcast Studio Canvas:
      - Blurred & color-graded ambient backdrop from the teacher's photo (no raw black look)
      - Left Frame (x=64, y=88, w=680, h=820): Rounded presenter studio slot
      - Right Card (x=784, y=88, w=1072, h=820): PowerPoint-style concept card with
        Cloudflare Workers AI illustration + high-contrast bullet points
      - Bottom area (y=925..1065): Clean lower-third zone for synchronized on-screen lyrics
    """
    W, H = 1920, 1080
    if raw_avatar_path and os.path.exists(raw_avatar_path):
        bg_pil = Image.open(raw_avatar_path).convert("RGB").resize((W, H), Image.Resampling.BICUBIC)
        bg_pil = bg_pil.filter(ImageFilter.GaussianBlur(radius=38))
        # Darken and tint toward rich studio navy
        navy_tint = Image.new("RGB", (W, H), (15, 23, 42))
        bg_pil = Image.blend(bg_pil, navy_tint, alpha=0.76)
    else:
        bg_pil = Image.new("RGB", (W, H), (15, 23, 42))

    draw = ImageDraw.Draw(bg_pil, "RGBA")

    # Top subtle header bar
    draw.rounded_rectangle([64, 22, 1856, 72], radius=14, fill=(30, 41, 59, 215), outline=(71, 85, 105, 160), width=1)
    font_header = _load_font(22, bold=True)
    font_badge = _load_font(19, bold=True)
    header_text = f"  {lesson_title[:65]}"
    draw.text((84, 34), header_text, font=font_header, fill=(241, 245, 249, 255))
    part_label = f"SECTION {block_idx + 1} OF {total_blocks}"
    draw.text((1640, 35), part_label, font=font_badge, fill=(56, 189, 248, 255))

    # Right PowerPoint Concept Card (x=784..1856, y=88..908)
    draw.rounded_rectangle([784, 88, 1856, 908], radius=24, fill=(255, 255, 255, 248), outline=(203, 213, 225, 255), width=3)
    # Accent banner inside PowerPoint Card
    draw.rounded_rectangle([784, 88, 1856, 178], radius=24, fill=(30, 58, 138, 255))
    draw.rectangle([784, 150, 1856, 178], fill=(30, 58, 138, 255))

    font_slide_title = _load_font(34, bold=True)
    heading = (slide_title or lesson_title or f"Key Concept #{block_idx + 1}").strip()
    draw.text((820, 114), heading[:46], font=font_slide_title, fill=(255, 255, 255, 255))

    # If a Cloudflare Workers AI illustration is available, embed it cleanly on the right card
    notes_max_w = 990
    notes_start_y = 210
    if cf_image_path and os.path.exists(cf_image_path):
        try:
            cf_img = Image.open(cf_image_path).convert("RGB").resize((992, 340), Image.Resampling.LANCZOS)
            mask = Image.new("L", (992, 340), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, 992, 340], radius=16, fill=255)
            bg_pil.paste(cf_img, (824, 200), mask)
            notes_start_y = 565
        except Exception as e:
            print(f"CF image embed note: {e}")

    # Render summarized bullet points with crisp PowerPoint styling
    font_bullet = _load_font(29, bold=False)
    curr_y = notes_start_y
    bullet_colors = [(37, 99, 235), (16, 185, 129), (245, 158, 11), (139, 92, 246)]

    for idx, note in enumerate((slide_notes or [])[:4]):
        clean_note = str(note).strip()
        if not clean_note:
            continue
        lines = _wrap_text(draw, clean_note, font_bullet, notes_max_w - 55)
        b_col = bullet_colors[idx % len(bullet_colors)]
        draw.ellipse([824, curr_y + 8, 846, curr_y + 30], fill=b_col + (255,))
        for line in lines[:2]:
            draw.text((864, curr_y), line, font=font_bullet, fill=(15, 23, 42, 255))
            curr_y += 40
        curr_y += 14
        if curr_y > 865:
            break

    # Left Presenter Frame border shadow
    draw.rounded_rectangle([58, 82, 750, 914], radius=26, fill=(15, 23, 42, 180), outline=(56, 189, 248, 220), width=4)

    # Bottom Lyrics Bar Backdrop (y=930..1058)
    draw.rounded_rectangle([64, 932, 1856, 1056], radius=18, fill=(15, 23, 42, 215), outline=(51, 65, 85, 180), width=2)

    return cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)


def _create_rounded_mask(w, h, radius=22):
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w, h], radius=radius, fill=255)
    return np.array(mask)


def _create_feathered_mouth_mask(w, h):
    """
    Creates a smooth elliptical alpha mask focused on the lower half (mouth/jaw)
    and feathered across the cheeks/nose so Wav2Lip blends invisibly into the original face.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, int(h * 0.62))
    axes = (int(w * 0.40), int(h * 0.34))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    blur_k = max(3, (min(w, h) // 5) | 1)
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    return np.expand_dims(mask, axis=-1)


def render_lipsync_teacher_block(
    raw_avatar_path,
    audio_wav_path,
    output_mp4_path,
    lesson_title,
    slide_title,
    slide_notes,
    cf_image_path=None,
    cf_image_2=None,
    block_idx=0,
    total_blocks=1,
    fps=24,
    device="cuda",
    run_ffmpeg_fn=None,
    include_mid_cutaway=False
):
    """
    Generates an audio-driven lip-synced teacher video (24fps, 1920x1080) inside the
    Broadcast Studio Layout with:
      1. Wav2Lip-GAN mouth synchronization (or audio-envelope jaw articulation fallback)
      2. Feathered elliptical blending (zero bounding-box edges)
      3. LivePortrait-style natural head sway & breathing micro-motion
      4. Optional mid-block PowerPoint/B-roll cutaway for short 1-2 block videos
    """
    presenter_w, presenter_h = 680, 820
    presenter_bgr, (x1, y1, x2, y2), _ = prepare_presenter_portrait(raw_avatar_path, presenter_w, presenter_h)
    studio_bg_bgr = build_studio_Composite_background(
        raw_avatar_path=raw_avatar_path,
        lesson_title=lesson_title,
        slide_title=slide_title,
        slide_notes=slide_notes,
        cf_image_path=cf_image_path,
        block_idx=block_idx,
        total_blocks=total_blocks
    )

    cutaway_slide_bgr = None
    if include_mid_cutaway:
        tmp_slide_png = f"{output_mp4_path}.cutaway.png"
        render_powerpoint_slide_frame(
            lesson_title=lesson_title,
            slide_title=slide_title,
            slide_notes=slide_notes,
            cf_image_path=cf_image_2 or cf_image_path,
            raw_avatar_path=raw_avatar_path,
            output_png=tmp_slide_png,
            block_idx=block_idx,
            total_blocks=total_blocks,
            reveal_all=True
        )
        cutaway_slide_bgr = cv2.imread(tmp_slide_png)
        if os.path.exists(tmp_slide_png):
            os.remove(tmp_slide_png)

    mel, wav_16k = _compute_melspectrogram(audio_wav_path)
    mel_step_size = 16
    mel_idx_multiplier = 80.0 / float(fps)
    mel_chunks = []
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > mel.shape[1]:
            if mel.shape[1] >= mel_step_size:
                mel_chunks.append(mel[:, mel.shape[1] - mel_step_size:])
            else:
                pad_mel = np.pad(mel, ((0, 0), (0, mel_step_size - mel.shape[1])), mode="edge")
                mel_chunks.append(pad_mel)
            break
        mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])
        i += 1

    num_frames = max(1, len(mel_chunks))

    # Compute per-frame RMS speech energy for natural head nod & jaw fallback
    samples_per_frame = max(1, len(wav_16k) // num_frames)
    frame_energies = []
    for f_idx in range(num_frames):
        seg = wav_16k[f_idx * samples_per_frame:(f_idx + 1) * samples_per_frame]
        rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) > 0 else 0.0
        frame_energies.append(min(1.0, rms * 8.0))

    face_crop_orig = presenter_bgr[y1:y2, x1:x2].copy()
    fh, fw = face_crop_orig.shape[:2]
    feather_mask = _create_feathered_mouth_mask(fw, fh)
    rounded_card_mask = _create_rounded_mask(presenter_w, presenter_h, radius=22)
    card_alpha = (rounded_card_mask.astype(np.float32) / 255.0)[..., None]

    # Run Wav2Lip-GAN in fast GPU batches
    wav2lip_model = get_wav2lip_model(device=device)
    predicted_faces = []

    if wav2lip_model is not None and fw > 16 and fh > 16:
        face_96 = cv2.resize(face_crop_orig, (96, 96), interpolation=cv2.INTER_AREA)
        face_masked = face_96.copy()
        face_masked[48:, :] = 0
        face_concat = np.concatenate((face_masked, face_96), axis=2) / 255.0
        face_tensor_single = torch.from_numpy(np.transpose(face_concat, (2, 0, 1))).float().unsqueeze(0).to(device)

        batch_size = 64
        with torch.no_grad():
            for b_start in range(0, num_frames, batch_size):
                b_mels = np.array(mel_chunks[b_start:b_start + batch_size])
                b_len = len(b_mels)
                b_mels = np.reshape(b_mels, (b_len, b_mels.shape[1], b_mels.shape[2], 1))
                mel_tensor = torch.from_numpy(np.transpose(b_mels, (0, 3, 1, 2))).float().to(device)
                img_tensor = face_tensor_single.repeat(b_len, 1, 1, 1)
                preds = wav2lip_model(mel_tensor, img_tensor)
                preds_np = (preds.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
                for p in preds_np:
                    predicted_faces.append(p)

    # Write frames directly into an intermediate video stream
    silent_mp4 = f"{output_mp4_path}.silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_mp4, fourcc, float(fps), (1920, 1080))

    base_slot = studio_bg_bgr[88:88 + presenter_h, 64:64 + presenter_w].astype(np.float32)
    cut_start = int(num_frames * 0.18)
    cut_end = int(num_frames * 0.58)

    for f_idx in range(num_frames):
        if cutaway_slide_bgr is not None and cut_start <= f_idx <= cut_end:
            slide_frame = cutaway_slide_bgr.copy()
            prog = (f_idx - cut_start) / float(max(1, cut_end - cut_start))
            zoom = 1.0 + 0.045 * prog
            ix1, iy1, ix2, iy2 = 1078, 196, 1840, 892
            iw, ih = ix2 - ix1, iy2 - iy1
            sub_img = slide_frame[iy1:iy2, ix1:ix2]
            M_sub = np.float32([
                [zoom, 0.0, (1.0 - zoom) * (iw * 0.5)],
                [0.0, zoom, (1.0 - zoom) * (ih * 0.5)]
            ])
            slide_frame[iy1:iy2, ix1:ix2] = cv2.warpAffine(sub_img, M_sub, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            writer.write(slide_frame)
            continue

        frame_presenter = presenter_bgr.copy()
        energy = frame_energies[f_idx]

        if f_idx < len(predicted_faces):
            pred_resized = cv2.resize(predicted_faces[f_idx], (fw, fh), interpolation=cv2.INTER_CUBIC)
            blended_face = (
                pred_resized.astype(np.float32) * feather_mask
                + face_crop_orig.astype(np.float32) * (1.0 - feather_mask)
            ).astype(np.uint8)
            frame_presenter[y1:y2, x1:x2] = blended_face
        elif energy > 0.05:
            # Smooth jaw/mouth articulation fallback if checkpoint wasn't used
            jaw_y1 = y1 + int(fh * 0.58)
            jaw_h = y2 - jaw_y1
            if jaw_h > 8:
                stretch = int(min(10, energy * 9.0))
                jaw_patch = cv2.resize(face_crop_orig[int(fh * 0.58):, :], (fw, jaw_h + stretch))
                frame_presenter[jaw_y1:y2, x1:x2] = jaw_patch[:jaw_h, :]

        # LivePortrait-style natural breathing + subtle speech-driven head sway
        t_sec = f_idx / float(fps)
        scale = 1.012 + 0.006 * math.sin(2.0 * math.pi * 0.35 * t_sec) + 0.004 * energy
        tx = 2.5 * math.sin(2.0 * math.pi * 0.22 * t_sec)
        ty = 2.0 * math.cos(2.0 * math.pi * 0.40 * t_sec) - 2.2 * energy
        M = np.float32([
            [scale, 0.0, (1.0 - scale) * (presenter_w * 0.5) + tx],
            [0.0, scale, (1.0 - scale) * (presenter_h * 0.45) + ty]
        ])
        animated_presenter = cv2.warpAffine(
            frame_presenter, M, (presenter_w, presenter_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

        full_frame = studio_bg_bgr.copy()
        comp_slot = (animated_presenter.astype(np.float32) * card_alpha + base_slot * (1.0 - card_alpha)).astype(np.uint8)
        full_frame[88:88 + presenter_h, 64:64 + presenter_w] = comp_slot
        writer.write(full_frame)

    writer.release()

    # Mux with high-quality AAC audio and H.264 video
    mux_cmd = [
        "ffmpeg", "-y",
        "-i", silent_mp4,
        "-i", audio_wav_path,
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
        "-shortest",
        output_mp4_path
    ]
    if run_ffmpeg_fn:
        run_ffmpeg_fn(mux_cmd, desc="Mux lip-synced teacher video with audio")
    else:
        import subprocess
        subprocess.run(mux_cmd, check=True)

    if os.path.exists(silent_mp4):
        os.remove(silent_mp4)
    return output_mp4_path


# ==============================================================================
# 4. POWERPOINT SLIDE RENDERER & PROCEDURAL INFOGRAPHIC FALLBACK
# ==============================================================================

def generate_fallback_illustration(prompt_text, title_text, output_png, variant=0):
    """
    Generates a clean, high-resolution (1280x720) educational visual illustration card
    whenever Cloudflare Workers AI token is not yet supplied or if the API is unreachable.
    """
    W, H = 1280, 720
    palettes = [
        ((15, 23, 42), (30, 64, 175), (56, 189, 248)),
        ((20, 83, 45), (6, 95, 70), (52, 211, 153)),
        ((49, 46, 129), (88, 28, 135), (192, 132, 252)),
        ((124, 45, 18), (154, 52, 18), (251, 146, 60)),
    ]
    c1, c2, accent = palettes[variant % len(palettes)]
    img = Image.new("RGB", (W, H), c1)
    draw = ImageDraw.Draw(img, "RGBA")

    # Decorative geometric studio circles
    draw.ellipse([W - 420, -120, W + 180, 480], fill=c2 + (140,))
    draw.ellipse([-160, H - 380, 420, H + 200], fill=accent + (55,))
    draw.rounded_rectangle([70, 70, W - 70, H - 70], radius=28, fill=(15, 23, 42, 195), outline=accent + (210,), width=3)

    font_tag = _load_font(22, bold=True)
    font_title = _load_font(44, bold=True)
    font_body = _load_font(28, bold=False)

    draw.text((120, 115), "KEY VISUAL SUMMARY", font=font_tag, fill=accent + (255,))
    t_lines = _wrap_text(draw, title_text or "Core Lesson Concept", font_title, W - 240)
    y = 175
    for line in t_lines[:2]:
        draw.text((120, y), line, font=font_title, fill=(255, 255, 255, 255))
        y += 58

    draw.line([120, y + 15, W - 120, y + 15], fill=accent + (160,), width=2)
    y += 45
    p_lines = _wrap_text(draw, prompt_text or "", font_body, W - 240)
    for line in p_lines[:5]:
        draw.text((120, y), line, font=font_body, fill=(203, 213, 225, 255))
        y += 42

    img.save(output_png, "PNG")
    return output_png


def render_powerpoint_slide_frame(
    lesson_title,
    slide_title,
    slide_notes,
    cf_image_path,
    raw_avatar_path,
    output_png,
    block_idx=0,
    total_blocks=1,
    reveal_all=True
):
    """
    Renders a crisp 1920x1080 PowerPoint Presentation Slide with:
      - Top Navy/Cyan Slide Header Bar + Circular Teacher Avatar Badge
      - Left Column: Easy-to-read numbered/bulleted summarized notes
      - Right Column: High-resolution Cloudflare Workers AI generated illustration
      - Bottom Reserved Band for Synchronized On-Screen Lyrics
    """
    W, H = 1920, 1080
    slide = Image.new("RGB", (W, H), (248, 250, 252))
    draw = ImageDraw.Draw(slide, "RGBA")

    # Top PowerPoint Header Banner (y=0..148)
    draw.rectangle([0, 0, W, 148], fill=(15, 23, 42, 255))
    draw.rectangle([0, 142, W, 148], fill=(56, 189, 248, 255))

    font_course = _load_font(21, bold=True)
    font_heading = _load_font(42, bold=True)
    draw.text((72, 24), f"{lesson_title.upper()[:70]}  •  SLIDE {block_idx + 1} OF {total_blocks}", font=font_course, fill=(56, 189, 248, 255))
    draw.text((72, 64), (slide_title or lesson_title)[:52], font=font_heading, fill=(255, 255, 255, 255))

    # Circular Teacher Avatar Badge in top-right header
    if raw_avatar_path and os.path.exists(raw_avatar_path):
        try:
            pres_bgr, _, _ = prepare_presenter_portrait(raw_avatar_path, 160, 160)
            pres_pil = Image.fromarray(cv2.cvtColor(pres_bgr, cv2.COLOR_BGR2RGB)).resize((104, 104), Image.Resampling.LANCZOS)
            mask = Image.new("L", (104, 104), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, 104, 104], fill=255)
            draw.ellipse([1744, 18, 1856, 130], fill=(56, 189, 248, 255))
            slide.paste(pres_pil, (1748, 22), mask)
        except Exception:
            pass

    # Left Card: Summarized Bullet Points (x=64..1030, y=180..908)
    draw.rounded_rectangle([64, 180, 1030, 908], radius=24, fill=(255, 255, 255, 255), outline=(226, 232, 240, 255), width=3)
    font_sub = _load_font(24, bold=True)
    draw.text((104, 212), "SUMMARY NOTES & KEY TAKEAWAYS", font=font_sub, fill=(37, 99, 235, 255))
    draw.line([104, 252, 990, 252], fill=(226, 232, 240, 255), width=2)

    notes_list = list(slide_notes or [])
    if not reveal_all and len(notes_list) > 1:
        notes_list = notes_list[:max(1, (len(notes_list) + 1) // 2)]

    font_bullet = _load_font(34, bold=False)
    font_num = _load_font(24, bold=True)
    curr_y = 282
    badge_colors = [(37, 99, 235), (16, 185, 129), (245, 158, 11), (139, 92, 246)]

    for i, note in enumerate(notes_list[:4]):
        clean = str(note).strip()
        if not clean:
            continue
        b_col = badge_colors[i % len(badge_colors)]
        draw.rounded_rectangle([104, curr_y + 4, 148, curr_y + 48], radius=12, fill=b_col + (255,))
        draw.text((119, curr_y + 11), str(i + 1), font=font_num, fill=(255, 255, 255, 255))

        wrapped = _wrap_text(draw, clean, font_bullet, 810)
        for line in wrapped[:3]:
            draw.text((168, curr_y + 2), line, font=font_bullet, fill=(15, 23, 42, 255))
            curr_y += 46
        curr_y += 28
        if curr_y > 855:
            break

    # Right Card: Cloudflare Workers AI Illustration (x=1062..1856, y=180..908)
    draw.rounded_rectangle([1062, 180, 1856, 908], radius=24, fill=(15, 23, 42, 255), outline=(203, 213, 225, 255), width=3)
    if cf_image_path and os.path.exists(cf_image_path):
        try:
            ill = Image.open(cf_image_path).convert("RGB")
            # Cover-fit 762x696
            target_w, target_h = 762, 696
            iw, ih = ill.size
            scale = max(target_w / float(iw), target_h / float(ih))
            nw, nh = int(iw * scale), int(ih * scale)
            ill = ill.resize((nw, nh), Image.Resampling.LANCZOS)
            left = (nw - target_w) // 2
            top = (nh - target_h) // 2
            ill = ill.crop((left, top, left + target_w, top + target_h))

            mask = Image.new("L", (target_w, target_h), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, target_w, target_h], radius=18, fill=255)
            slide.paste(ill, (1078, 196), mask)
        except Exception as e:
            print(f"Slide image paste note: {e}")

    # Bottom Lyrics Bar (y=928..1060)
    draw.rounded_rectangle([64, 932, 1856, 1056], radius=18, fill=(15, 23, 42, 245), outline=(56, 189, 248, 180), width=2)

    slide.save(output_png, "PNG")
    return output_png


def _get_audio_duration_sec(audio_wav_path):
    try:
        import soundfile as sf
        info = sf.info(audio_wav_path)
        return max(1.0, float(info.duration))
    except Exception:
        wav, _ = librosa.load(audio_wav_path, sr=16000)
        return max(1.0, len(wav) / 16000.0)


def render_slides_block(
    lesson_title,
    slide_title,
    slide_notes,
    cf_image_1,
    cf_image_2,
    raw_avatar_path,
    audio_wav_path,
    output_mp4_path,
    block_idx=0,
    total_blocks=1,
    fps=24,
    run_ffmpeg_fn=None
):
    """
    Renders a PowerPoint Slide Deck block (1920x1080 @ 24fps) where:
      - First half displays progressive bullets (Part 1) + Illustration 1
      - Second half crossfades into complete bullets (Part 2) + Illustration 2
      - Subtle dynamic motion on the visual illustration keeps the screen alive
    """
    dur = _get_audio_duration_sec(audio_wav_path)
    num_frames = max(24, int(dur * fps))
    mid_frame = num_frames // 2
    fade_frames = min(15, max(4, num_frames // 10))

    p1_png = f"{output_mp4_path}.slide1.png"
    p2_png = f"{output_mp4_path}.slide2.png"

    render_powerpoint_slide_frame(
        lesson_title=lesson_title,
        slide_title=slide_title,
        slide_notes=slide_notes,
        cf_image_path=cf_image_1,
        raw_avatar_path=raw_avatar_path,
        output_png=p1_png,
        block_idx=block_idx,
        total_blocks=total_blocks,
        reveal_all=False
    )
    render_powerpoint_slide_frame(
        lesson_title=lesson_title,
        slide_title=slide_title,
        slide_notes=slide_notes,
        cf_image_path=cf_image_2 or cf_image_1,
        raw_avatar_path=raw_avatar_path,
        output_png=p2_png,
        block_idx=block_idx,
        total_blocks=total_blocks,
        reveal_all=True
    )

    frame1_bgr = cv2.imread(p1_bgr_path := p1_png)
    frame2_bgr = cv2.imread(p2_bgr_path := p2_png)

    silent_mp4 = f"{output_mp4_path}.silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_mp4, fourcc, float(fps), (1920, 1080))

    # Region of the right-side illustration card (x=1078..1840, y=196..892)
    ix1, iy1, ix2, iy2 = 1078, 196, 1840, 892
    iw, ih = ix2 - ix1, iy2 - iy1

    for f_idx in range(num_frames):
        if f_idx < mid_frame - fade_frames // 2:
            base = frame1_bgr.copy()
            progress = f_idx / float(max(1, mid_frame))
        elif f_idx < mid_frame + fade_frames // 2:
            alpha = (f_idx - (mid_frame - fade_frames // 2)) / float(max(1, fade_frames))
            alpha = max(0.0, min(1.0, alpha))
            base = cv2.addWeighted(frame2_bgr, alpha, frame1_bgr, 1.0 - alpha, 0.0)
            progress = 0.5
        else:
            base = frame2_bgr.copy()
            progress = (f_idx - mid_frame) / float(max(1, num_frames - mid_frame))

        # Subtle Ken Burns zoom on the embedded illustration region
        zoom = 1.0 + 0.045 * progress
        sub_img = base[iy1:iy2, ix1:ix2]
        M = np.float32([
            [zoom, 0.0, (1.0 - zoom) * (iw * 0.5)],
            [0.0, zoom, (1.0 - zoom) * (ih * 0.5)]
        ])
        base[iy1:iy2, ix1:ix2] = cv2.warpAffine(sub_img, M, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        writer.write(base)

    writer.release()

    mux_cmd = [
        "ffmpeg", "-y",
        "-i", silent_mp4,
        "-i", audio_wav_path,
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
        "-shortest",
        output_mp4_path
    ]
    if run_ffmpeg_fn:
        run_ffmpeg_fn(mux_cmd, desc="Mux PowerPoint slide block with audio")
    else:
        import subprocess
        subprocess.run(mux_cmd, check=True)

    for tmp in (silent_mp4, p1_png, p2_png):
        if os.path.exists(tmp):
            os.remove(tmp)

    return output_mp4_path


def _prepare_broll_canvas(img_path, W=1920, H=1080):
    if img_path and os.path.exists(img_path):
        pil_img = Image.open(img_path).convert("RGB")
        iw, ih = pil_img.size
        scale = max(W / float(iw), H / float(ih))
        nw, nh = int(iw * scale), int(ih * scale)
        pil_img = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
        left = (nw - W) // 2
        top = (nh - H) // 2
        pil_img = pil_img.crop((left, top, left + W, top + H))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return np.full((H, W, 3), (30, 41, 59), dtype=np.uint8)


def _build_broll_overlay_rgba(lesson_title, slide_title, note_text, raw_avatar_path, block_idx, total_blocks):
    W, H = 1920, 1080
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov, "RGBA")

    # Top floating studio pill header
    draw.rounded_rectangle([64, 28, 1856, 118], radius=22, fill=(15, 23, 42, 215), outline=(56, 189, 248, 190), width=2)
    font_tag = _load_font(20, bold=True)
    font_head = _load_font(34, bold=True)
    draw.text((96, 40), f"VISUAL CONCEPT  •  SECTION {block_idx + 1} OF {total_blocks}", font=font_tag, fill=(56, 189, 248, 255))
    draw.text((96, 68), (slide_title or lesson_title)[:55], font=font_head, fill=(255, 255, 255, 255))

    # Circular Teacher Badge on top right
    if raw_avatar_path and os.path.exists(raw_avatar_path):
        try:
            pres_bgr, _, _ = prepare_presenter_portrait(raw_avatar_path, 140, 140)
            pres_pil = Image.fromarray(cv2.cvtColor(pres_bgr, cv2.COLOR_BGR2RGB)).resize((76, 76), Image.Resampling.LANCZOS)
            mask = Image.new("L", (76, 76), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, 76, 76], fill=255)
            draw.ellipse([1754, 33, 1838, 117], fill=(56, 189, 248, 255))
            ov.paste(pres_pil, (1758, 37), mask)
        except Exception:
            pass

    # Floating Key Note Callout Card on lower-left (above lyrics bar)
    if note_text:
        draw.rounded_rectangle([64, 730, 1180, 905], radius=20, fill=(15, 23, 42, 220), outline=(255, 255, 255, 150), width=2)
        draw.rounded_rectangle([64, 730, 82, 905], radius=8, fill=(56, 189, 248, 255))
        font_note = _load_font(30, bold=True)
        lines = _wrap_text(draw, note_text, font_note, 1050)
        y = 758
        for line in lines[:3]:
            draw.text((106, y), line, font=font_note, fill=(248, 250, 252, 255))
            y += 42

    # Bottom Lyrics Bar (y=932..1056)
    draw.rounded_rectangle([64, 932, 1856, 1056], radius=18, fill=(15, 23, 42, 235), outline=(56, 189, 248, 190), width=2)

    ov_np = np.array(ov)
    rgb = ov_np[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    alpha = (ov_np[:, :, 3].astype(np.float32) / 255.0)[..., None]
    return bgr, alpha


def render_broll_motion_block(
    lesson_title,
    slide_title,
    slide_notes,
    cf_image_1,
    cf_image_2,
    raw_avatar_path,
    audio_wav_path,
    output_mp4_path,
    block_idx=0,
    total_blocks=1,
    fps=24,
    run_ffmpeg_fn=None
):
    """
    Renders a dynamic 30s B-Roll Visual Motion Clip (1920x1080 @ 24fps):
      - Smooth 24fps Ken Burns cinematic pan & zoom across two contextual AI visuals
      - Smooth crossfade transition at midpoint
      - Glassmorphic concept callout card + bottom synchronized lyric bar
    """
    dur = _get_audio_duration_sec(audio_wav_path)
    num_frames = max(24, int(dur * fps))
    mid_frame = num_frames // 2
    fade_frames = min(18, max(6, num_frames // 8))

    canvas1 = _prepare_broll_canvas(cf_image_1, 1920, 1080)
    canvas2 = _prepare_broll_canvas(cf_image_2 or cf_image_1, 1920, 1080)

    notes_list = [str(n).strip() for n in (slide_notes or []) if str(n).strip()]
    note1 = notes_list[0] if len(notes_list) > 0 else slide_title
    note2 = notes_list[1] if len(notes_list) > 1 else note1

    ov1_bgr, ov1_a = _build_broll_overlay_rgba(lesson_title, slide_title, note1, raw_avatar_path, block_idx, total_blocks)
    ov2_bgr, ov2_a = _build_broll_overlay_rgba(lesson_title, slide_title, note2, raw_avatar_path, block_idx, total_blocks)

    silent_mp4 = f"{output_mp4_path}.silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_mp4, fourcc, float(fps), (1920, 1080))

    W, H = 1920, 1080
    for f_idx in range(num_frames):
        p1 = min(1.0, f_idx / float(max(1, mid_frame)))
        z1 = 1.02 + 0.10 * p1
        tx1 = -18.0 * p1
        ty1 = -10.0 * p1
        M1 = np.float32([
            [z1, 0.0, (1.0 - z1) * (W * 0.5) + tx1],
            [0.0, z1, (1.0 - z1) * (H * 0.5) + ty1]
        ])

        p2 = max(0.0, (f_idx - mid_frame + fade_frames) / float(max(1, num_frames - mid_frame + fade_frames)))
        z2 = 1.12 - 0.09 * min(1.0, p2)
        tx2 = 16.0 * (1.0 - min(1.0, p2))
        ty2 = 8.0 * (1.0 - min(1.0, p2))
        M2 = np.float32([
            [z2, 0.0, (1.0 - z2) * (W * 0.5) + tx2],
            [0.0, z2, (1.0 - z2) * (H * 0.5) + ty2]
        ])

        if f_idx < mid_frame - fade_frames // 2:
            bg = cv2.warpAffine(canvas1, M1, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            comp = (bg.astype(np.float32) * (1.0 - ov1_a) + ov1_bgr * ov1_a).astype(np.uint8)
        elif f_idx < mid_frame + fade_frames // 2:
            alpha = (f_idx - (mid_frame - fade_frames // 2)) / float(max(1, fade_frames))
            alpha = max(0.0, min(1.0, alpha))
            bg1 = cv2.warpAffine(canvas1, M1, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            bg2 = cv2.warpAffine(canvas2, M2, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            bg = cv2.addWeighted(bg2, alpha, bg1, 1.0 - alpha, 0.0)
            comp = (bg.astype(np.float32) * (1.0 - ov2_a) + ov2_bgr * ov2_a).astype(np.uint8)
        else:
            bg = cv2.warpAffine(canvas2, M2, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            comp = (bg.astype(np.float32) * (1.0 - ov2_a) + ov2_bgr * ov2_a).astype(np.uint8)

        writer.write(comp)

    writer.release()

    mux_cmd = [
        "ffmpeg", "-y",
        "-i", silent_mp4,
        "-i", audio_wav_path,
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
        "-shortest",
        output_mp4_path
    ]
    if run_ffmpeg_fn:
        run_ffmpeg_fn(mux_cmd, desc="Mux B-roll motion clip with audio")
    else:
        import subprocess
        subprocess.run(mux_cmd, check=True)

    if os.path.exists(silent_mp4):
        os.remove(silent_mp4)

    return output_mp4_path

