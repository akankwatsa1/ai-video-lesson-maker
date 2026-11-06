import os
import sys
import gc
import math
import hashlib
import threading
import cv2
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import librosa
from scipy import signal
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import re
import subprocess
import shutil
import json
import time

# ==============================================================================
# 1. S3FD NEURAL FACE DETECTOR (OFFICIAL WAV2LIP DETECTOR)
# ==============================================================================

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale=1.0):
        super().__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.weight.data *= 0.0
        self.weight.data += self.scale

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        return x / norm * self.weight.view(1, -1, 1, 1)


class S3FDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv3_3_norm = L2Norm(256, scale=10)
        self.conv4_3_norm = L2Norm(512, scale=8)
        self.conv5_3_norm = L2Norm(512, scale=5)

        self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv3_3_norm_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

        self.fc7_mbox_conf = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_loc = nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_conf = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        f3_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        f4_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        f5_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        ffc7 = h
        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h))
        f6_2 = h
        h = F.relu(self.conv7_1(h))
        h = F.relu(self.conv7_2(h))
        f7_2 = h

        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)

        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)
        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        cls4 = self.fc7_mbox_conf(ffc7)
        reg4 = self.fc7_mbox_loc(ffc7)
        cls5 = self.conv6_2_mbox_conf(f6_2)
        reg5 = self.conv6_2_mbox_loc(f6_2)
        cls6 = self.conv7_2_mbox_conf(f7_2)
        reg6 = self.conv7_2_mbox_loc(f7_2)

        chunk = torch.chunk(cls1, 4, 1)
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls1 = torch.cat([bmax, chunk[3]], dim=1)

        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]


_S3FD_MODEL = None
S3FD_CHECKPOINT_PATHS = [
    "/app/checkpoints/s3fd-619a316812.pth",
    "/app/s3fd-619a316812.pth",
]


def get_s3fd_detector(device="cuda"):
    global _S3FD_MODEL
    if _S3FD_MODEL is not None:
        return _S3FD_MODEL

    ckpt_path = None
    for p in S3FD_CHECKPOINT_PATHS:
        if os.path.exists(p) and os.path.getsize(p) > 10_000_000:
            ckpt_path = p
            break

    if not ckpt_path:
        try:
            from huggingface_hub import hf_hub_download
            print("Downloading s3fd-619a316812.pth neural face detector from HuggingFace...")
            ckpt_path = hf_hub_download(
                repo_id="camenduru/Wav2Lip",
                filename="checkpoints/s3fd-619a316812.pth",
                local_dir="/app"
            )
        except Exception as e:
            print(f"S3FD download note ({e}); falling back to OpenCV cascade.")
            return None

    try:
        net = S3FDNet()
        weights = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(weights)
        net = net.to(device).eval()
        _S3FD_MODEL = net
        print("S3FD neural face detector loaded successfully!")
        return _S3FD_MODEL
    except Exception as e:
        print(f"S3FD load note ({e}); falling back to OpenCV cascade.")
        return None


def _s3fd_decode_box(loc, priors, variances=(0.1, 0.2)):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
    ), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2.0
    boxes[:, 2:] += boxes[:, :2]
    return boxes


# ==============================================================================
# 2. WAV2LIP NEURAL ARCHITECTURE & MEL-SPECTROGRAM PROCESSOR
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
            print(f"Wav2Lip checkpoint download note ({e}); using jaw/lip articulation.")
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
        print(f"Wav2Lip load warning ({e}); falling back to syllable lip articulation.")
        return None


# ==============================================================================
# 3. S3FD + CASCADE FACE DETECTION & SMART PHONE PHOTO CROPPING
# ==============================================================================

def detect_face_bbox(img_bgr, device="cuda"):
    """
    Uses Wav2Lip's official S3FD neural detector (with pady2=10 chin padding)
    so the mouth and jaw sit at the exact coordinates Wav2Lip-GAN expects.
    """
    h, w = img_bgr.shape[:2]

    # 1. Try S3FD neural face detector first
    s3fd_net = get_s3fd_detector(device=device)
    if s3fd_net is not None:
        try:
            scale = min(1.0, 640.0 / float(max(h, w)))
            det_w, det_h = max(64, int(w * scale)), max(64, int(h * scale))
            det_img = cv2.resize(img_bgr, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
            inp = det_img.astype(np.float32) - np.array([104.0, 117.0, 123.0], dtype=np.float32)
            inp = inp.transpose(2, 0, 1)[None, ...]
            tensor_inp = torch.from_numpy(inp).float().to(device)

            with torch.no_grad():
                olist = s3fd_net(tensor_inp)

            bboxlist = []
            for i in range(len(olist) // 2):
                olist[i * 2] = F.softmax(olist[i * 2], dim=1)
            olist = [o.data.cpu() for o in olist]
            for i in range(len(olist) // 2):
                ocls, oreg = olist[i * 2], olist[i * 2 + 1]
                stride = 2 ** (i + 2)
                poss = zip(*np.where(ocls[:, 1, :, :] > 0.35))
                for _, hindex, windex in poss:
                    axc, ayc = stride / 2.0 + windex * stride, stride / 2.0 + hindex * stride
                    score = float(ocls[0, 1, hindex, windex])
                    loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
                    priors = torch.Tensor([[axc, ayc, stride * 4.0, stride * 4.0]])
                    box = _s3fd_decode_box(loc, priors)
                    bx1, by1, bx2, by2 = box[0].numpy()
                    bboxlist.append([bx1 / scale, by1 / scale, bx2 / scale, by2 / scale, score])

            if len(bboxlist) > 0:
                # Pick highest-confidence large face
                best = max(bboxlist, key=lambda b: b[4] * max(1.0, (b[2] - b[0]) * (b[3] - b[1])))
                x1, y1, x2, y2, _ = best
                fh = y2 - y1
                # Official Wav2Lip padding: pady2 = 10 (scaled to face height) so chin movement is never clipped
                pad_bottom = max(10, int(fh * 0.08))
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2 + pad_bottom))
                if (x2 - x1) > 36 and (y2 - y1) > 36:
                    return (x1, y1, x2, y2), True
        except Exception as e:
            print(f"S3FD inference note ({e}); using OpenCV cascade.")

    # 2. Fallback to OpenCV Haar cascades
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    for xml_name in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"):
        cascade_path = os.path.join(cv2.data.haarcascades, xml_name)
        if os.path.exists(cascade_path):
            detector = cv2.CascadeClassifier(cascade_path)
            faces = detector.detectMultiScale(gray, scaleFactor=1.06, minNeighbors=3, minSize=(48, 48))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                x1 = max(0, int(fx - fw * 0.04))
                x2 = min(w, int(fx + fw * 1.04))
                y1 = max(0, int(fy - fh * 0.02))
                y2 = min(h, int(fy + fh * 1.16))
                return (x1, y1, x2, y2), True

    # 3. Portrait heuristic fallback
    cx, cy = w // 2, int(h * 0.38)
    fw, fh = int(min(w, h) * 0.44), int(min(w, h) * 0.50)
    return (max(0, cx - fw // 2), max(0, cy - fh // 2), min(w, cx + fw // 2), min(h, cy + fh // 2)), False


def _is_video_file_path(path):
    if not path:
        return False
    ext = os.path.splitext(str(path).lower())[1]
    return ext in (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v")


def _read_first_avatar_frame_bgr(raw_avatar_path):
    """
    Reads the first valid BGR frame from either an image file (.jpg/.png/.webp)
    or a short video avatar clip (.mp4/.webm/.mov).
    """
    if not raw_avatar_path or not os.path.exists(raw_avatar_path):
        return None
    if not _is_video_file_path(raw_avatar_path):
        img = cv2.imread(raw_avatar_path)
        if img is not None:
            return img
        try:
            pil_img = Image.open(raw_avatar_path).convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    try:
        cap = cv2.VideoCapture(raw_avatar_path)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return frame
    except Exception:
        pass
    return None


def _polish_avatar_frame_bgr(frame_bgr):
    """
    Removes grainy/rice-like sensor noise and block compression artifacts from
    uploaded video loop clips or portrait photos while keeping eyes, hair, and facial
    contours crisp and broadcast-polished.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    try:
        smoothed = cv2.bilateralFilter(frame_bgr, d=5, sigmaColor=22, sigmaSpace=22)
        blended = cv2.addWeighted(frame_bgr, 0.35, smoothed, 0.65, 0.0)
        blur_fine = cv2.GaussianBlur(blended, (0, 0), sigmaX=1.0)
        polished = cv2.addWeighted(blended, 1.18, blur_fine, -0.18, 0.0)
        return polished
    except Exception:
        return frame_bgr


def _pad_frame_with_studio_backdrop(img_bgr, pad_top=0, pad_bot=0, pad_left=0, pad_right=0):
    """
    Smoothly extends an image/video frame when the speaker's head is too close to the top
    edge or the clip is narrower than the target aspect ratio, using a softly blurred
    extension of the frame's own ambient background so the head is never cut off.
    """
    if pad_top <= 0 and pad_bot <= 0 and pad_left <= 0 and pad_right <= 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    new_h = h + pad_top + pad_bot
    new_w = w + pad_left + pad_right
    try:
        small = cv2.resize(img_bgr, (max(16, new_w // 12), max(16, new_h // 12)), interpolation=cv2.INTER_LINEAR)
        small = cv2.GaussianBlur(small, (15, 15), 0)
        ambient = cv2.resize(small, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        ambient = (ambient.astype(np.float32) * 0.78).astype(np.uint8)
        reflected = cv2.copyMakeBorder(
            img_bgr, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_REFLECT_101
        )
        reflected_blur = cv2.GaussianBlur(reflected, (31, 31), 0)
        canvas = cv2.addWeighted(reflected_blur, 0.65, ambient, 0.35, 0.0)
        canvas[pad_top:pad_top + h, pad_left:pad_left + w] = img_bgr
        # Feather 10px seam along padded edges if any
        if pad_top > 0 and h > 12:
            for py in range(min(10, h)):
                a = py / 10.0
                row_y = pad_top + py
                canvas[row_y, pad_left:pad_left + w] = (
                    img_bgr[py].astype(np.float32) * a
                    + canvas[row_y, pad_left:pad_left + w].astype(np.float32) * (1.0 - a)
                ).astype(np.uint8)
        return canvas
    except Exception:
        return cv2.copyMakeBorder(img_bgr, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_REPLICATE)


def _compute_balanced_crop_spec(img_bgr, target_w=680, target_h=820, device="cuda", bbox_override=None):
    """
    Computes a balanced Chest-Up Broadcast Portrait crop specification:
      (pad_top, pad_bot, pad_left, pad_right, x0, y0, crop_w, crop_h)
    Guarantees generous crown/hair headroom above the face (`0.60 * face_h` above `fy1`)
    so the top of the avatar's head is NEVER cut off on either close-up or vertical clips.
    """
    h, w = img_bgr.shape[:2]
    if bbox_override is not None:
        (fx1, fy1, fx2, fy2), found = bbox_override, True
    else:
        (fx1, fy1, fx2, fy2), found = detect_face_bbox(img_bgr, device=device)

    face_cx = (fx1 + fx2) // 2
    face_h = max(48, fy2 - fy1)
    target_aspect = target_w / float(target_h)

    if not found:
        # Safe top-biased crop when no face detector hit
        img_aspect = w / float(max(1, h))
        if img_aspect > target_aspect:
            crop_h = h
            crop_w = max(1, int(h * target_aspect))
            x0 = max(0, (w - crop_w) // 2)
            y0 = 0
        else:
            crop_w = w
            crop_h = max(1, int(w / target_aspect))
            x0 = 0
            y0 = max(0, min(h - crop_h, int(h * 0.04)))
        return 0, 0, 0, 0, x0, y0, crop_w, crop_h

    # Required top coordinate above face bbox so crown + hair + headroom are 100% visible
    head_crown_margin = int(face_h * 0.62)
    ideal_y0 = fy1 - head_crown_margin
    # Ideal chest-up height is ~2.55x face height (crown + face + neck + upper chest)
    ideal_crop_h = max(int(face_h * 2.55), int((fy2 - ideal_y0) * 1.52))
    ideal_crop_w = int(round(ideal_crop_h * target_aspect))

    pad_top = max(0, -ideal_y0)
    pad_bot = max(0, (ideal_y0 + ideal_crop_h) - h)

    ideal_x0 = face_cx - (ideal_crop_w // 2)
    pad_left = max(0, -ideal_x0)
    pad_right = max(0, (ideal_x0 + ideal_crop_w) - w)

    # Cap padding to reasonable bounds while strictly preserving top headroom
    max_pad_side = int(w * 0.38)
    if pad_left + pad_right > max_pad_side * 2:
        # Fit width to padded bounds and adjust crop_h proportionally while keeping top headroom
        avail_w = w + max_pad_side * 2
        pad_left = min(pad_left, max_pad_side)
        pad_right = min(pad_right, max_pad_side)
        ideal_crop_w = min(ideal_crop_w, w + pad_left + pad_right)
        ideal_crop_h = max(1, int(round(ideal_crop_w / target_aspect)))
        pad_bot = max(0, (ideal_y0 + ideal_crop_h) - h)

    new_w = w + pad_left + pad_right
    new_h = h + pad_top + pad_bot

    crop_w = min(new_w, max(1, ideal_crop_w))
    crop_h = min(new_h, max(1, int(round(crop_w / target_aspect))))
    crop_w = min(new_w, max(1, int(round(crop_h * target_aspect))))

    shifted_fx_center = face_cx + pad_left
    shifted_ideal_y0 = ideal_y0 + pad_top

    x0 = max(0, min(new_w - crop_w, shifted_fx_center - (crop_w // 2)))
    y0 = max(0, min(new_h - crop_h, shifted_ideal_y0))

    return pad_top, pad_bot, pad_left, pad_right, x0, y0, max(1, crop_w), max(1, crop_h)


def _apply_crop_spec_to_frame(img_bgr, crop_spec, target_w=680, target_h=820):
    pad_top, pad_bot, pad_left, pad_right, x0, y0, crop_w, crop_h = crop_spec
    padded = _pad_frame_with_studio_backdrop(img_bgr, pad_top, pad_bot, pad_left, pad_right)
    cropped = padded[y0:y0 + crop_h, x0:x0 + crop_w]
    if cropped.size == 0:
        cropped = padded
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return _polish_avatar_frame_bgr(resized)


def _compute_chest_up_crop_coords(img_bgr, target_w=680, target_h=820, device="cuda"):
    """
    Backward-compatible wrapper returning (x0, y0, crop_w, crop_h) inside img_bgr.
    """
    h, w = img_bgr.shape[:2]
    _, _, _, _, x0, y0, cw, ch = _compute_balanced_crop_spec(img_bgr, target_w=target_w, target_h=target_h, device=device)
    cw = min(w, cw)
    ch = min(h, ch)
    x0 = max(0, min(w - cw, x0))
    y0 = max(0, min(h - ch, y0))
    return x0, y0, max(1, cw), max(1, ch)


def prepare_presenter_portrait(raw_avatar_path, target_w=680, target_h=820, device="cuda"):
    """
    Takes any phone photo (.jpg/.png/.webp) OR short teacher video clip (.mp4/.webm/.mov),
    detects the teacher's face using S3FD, and smartly pads/crops/scales it into a clean,
    well-balanced Chest-Up Broadcast Portrait (680x820) with guaranteed top headroom so
    the head is never cut off.
    """
    img = _read_first_avatar_frame_bgr(raw_avatar_path)
    if img is None:
        canvas = np.full((target_h, target_w, 3), (45, 55, 75), dtype=np.uint8)
        return canvas, (target_w // 4, target_h // 4, 3 * target_w // 4, 3 * target_h // 4), False

    crop_spec = _compute_balanced_crop_spec(img, target_w=target_w, target_h=target_h, device=device)
    presenter_bgr = _apply_crop_spec_to_frame(img, crop_spec, target_w=target_w, target_h=target_h)

    bbox, found_final = detect_face_bbox(presenter_bgr, device=device)
    return presenter_bgr, bbox, found_final


def _extract_video_avatar_loop(raw_avatar_path, target_w=680, target_h=820, max_frames=220, device="cuda"):
    """
    If `raw_avatar_path` is a short video clip (.mp4/.webm/.mov), decodes up to `max_frames`,
    computes the multi-frame face bounding envelope across the clip so even upward head nods
    never cut off the top of the head, applies uniform balanced framing + polish to every frame,
    and constructs a seamless ping-pong loop (forward + reverse). Returns None for static images.
    """
    if not _is_video_file_path(raw_avatar_path) or not os.path.exists(raw_avatar_path):
        return None
    try:
        cap = cv2.VideoCapture(raw_avatar_path)
        if not cap.isOpened():
            return None
        raw_frames = []
        while len(raw_frames) < max_frames:
            ret, frm = cap.read()
            if not ret or frm is None:
                break
            raw_frames.append(frm)
        cap.release()

        if len(raw_frames) <= 1:
            return None

        # Sample up to 8 frames across the video loop to find the highest head position (min_fy1)
        # and full movement envelope so the head is never cut off at any point in the loop
        sample_step = max(1, len(raw_frames) // 8)
        detected_boxes = []
        for s_idx in range(0, len(raw_frames), sample_step):
            b, ok = detect_face_bbox(raw_frames[s_idx], device=device)
            if ok:
                detected_boxes.append(b)

        bbox_env = None
        if detected_boxes:
            min_fx1 = int(np.percentile([b[0] for b in detected_boxes], 15))
            min_fy1 = int(min(b[1] for b in detected_boxes))  # Highest head point in entire clip
            max_fx2 = int(np.percentile([b[2] for b in detected_boxes], 85))
            max_fy2 = int(np.percentile([b[3] for b in detected_boxes], 85))
            bbox_env = (min_fx1, min_fy1, max_fx2, max_fy2)

        crop_spec = _compute_balanced_crop_spec(
            raw_frames[0], target_w=target_w, target_h=target_h, device=device, bbox_override=bbox_env
        )
        frames = [
            _apply_crop_spec_to_frame(frm, crop_spec, target_w=target_w, target_h=target_h)
            for frm in raw_frames
        ]
        del raw_frames

        # Ping-pong seamless loop avoids hard jump cuts at loop boundaries
        if len(frames) > 4:
            frames = frames + frames[-2:0:-1]
        return frames
    except Exception as e:
        print(f"Video avatar loop note: {e}")
        return None


# ==============================================================================
# 4. ENHANCED FONTS, LIP-SYNC ARTICULATION & STUDIO COMPOSITING
# ==============================================================================

def _load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf" if bold else "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
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


def _create_rounded_mask(w, h, radius=22):
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w, h], radius=radius, fill=255)
    return np.array(mask)


def _create_full_opacity_mouth_mask(w, h):
    """
    Preserves 100% (alpha=1.0) of Wav2Lip's lower-face/mouth/jaw output across
    y=0.48*h .. 0.96*h and x=0.08*w .. 0.92*w, feathering only the outer boundary
    so lip movements are vivid and unmistakable without visible box seams.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    y_top = int(h * 0.46)
    y_bot = int(h * 0.96)
    x_left = int(w * 0.08)
    x_right = int(w * 0.92)
    cv2.rectangle(mask, (x_left, y_top), (x_right, y_bot), 1.0, -1)
    blur_k = max(5, (min(w, h) // 7) | 1)
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    # Boost core mouth region back to full 1.0 opacity
    mask = np.clip(mask * 1.35, 0.0, 1.0)
    return np.expand_dims(mask, axis=-1)


_BROW_MESH_CACHE = {}
_JAW_HORIZ_CACHE = {}


def _apply_syllable_mouth_and_blink(face_bgr, energy, f_idx, fps=24):
    """
    Reinforces natural jaw articulation on spoken syllables, adds speech-synced
    eyebrow micro-lift, and performs natural single & occasional double eye blinks
    for static portrait photos (without any artificial white teeth/rice patch).
    """
    fh, fw = face_bgr.shape[:2]
    out = face_bgr.copy()
    t_sec = f_idx / float(max(1, fps))

    # 1. Eyebrow Micro-Lift on Emphasized Syllables (y: 16% -> 27%, x: 16% -> 84%)
    if energy > 0.14 and fh > 48 and fw > 48:
        brow_lift_px = min(3.8, float(energy) * 3.2 + 0.7 * max(0.0, math.sin(t_sec * 6.2)))
        by1, by2 = max(2, int(fh * 0.16)), min(fh - 2, int(fh * 0.27))
        bx1, bx2 = max(2, int(fw * 0.16)), min(fw - 2, int(fw * 0.84))
        if by2 > by1 + 6 and bx2 > bx1 + 12:
            brow_roi = out[by1:by2, bx1:bx2]
            bh_r, bw_r = brow_roi.shape[:2]
            b_key = (bh_r, bw_r)
            if b_key not in _BROW_MESH_CACHE:
                grid_x, grid_y = np.meshgrid(np.arange(bw_r, dtype=np.float32), np.arange(bh_r, dtype=np.float32))
                ny = grid_y / float(max(1, bh_r))
                nx = (grid_x - bw_r * 0.5) / float(max(1, bw_r * 0.5))
                weight = (np.sin(ny * math.pi) * np.clip(1.0 - (nx ** 2), 0.0, 1.0)).astype(np.float32)
                _BROW_MESH_CACHE[b_key] = (grid_x, grid_y, weight)
            else:
                grid_x, grid_y, weight = _BROW_MESH_CACHE[b_key]
            map_y = np.clip(grid_y + weight * brow_lift_px, 0, bh_r - 1)
            out[by1:by2, bx1:bx2] = cv2.remap(brow_roi, grid_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # 2. Smooth lower-jaw vertical articulation (no synthetic teeth/cavity overlay)
    if energy > 0.08 and fh > 40 and fw > 40:
        mouth_cy = int(fh * 0.74)
        jaw_top = mouth_cy
        jaw_bot = int(fh * 0.96)
        jaw_h = jaw_bot - jaw_top
        if jaw_h > 8:
            shift_px = max(1, int(fh * 0.036 * energy))
            jaw_patch = out[jaw_top:jaw_bot, :]
            stretched = cv2.resize(jaw_patch, (fw, jaw_h + shift_px), interpolation=cv2.INTER_LINEAR)
            if fw not in _JAW_HORIZ_CACHE:
                x_grid = np.linspace(-1.0, 1.0, fw, dtype=np.float32)
                horiz_w = np.clip(1.0 - (x_grid ** 2) * 1.4, 0.0, 1.0)[None, :, None]
                _JAW_HORIZ_CACHE[fw] = (horiz_w, 1.0 - horiz_w)
            horiz_w, inv_horiz_w = _JAW_HORIZ_CACHE[fw]
            out[jaw_top:jaw_bot, :] = (
                stretched[:jaw_h, :].astype(np.float32) * horiz_w
                + jaw_patch.astype(np.float32) * inv_horiz_w
            ).astype(np.uint8)

    # 3. Natural Single & Occasional Double Eye-Blinking (every ~3.8 seconds)
    blink_cycle = (t_sec + 0.75) % 3.8
    blink_strength = 0.0
    if blink_cycle < 0.15:
        blink_strength = math.sin((blink_cycle / 0.15) * math.pi)
    elif (int((t_sec + 0.75) // 3.8) % 2 == 1) and (0.24 <= blink_cycle < 0.38):
        blink_strength = 0.88 * math.sin(((blink_cycle - 0.24) / 0.14) * math.pi)

    if blink_strength > 0.05 and fh > 48:
        ey1, ey2 = int(fh * 0.26), int(fh * 0.41)
        eh = ey2 - ey1
        if eh > 4:
            upper_lid = out[ey1:ey1 + max(2, eh // 2), :]
            lid_stretched = cv2.resize(upper_lid, (fw, eh), interpolation=cv2.INTER_LINEAR)
            b_alpha = min(0.88, blink_strength * 0.88)
            blended_eyes = cv2.addWeighted(lid_stretched, b_alpha, out[ey1:ey2, :], 1.0 - b_alpha, 0.0)
            lash_y = min(eh - 1, max(1, int(eh * 0.68 * blink_strength)))
            blended_eyes[lash_y, int(fw * 0.18):int(fw * 0.82)] = (
                blended_eyes[lash_y, int(fw * 0.18):int(fw * 0.82)].astype(np.float32) * 0.55
            ).astype(np.uint8)
            out[ey1:ey2, :] = blended_eyes

    return out


def _precompute_torso_articulation_grids(presenter_w, presenter_h, face_y2):
    """
    Pre-computes normalized spatial weight masks for below-the-neck articulation
    (Neck/Collar pivot, Independent Left & Right Shoulder Shrugs, and Chest Breathing)
    so per-frame `cv2.remap` runs in <0.5ms with zero per-frame grid allocations.
    """
    y_top = min(presenter_h - 48, max(int(presenter_h * 0.50), int(face_y2 - 14)))
    th = presenter_h - y_top
    tw = presenter_w
    if th < 24 or tw < 24:
        return None

    gx, gy = np.meshgrid(np.arange(tw, dtype=np.float32), np.arange(th, dtype=np.float32))
    nx = (gx - tw * 0.5) / float(max(1.0, tw * 0.5))  # -1.0 (left edge) .. +1.0 (right edge)
    ny = gy / float(max(1.0, th))                      # 0.0 (chin/neck) .. 1.0 (bottom)

    # Smooth zero-displacement taper at top boundary (ny=0) and outer edges so seams never appear
    edge_taper = np.sin(np.clip(ny * math.pi, 0.0, math.pi)) * np.clip(1.0 - (nx ** 4) * 0.65, 0.0, 1.0)
    top_lock = np.clip(ny / 0.16, 0.0, 1.0)

    # Left shoulder zone (nx in -0.85 .. -0.15, ny in 0.12 .. 0.68)
    left_sh = np.exp(-((nx + 0.46) ** 2) / 0.11) * np.exp(-((ny - 0.34) ** 2) / 0.08) * top_lock
    # Right shoulder zone (nx in +0.15 .. +0.85, ny in 0.12 .. 0.68)
    right_sh = np.exp(-((nx - 0.46) ** 2) / 0.11) * np.exp(-((ny - 0.34) ** 2) / 0.08) * top_lock
    # Center chest / ribcage breathing & posture sway zone
    chest_zone = np.exp(-(nx ** 2) / 0.32) * np.sin(np.clip(ny * math.pi, 0.0, math.pi)) * top_lock
    # Neck & collar horizontal follow zone (ny in 0.05 .. 0.38)
    neck_zone = np.exp(-(nx ** 2) / 0.18) * np.exp(-((ny - 0.18) ** 2) / 0.03) * top_lock

    return {
        "y_top": y_top,
        "gx": gx,
        "gy": gy,
        "nx": nx,
        "edge_taper": edge_taper.astype(np.float32),
        "left_sh": left_sh.astype(np.float32),
        "right_sh": right_sh.astype(np.float32),
        "chest_zone": chest_zone.astype(np.float32),
        "neck_zone": neck_zone.astype(np.float32),
    }


def _build_studio_desk_foreground(presenter_w=680, presenter_h=820):
    """
    Pre-renders a sleek, television-style Broadcast Studio Desk / Lectern foreground bar
    at the bottom of the presenter portrait (bottom 68px) with a subtle ambient shadow
    and studio accent trim—naturally masking the lower torso cutoff like a real TV anchor desk.
    """
    desk_h = 74
    ov = Image.new("RGBA", (presenter_w, desk_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov, "RGBA")

    # Soft upper drop-shadow gradient (y=0..18)
    for sy in range(18):
        a_sh = int(135 * (sy / 18.0) ** 1.5)
        draw.line([(0, sy), (presenter_w, sy)], fill=(6, 10, 22, a_sh))

    # Sleek studio desk surface (y=18..desk_h)
    draw.rectangle([0, 18, presenter_w, desk_h], fill=(12, 19, 36, 242))
    # Metallic/glowing top bevel rim on desk edge
    draw.rectangle([0, 16, presenter_w, 21], fill=(56, 189, 248, 225))
    draw.rectangle([0, 21, presenter_w, 24], fill=(250, 204, 21, 175))
    # Subtle studio desk center panel highlight
    draw.rounded_rectangle([presenter_w // 2 - 90, 32, presenter_w // 2 + 90, desk_h - 10], radius=8, fill=(22, 34, 62, 235), outline=(56, 189, 248, 140), width=2)

    ov_np = np.array(ov)
    bgr = cv2.cvtColor(ov_np[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    alpha = (ov_np[:, :, 3].astype(np.float32) / 255.0)[..., None]
    return bgr, alpha


_AVATAR_BASE_CACHE = {}


def _get_cached_avatar_base(raw_avatar_path, presenter_w=680, presenter_h=820, device="cuda"):
    """
    Caches S3FD face detection, chest-up portrait crop, video avatar loop frames,
    torso articulation grids, and studio desk overlay across blocks so a 10-minute
    (20-block) video only runs S3FD and video avatar decoding ONCE per job.
    """
    cache_key = f"v763|{raw_avatar_path}|{presenter_w}x{presenter_h}"
    if cache_key in _AVATAR_BASE_CACHE:
        return _AVATAR_BASE_CACHE[cache_key]

    video_loop_frames = _extract_video_avatar_loop(
        raw_avatar_path, presenter_w, presenter_h, device=device
    )
    if video_loop_frames and len(video_loop_frames) > 0:
        presenter_bgr = video_loop_frames[0].copy()
        (x1, y1, x2, y2), _ = detect_face_bbox(presenter_bgr, device=device)
    else:
        presenter_bgr, (x1, y1, x2, y2), _ = prepare_presenter_portrait(
            raw_avatar_path, presenter_w, presenter_h, device=device
        )
    face_crop_orig = presenter_bgr[y1:y2, x1:x2].copy()
    fh, fw = face_crop_orig.shape[:2]
    mouth_mask = _create_full_opacity_mouth_mask(fw, fh)
    torso_grids = None if video_loop_frames else _precompute_torso_articulation_grids(presenter_w, presenter_h, y2)
    desk_bgr, desk_alpha = _build_studio_desk_foreground(presenter_w, presenter_h)

    base_entry = {
        "presenter_bgr": presenter_bgr,
        "video_loop_frames": video_loop_frames,
        "bbox": (x1, y1, x2, y2),
        "face_crop_orig": face_crop_orig,
        "mouth_mask": mouth_mask,
        "torso_grids": torso_grids,
        "desk_bgr": desk_bgr,
        "desk_alpha": desk_alpha,
    }
    if len(_AVATAR_BASE_CACHE) >= 2:
        _AVATAR_BASE_CACHE.clear()
    _AVATAR_BASE_CACHE[cache_key] = base_entry
    return base_entry


def synthesize_talking_presenter_frames(raw_avatar_path, audio_wav_path, fps=24, device="cuda"):
    """
    Pre-renders the teacher portrait state (680x820) for a block.
    - When `raw_avatar_path` is a video loop clip (.mp4/.webm/.mov), plays the full balanced
      video loop naturally WITHOUT pasting a duplicate static mouth patch at fixed Frame-0
      coordinates!
    - When `raw_avatar_path` is a static photo (.jpg/.png/.webp), synthesizes Wav2Lip-GAN
      mouth motion + natural jaw/eyebrow/blink articulation.
    """
    presenter_w, presenter_h = 680, 820
    base = _get_cached_avatar_base(raw_avatar_path, presenter_w, presenter_h, device=device)
    presenter_bgr = base["presenter_bgr"]
    video_loop_frames = base["video_loop_frames"]
    x1, y1, x2, y2 = base["bbox"]
    face_crop_orig = base["face_crop_orig"]
    mouth_mask = base["mouth_mask"]
    torso_grids = base["torso_grids"]
    desk_bgr = base["desk_bgr"]
    desk_alpha = base["desk_alpha"]
    fh, fw = face_crop_orig.shape[:2]

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

    # Bandpass filter speech (300-3200Hz) for crisp syllable mouth envelope
    try:
        sos = signal.butter(4, [300.0 / 8000.0, 3200.0 / 8000.0], btype="bandpass", output="sos")
        vocal_band = signal.sosfilt(sos, wav_16k)
    except Exception:
        vocal_band = wav_16k

    samples_per_frame = max(1, len(vocal_band) // num_frames)
    frame_energies = []
    for f_idx in range(num_frames):
        seg = vocal_band[f_idx * samples_per_frame:(f_idx + 1) * samples_per_frame]
        rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) > 0 else 0.0
        frame_energies.append(min(1.0, rms * 11.5))

    if len(frame_energies) >= 3:
        smoothed = np.convolve(frame_energies, [0.25, 0.50, 0.25], mode="same")
        frame_energies = [float(x) for x in smoothed]

    predicted_faces = []
    is_video_loop = bool(video_loop_frames and len(video_loop_frames) > 1)

    # Only synthesize fixed-coordinate Wav2Lip mouth patches for static portrait photos;
    # for uploaded video loop clips, let the full video loop run naturally without a second floating mouth!
    if not is_video_loop:
        wav2lip_model = get_wav2lip_model(device=device)
        if wav2lip_model is not None and fw > 16 and fh > 16:
            face_96 = cv2.resize(face_crop_orig, (96, 96), interpolation=cv2.INTER_AREA)
            face_masked = face_96.copy()
            face_masked[48:, :] = 0
            face_concat = np.concatenate((face_masked, face_96), axis=2) / 255.0
            face_tensor_single = torch.from_numpy(np.transpose(face_concat, (2, 0, 1))).float().unsqueeze(0).to(device)

            stride = 2 if num_frames >= 192 else 1
            key_indices = list(range(0, num_frames, stride))
            key_mels = [mel_chunks[k] for k in key_indices]
            key_preds = []

            batch_size = 64
            use_cuda_amp = str(device).startswith("cuda") and torch.cuda.is_available()
            with torch.no_grad():
                for b_start in range(0, len(key_mels), batch_size):
                    b_mels = np.array(key_mels[b_start:b_start + batch_size])
                    b_len = len(b_mels)
                    b_mels = np.reshape(b_mels, (b_len, b_mels.shape[1], b_mels.shape[2], 1))
                    mel_tensor = torch.from_numpy(np.transpose(b_mels, (0, 3, 1, 2))).float().to(device)
                    img_tensor = face_tensor_single.repeat(b_len, 1, 1, 1)
                    if use_cuda_amp:
                        with torch.cuda.amp.autocast():
                            preds = wav2lip_model(mel_tensor, img_tensor)
                    else:
                        preds = wav2lip_model(mel_tensor, img_tensor)
                    preds_np = (preds.detach().cpu().float().numpy().transpose(0, 2, 3, 1) * 255.0).clip(0, 255).astype(np.uint8)
                    for p in preds_np:
                        key_preds.append(p)

            if stride == 1:
                predicted_faces = key_preds
            else:
                n_keys = len(key_preds)
                for f_idx in range(num_frames):
                    k0 = min(n_keys - 1, f_idx // stride)
                    if f_idx % stride == 0 or k0 + 1 >= n_keys:
                        predicted_faces.append(key_preds[k0])
                    else:
                        blended = cv2.addWeighted(key_preds[k0], 0.5, key_preds[k0 + 1], 0.5, 0)
                        predicted_faces.append(blended)

    return {
        "presenter_bgr": presenter_bgr,
        "video_loop_frames": video_loop_frames,
        "bbox": (x1, y1, x2, y2),
        "face_crop_orig": face_crop_orig,
        "mouth_mask": mouth_mask,
        "predicted_faces": predicted_faces,
        "frame_energies": frame_energies,
        "torso_grids": torso_grids,
        "desk_bgr": desk_bgr,
        "desk_alpha": desk_alpha,
        "num_frames": num_frames,
        "fps": fps,
        "_torso_step_cache": (-1, None),
        "_frame_cache": (-1, False, None),
    }


def get_animated_presenter_frame(talk_state, f_idx, camera_closeup=False):
    """
    Returns the single 680x820 animated presenter frame for frame index `f_idx`:
      1. If `video_loop_frames` is present (user uploaded a video clip avatar), plays the
         balanced, full-headroom video loop directly without any duplicate static mouth
         patch or head-clipping zoom!
      2. If static photo avatar, applies Wav2Lip-GAN mouth synthesis + syllable jaw/brow/blink
         + below-the-neck torso articulation + top-anchored posture sway (never cutting off head).
      3. Overlays the Broadcast Studio Desk foreground at the bottom 74px.
    """
    cached_f, cached_cu, cached_out = talk_state.get("_frame_cache", (-1, False, None))
    if cached_f == f_idx and cached_cu == camera_closeup and cached_out is not None:
        return cached_out

    presenter_bgr = talk_state["presenter_bgr"]
    video_loop_frames = talk_state.get("video_loop_frames")
    x1, y1, x2, y2 = talk_state["bbox"]
    face_crop_orig = talk_state["face_crop_orig"]
    mouth_mask = talk_state["mouth_mask"]
    predicted_faces = talk_state["predicted_faces"]
    frame_energies = talk_state["frame_energies"]
    torso_grids = talk_state.get("torso_grids")
    desk_bgr = talk_state.get("desk_bgr")
    desk_alpha = talk_state.get("desk_alpha")
    fps = talk_state["fps"]

    presenter_h, presenter_w = presenter_bgr.shape[:2]
    fh, fw = face_crop_orig.shape[:2]
    energy = frame_energies[min(f_idx, len(frame_energies) - 1)]
    t_sec = f_idx / float(max(1, fps))

    # Case A: Video Loop Avatar — play the full balanced video loop clip naturally
    # with ZERO static mouth overlay and ZERO head-chopping affine zoom!
    if video_loop_frames and len(video_loop_frames) > 1:
        animated_presenter = video_loop_frames[f_idx % len(video_loop_frames)].copy()
        if desk_bgr is not None and desk_alpha is not None:
            dh = desk_bgr.shape[0]
            bot_slice = animated_presenter[presenter_h - dh:, :].astype(np.float32)
            animated_presenter[presenter_h - dh:, :] = (
                bot_slice * (1.0 - desk_alpha) + desk_bgr * desk_alpha
            ).astype(np.uint8)
        talk_state["_frame_cache"] = (f_idx, camera_closeup, animated_presenter)
        return animated_presenter

    # Case B: Static Photo Avatar
    if energy < 0.03 and (f_idx % 2 == 1) and cached_f == (f_idx - 1) and cached_cu == camera_closeup and cached_out is not None:
        talk_state["_frame_cache"] = (f_idx, camera_closeup, cached_out)
        return cached_out

    frame_presenter = presenter_bgr.copy()
    base_face_for_frame = face_crop_orig

    if f_idx < len(predicted_faces) and fw > 16 and fh > 16:
        pred_resized = cv2.resize(predicted_faces[f_idx], (fw, fh), interpolation=cv2.INTER_LINEAR)
        blended_face = (
            pred_resized.astype(np.float32) * mouth_mask
            + base_face_for_frame.astype(np.float32) * (1.0 - mouth_mask)
        ).astype(np.uint8)
    else:
        blended_face = base_face_for_frame.copy()

    blended_face = _apply_syllable_mouth_and_blink(blended_face, energy, f_idx, fps=fps)
    frame_presenter[y1:y2, x1:x2] = blended_face

    if torso_grids is not None:
        ty_top = torso_grids["y_top"]
        torso_step = f_idx // 2
        prev_t_step, prev_torso_img = talk_state.get("_torso_step_cache", (-1, None))
        if prev_t_step == torso_step and prev_torso_img is not None:
            frame_presenter[ty_top:, :] = prev_torso_img
        else:
            torso_roi = frame_presenter[ty_top:, :]
            gx = torso_grids["gx"]
            gy = torso_grids["gy"]
            left_sh = torso_grids["left_sh"]
            right_sh = torso_grids["right_sh"]
            chest_zone = torso_grids["chest_zone"]
            neck_zone = torso_grids["neck_zone"]
            edge_taper = torso_grids["edge_taper"]

            shrug_l = (-4.2 * energy * (0.65 + 0.35 * math.sin(t_sec * 3.4)) - 1.8 * math.sin(t_sec * 1.65))
            shrug_r = (-4.2 * energy * (0.65 + 0.35 * math.cos(t_sec * 3.1)) - 1.8 * math.sin(t_sec * 1.65 + 0.45))
            breath_y = -2.6 * math.sin(2.0 * math.pi * 0.32 * t_sec) - 1.4 * energy
            sway_x = 3.4 * math.sin(2.0 * math.pi * 0.24 * t_sec) + 2.0 * math.cos(t_sec * 2.7) * energy
            neck_x = 2.4 * math.sin(2.0 * math.pi * 0.27 * t_sec)

            dy = (left_sh * shrug_l + right_sh * shrug_r + chest_zone * breath_y) * edge_taper
            dx = (chest_zone * sway_x + neck_zone * neck_x + (left_sh - right_sh) * (1.5 * energy)) * edge_taper

            map_x = np.clip(gx - dx, 0, presenter_w - 1).astype(np.float32)
            map_y = np.clip(gy - dy, 0, torso_roi.shape[0] - 1).astype(np.float32)
            remapped_torso = cv2.remap(
                torso_roi, map_x, map_y,
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            frame_presenter[ty_top:, :] = remapped_torso
            talk_state["_torso_step_cache"] = (torso_step, remapped_torso)

    # Anchor rotation/scale pivot near top of head (y = max(24, y1 - 0.45*fh)) and clamp ty >= 0
    # so zooming or swaying can NEVER push the top of the head above y = 0!
    scale = 1.008 + 0.005 * math.sin(2.0 * math.pi * 0.38 * t_sec) + 0.004 * energy
    angle_deg = 0.75 * math.sin(2.0 * math.pi * 0.27 * t_sec)
    tx = 2.5 * math.sin(2.0 * math.pi * 0.24 * t_sec)
    ty = max(0.0, 1.5 * math.cos(2.0 * math.pi * 0.42 * t_sec))

    if camera_closeup:
        scale *= 1.045
        ty += 6.0

    pivot_y = float(max(20, int(y1 - fh * 0.45)))
    M = cv2.getRotationMatrix2D((presenter_w * 0.5, pivot_y), angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    animated_presenter = cv2.warpAffine(
        frame_presenter, M, (presenter_w, presenter_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    if desk_bgr is not None and desk_alpha is not None:
        dh = desk_bgr.shape[0]
        bot_slice = animated_presenter[presenter_h - dh:, :].astype(np.float32)
        animated_presenter[presenter_h - dh:, :] = (
            bot_slice * (1.0 - desk_alpha) + desk_bgr * desk_alpha
        ).astype(np.uint8)

    talk_state["_frame_cache"] = (f_idx, camera_closeup, animated_presenter)
    return animated_presenter


# ==============================================================================
# 5. 6 YOUTUBE STUDIO TEMPLATES & ACCESSIBILITY-ENHANCED SLIDES
# ==============================================================================

STUDIO_TEMPLATES = {
    "broadcast_studio": {
        "name": "Broadcast Newsroom Studio",
        "presenter_side": "left",       # Presenter at x=64, Concept Card at x=784
        "slide_img_side": "right",
        "bg_tint": (12, 18, 36),
        "header_bg": (15, 23, 42, 232),
        "accent": (56, 189, 248),
        "accent_alt": (250, 204, 21),
        "card_bg": (255, 255, 255, 252),
        "card_header_bg": (15, 23, 42, 255),
        "card_text": (15, 23, 42, 255),
        "slide_bg_light": (241, 245, 249),
        "slide_bg_dark": (15, 23, 42),
        "pip_slide_pos": "top_right",
        "pip_broll_pos": "bottom_right",
    },
    "ted_keynote": {
        "name": "TED-Style Spotlight Keynote",
        "presenter_side": "right",      # Presenter at x=1176, Concept Card at x=64
        "slide_img_side": "left",
        "bg_tint": (28, 10, 18),
        "header_bg": (38, 12, 24, 235),
        "accent": (244, 63, 94),        # Vibrant Keynote Crimson-Rose
        "accent_alt": (251, 191, 36),   # Warm Stage Gold
        "card_bg": (24, 15, 24, 248),
        "card_header_bg": (159, 18, 57, 255),
        "card_text": (254, 242, 242, 255),
        "slide_bg_light": (30, 16, 26),
        "slide_bg_dark": (18, 9, 16),
        "pip_slide_pos": "top_left",
        "pip_broll_pos": "bottom_left",
    },
    "documentary_split": {
        "name": "Widescreen Documentary Cinema",
        "presenter_side": "left",
        "slide_img_side": "left",
        "bg_tint": (6, 34, 28),
        "header_bg": (6, 48, 38, 235),
        "accent": (16, 185, 129),       # NatGeo / BBC Emerald
        "accent_alt": (245, 158, 11),   # Editorial Amber
        "card_bg": (10, 42, 34, 246),
        "card_header_bg": (4, 64, 50, 255),
        "card_text": (236, 253, 245, 255),
        "slide_bg_light": (240, 253, 244),
        "slide_bg_dark": (6, 42, 34),
        "pip_slide_pos": "bottom_left",
        "pip_broll_pos": "bottom_right",
    },
    "academic_chalkboard": {
        "name": "University Lecture Hall & Chalkboard",
        "presenter_side": "right",      # Professor on Right, Chalkboard on Left
        "slide_img_side": "right",
        "bg_tint": (14, 38, 24),
        "header_bg": (20, 48, 32, 238),
        "accent": (234, 179, 8),        # Warm Academic Gold
        "accent_alt": (134, 239, 172),  # Mint Chalk
        "card_bg": (17, 44, 30, 250),   # Deep Slate Green Chalkboard
        "card_header_bg": (30, 58, 40, 255),
        "card_text": (254, 252, 232, 255),
        "slide_bg_light": (22, 52, 36),
        "slide_bg_dark": (14, 36, 24),
        "pip_slide_pos": "top_right",
        "pip_broll_pos": "bottom_left",
    },
    "modern_minimal_light": {
        "name": "Daylight Minimalist Executive Studio",
        "presenter_side": "left",
        "slide_img_side": "right",
        "bg_tint": (226, 232, 240),
        "header_bg": (15, 23, 42, 242),
        "accent": (37, 99, 235),        # Royal Blue
        "accent_alt": (217, 119, 6),    # Deep Gold
        "card_bg": (255, 255, 255, 255),
        "card_header_bg": (30, 58, 138, 255),
        "card_text": (15, 23, 42, 255),
        "slide_bg_light": (255, 255, 255),
        "slide_bg_dark": (241, 245, 249),
        "pip_slide_pos": "top_right",
        "pip_broll_pos": "bottom_right",
    },
    "cyber_stem_lab": {
        "name": "Futuristic STEM & Tech Innovation Lab",
        "presenter_side": "right",
        "slide_img_side": "left",
        "bg_tint": (11, 13, 30),
        "header_bg": (17, 20, 46, 238),
        "accent": (139, 92, 246),       # Electric Violet
        "accent_alt": (6, 182, 212),    # Neon Cyan
        "card_bg": (15, 18, 42, 248),
        "card_header_bg": (46, 16, 101, 255),
        "card_text": (243, 232, 255, 255),
        "slide_bg_light": (19, 22, 48),
        "slide_bg_dark": (10, 12, 28),
        "pip_slide_pos": "top_left",
        "pip_broll_pos": "bottom_right",
    },
}

TEMPLATE_KEYS_ORDER = [
    "broadcast_studio",
    "ted_keynote",
    "documentary_split",
    "academic_chalkboard",
    "modern_minimal_light",
    "cyber_stem_lab",
]


def resolve_studio_template(visual_template, block_idx=0):
    tpl_key = str(visual_template or "broadcast_studio").strip().lower()
    if tpl_key == "auto_rotate":
        tpl_key = TEMPLATE_KEYS_ORDER[block_idx % len(TEMPLATE_KEYS_ORDER)]
    return STUDIO_TEMPLATES.get(tpl_key, STUDIO_TEMPLATES["broadcast_studio"])


def build_studio_Composite_background(
    raw_avatar_path,
    lesson_title,
    slide_title,
    slide_notes,
    cf_image_path=None,
    block_idx=0,
    total_blocks=1,
    visual_template="broadcast_studio"
):
    """
    Creates a 1920x1080 YouTube Studio Canvas styled according to the user's chosen
    `visual_template` (supports both Left-Docked and Right-Docked Presenter slots,
    custom studio color palettes, and enlarged accessibility typography for visually
    impaired learners).
    Returns: (studio_bgr, presenter_x, presenter_y)
    """
    W, H = 1920, 1080
    theme = resolve_studio_template(visual_template, block_idx=block_idx)
    tint_rgb = theme["bg_tint"]
    accent = theme["accent"]
    accent_alt = theme["accent_alt"]
    presenter_side = theme["presenter_side"]

    first_av_bgr = _read_first_avatar_frame_bgr(raw_avatar_path)
    if first_av_bgr is not None:
        bg_pil = Image.fromarray(cv2.cvtColor(first_av_bgr, cv2.COLOR_BGR2RGB)).resize((W, H), Image.Resampling.BICUBIC)
        bg_pil = bg_pil.filter(ImageFilter.GaussianBlur(radius=38))
        tint_layer = Image.new("RGB", (W, H), tint_rgb)
        bg_pil = Image.blend(bg_pil, tint_layer, alpha=0.76)
    else:
        bg_pil = Image.new("RGB", (W, H), tint_rgb)

    draw = ImageDraw.Draw(bg_pil, "RGBA")

    # Top studio header bar (enlarged accessibility fonts)
    draw.rounded_rectangle([64, 18, 1856, 74], radius=16, fill=theme["header_bg"], outline=accent + (210,), width=3)
    font_header = _load_font(27, bold=True)
    font_badge = _load_font(22, bold=True)
    draw.text((88, 29), f"  {lesson_title.upper()[:52]}", font=font_header, fill=(255, 255, 255, 255))
    draw.text((1580, 32), f"MODULE {block_idx + 1} OF {total_blocks}", font=font_badge, fill=accent_alt + (255,))

    # Determine coordinates based on presenter_side ("left" vs "right")
    # Strict 40px safety moat between Presenter Slot (680x820 at y=84..904) and Concept Card (1072x820 at y=84..904)
    if presenter_side == "right":
        pres_x, pres_y = 1176, 84
        card_x1, card_x2 = 64, 1136
    else:
        pres_x, pres_y = 64, 84
        card_x1, card_x2 = 784, 1856

    card_w = card_x2 - card_x1
    card_h = 820
    # Render Concept Card onto its own isolated RGBA sub-canvas so long titles/notes
    # are 100% clipped inside the card and can NEVER bleed into the Avatar area!
    card_pil = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    c_draw = ImageDraw.Draw(card_pil, "RGBA")

    c_draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1], radius=26, fill=theme["card_bg"], outline=accent + (230,), width=3)
    c_draw.rounded_rectangle([0, 0, card_w - 1, 102], radius=26, fill=theme["card_header_bg"])
    c_draw.rectangle([0, 70, card_w - 1, 102], fill=theme["card_header_bg"])
    c_draw.rectangle([0, 94, card_w - 1, 102], fill=accent + (255,))

    font_slide_title = _load_font(38, bold=True)
    heading = (slide_title or lesson_title or f"Key Concept #{block_idx + 1}").strip()
    h_lines = _wrap_text(c_draw, heading, font_slide_title, card_w - 72)
    c_draw.text((36, 26), (h_lines[0] if h_lines else heading)[:42], font=font_slide_title, fill=(255, 255, 255, 255))

    notes_max_w = card_w - 82
    notes_start_y = 126
    if cf_image_path and os.path.exists(cf_image_path):
        try:
            cf_img = Image.open(cf_image_path).convert("RGB")
            tw, th = card_w - 80, 326
            iw, ih = cf_img.size
            sc = max(tw / float(max(1, iw)), th / float(max(1, ih)))
            cf_img = cf_img.resize((int(iw * sc), int(ih * sc)), Image.Resampling.LANCZOS)
            lx = (cf_img.size[0] - tw) // 2
            ly = (cf_img.size[1] - th) // 2
            cf_img = cf_img.crop((lx, ly, lx + tw, ly + th))
            mask = Image.new("L", (tw, th), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw, th], radius=18, fill=255)
            card_pil.paste(cf_img, (40, 118), mask)
            notes_start_y = 464
        except Exception as e:
            print(f"Studio CF image embed note: {e}")

    font_bullet = _load_font(34, bold=True)
    curr_y = notes_start_y
    bullet_colors = [accent, accent_alt, (16, 185, 129), (59, 130, 246)]

    for idx, note in enumerate((slide_notes or [])[:4]):
        clean_note = str(note).strip()
        if not clean_note:
            continue
        lines = _wrap_text(c_draw, clean_note, font_bullet, notes_max_w - 72)
        b_col = bullet_colors[idx % len(bullet_colors)]
        c_draw.rounded_rectangle([40, curr_y + 5, 78, curr_y + 43], radius=10, fill=b_col + (255,))
        for line in lines[:2]:
            c_draw.text((94, curr_y), line, font=font_bullet, fill=theme["card_text"])
            curr_y += 44
        curr_y += 14
        if curr_y > card_h - 48:
            break

    bg_pil.paste(card_pil, (card_x1, 84), card_pil)

    # Presenter Frame glowing border (y=84..904)
    draw.rounded_rectangle(
        [pres_x - 6, pres_y - 6, pres_x + 680 + 6, pres_y + 820 + 6],
        radius=26,
        fill=(15, 23, 42, 195),
        outline=accent + (245,),
        width=4
    )

    return cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR), pres_x, pres_y


def generate_fallback_illustration(prompt_text, title_text, output_png, variant=0):
    """
    Generates a rich, colorful visual infographic diagram (with geometric vector charts,
    nodes, and visual cards) if both Cloudflare Workers AI and Wikimedia Commons are unreachable.
    """
    W, H = 1280, 720
    palettes = [
        ((15, 23, 42), (37, 99, 235), (56, 189, 248), (250, 204, 21)),
        ((6, 78, 59), (16, 185, 129), (52, 211, 153), (251, 191, 36)),
        ((49, 46, 129), (124, 58, 237), (192, 132, 252), (56, 189, 248)),
        ((124, 45, 18), (234, 88, 12), (251, 146, 60), (56, 189, 248)),
    ]
    c1, c2, accent, gold = palettes[variant % len(palettes)]
    img = Image.new("RGB", (W, H), c1)
    draw = ImageDraw.Draw(img, "RGBA")

    draw.ellipse([W - 480, -140, W + 200, 540], fill=c2 + (165,))
    draw.ellipse([-180, H - 420, 460, H + 220], fill=accent + (80,))

    cx, cy = 920, 360
    nodes = [(cx, cy - 160), (cx + 170, cy - 40), (cx + 110, cy + 150), (cx - 110, cy + 150), (cx - 170, cy - 40)]
    for nx, ny in nodes:
        draw.line([cx, cy, nx, ny], fill=accent + (210,), width=5)
        draw.ellipse([nx - 36, ny - 36, nx + 36, ny + 36], fill=c2 + (240,), outline=gold + (255,), width=4)
    draw.ellipse([cx - 58, cy - 58, cx + 58, cy + 58], fill=gold + (255,), outline=(255, 255, 255, 255), width=5)

    draw.rounded_rectangle([64, 64, 680, H - 64], radius=28, fill=(15, 23, 42, 225), outline=accent + (220,), width=3)
    font_tag = _load_font(24, bold=True)
    font_title = _load_font(46, bold=True)
    font_body = _load_font(31, bold=True)

    draw.text((104, 104), "VISUAL CONCEPT DIAGRAM", font=font_tag, fill=gold + (255,))
    t_lines = _wrap_text(draw, title_text or "Core Lesson Concept", font_title, 530)
    y = 158
    for line in t_lines[:3]:
        draw.text((104, y), line, font=font_title, fill=(255, 255, 255, 255))
        y += 58

    draw.line([104, y + 12, 620, y + 12], fill=accent + (190,), width=3)
    y += 36
    p_lines = _wrap_text(draw, prompt_text or "", font_body, 530)
    for line in p_lines[:5]:
        draw.text((104, y), line, font=font_body, fill=(226, 232, 240, 255))
        y += 42

    img.save(output_png, "PNG")
    return output_png


def render_data_chart_or_formula_card(chart_data, width, height, theme, slide_title="", anim_progress=1.0):
    """
    Renders a crisp, broadcast-grade Data Chart, Comparison Table, Process Flowchart,
    or Scientific Formula Board onto a PIL RGB Image of size (width, height).
    Supports progressive animation via `anim_progress` (0.0 .. 1.0):
      - 'bar'     : Metric bars grow smoothly from 0% -> target% with animated value counters
      - 'process' : Step-by-step connected flowchart nodes light up sequentially
      - 'formula' : Equation types in with staggered variable legend reveal
    """
    anim_p = max(0.05, min(1.0, float(anim_progress)))
    accent = theme["accent"]
    accent_alt = theme["accent_alt"]
    card = Image.new("RGB", (width, height), (13, 20, 38))
    draw = ImageDraw.Draw(card, "RGBA")

    for gx in range(40, width, 60):
        draw.line([(gx, 0), (gx, height)], fill=(255, 255, 255, 13), width=1)
    for gy in range(40, height, 60):
        draw.line([(0, gy), (width, gy)], fill=(255, 255, 255, 13), width=1)

    ctype = str(chart_data.get("type") or "bar").strip().lower()
    ctitle = str(chart_data.get("title") or slide_title or "Key Data & Metrics").strip()
    unit = str(chart_data.get("unit") or "").strip()

    draw.rounded_rectangle([18, 18, width - 18, 92], radius=16, fill=(22, 33, 62, 240), outline=accent + (180,), width=2)
    font_ctitle = _load_font(27, bold=True)
    draw.text((38, 38), ctitle[:38], font=font_ctitle, fill=accent_alt + (255,))

    labels = [str(x).strip() for x in (chart_data.get("labels") or []) if str(x).strip()]
    raw_vals = chart_data.get("values") or []
    vals = []
    for v in raw_vals:
        try:
            vals.append(float(str(v).replace("%", "").replace("$", "").replace(",", "").strip()))
        except Exception:
            vals.append(50.0)

    if ctype == "formula" or chart_data.get("formula"):
        formula_str = str(chart_data.get("formula") or "E = m · c²").strip()
        draw.rounded_rectangle([32, 118, width - 32, 310], radius=22, fill=(17, 28, 52, 255), outline=accent_alt + (235,), width=3)
        font_eq_tag = _load_font(20, bold=True)
        draw.text((54, 134), "CORE EQUATION / PRINCIPLE", font=font_eq_tag, fill=accent + (255,))
        font_eq = _load_font(44, bold=True)
        vis_chars = max(1, int(round(len(formula_str) * min(1.0, anim_p * 1.4))))
        shown_eq = formula_str[:vis_chars] + (" ▍" if vis_chars < len(formula_str) else "")
        eq_lines = _wrap_text(draw, shown_eq, font_eq, width - 110)
        ey = 176
        for el in eq_lines[:2]:
            draw.text((54, ey), el, font=font_eq, fill=(255, 255, 255, 255))
            ey += 54

        font_leg = _load_font(25, bold=True)
        ly = 338
        n_leg = max(1, len(labels[:5]))
        for idx, lbl in enumerate(labels[:5]):
            item_p = max(0.0, min(1.0, (anim_p * (n_leg + 0.5)) - idx * 0.7))
            if item_p <= 0.05:
                continue
            alpha_l = int(255 * item_p)
            b_col = accent if idx % 2 == 0 else accent_alt
            draw.ellipse([40, ly + 6, 58, ly + 24], fill=b_col + (alpha_l,))
            w_lines = _wrap_text(draw, lbl, font_leg, width - 110)
            for wl in w_lines[:2]:
                draw.text((74, ly), wl, font=font_leg, fill=(235, 242, 255, alpha_l))
                ly += 33
            ly += 12
            if ly > height - 45:
                break
        return card

    if ctype == "process" and labels:
        steps = labels[:5]
        n_steps = len(steps)
        box_h = min(102, max(68, (height - 150) // max(1, n_steps) - 16))
        curr_y = 116
        font_step = _load_font(26, bold=True)
        font_num = _load_font(24, bold=True)
        palette = [accent, (16, 185, 129), accent_alt, (139, 92, 246), (244, 63, 94)]
        active_step = min(n_steps - 1, int(anim_p * n_steps))
        for i, st in enumerate(steps):
            step_p = max(0.0, min(1.0, (anim_p * (n_steps + 0.4)) - i * 0.75))
            if step_p <= 0.02:
                break
            col = palette[i % len(palette)]
            is_curr = (i == active_step and anim_p < 0.96)
            a_box = int(245 * step_p)
            outline_col = (255, 255, 255, 255) if is_curr else (col + (int(235 * step_p),))
            draw.rounded_rectangle([36, curr_y, width - 36, curr_y + box_h], radius=16, fill=(24, 36, 68, a_box), outline=outline_col, width=4 if is_curr else 3)
            draw.ellipse([54, curr_y + (box_h - 44) // 2, 98, curr_y + (box_h + 44) // 2], fill=col + (int(255 * step_p),))
            draw.text((69, curr_y + (box_h - 30) // 2), str(i + 1), font=font_num, fill=(15, 23, 42, int(255 * step_p)))
            s_lines = _wrap_text(draw, st, font_step, width - 170)
            ty = curr_y + (box_h - len(s_lines[:2]) * 30) // 2
            for sl in s_lines[:2]:
                draw.text((116, ty), sl, font=font_step, fill=(255, 255, 255, int(255 * step_p)))
                ty += 30
            if i < n_steps - 1 and step_p > 0.6:
                cx = width // 2
                draw.line([(cx, curr_y + box_h), (cx, curr_y + box_h + 16)], fill=accent_alt + (int(255 * step_p),), width=4)
            curr_y += box_h + 18
        return card

    if not labels:
        labels = ["Core Efficiency", "Retention Rate", "Accuracy Benchmark", "Practical Impact"]
        vals = [92.0, 84.0, 96.0, 78.0]
    while len(vals) < len(labels):
        vals.append(float(70 + (len(vals) * 8) % 25))

    max_v = max([abs(x) for x in vals] + [1.0])
    curr_y = 122
    bar_colors = [accent, accent_alt, (16, 185, 129), (139, 92, 246), (244, 63, 94)]
    font_lbl = _load_font(26, bold=True)
    font_val = _load_font(25, bold=True)
    max_bar_w = width - 96

    for idx, lbl in enumerate(labels[:5]):
        v = vals[idx]
        bar_ease = max(0.08, min(1.0, (anim_p * 1.35) - idx * 0.14))
        bar_ease = math.sin(bar_ease * math.pi * 0.5)
        ratio = max(0.10, min(1.0, abs(v) / max_v)) * bar_ease
        b_col = bar_colors[idx % len(bar_colors)]
        shown_v = v * bar_ease
        val_str = f"{shown_v:.0f}{unit}" if unit else f"{shown_v:.0f}"

        draw.text((40, curr_y), lbl[:34], font=font_lbl, fill=(241, 245, 249, 255))
        draw.text((width - 145, curr_y), val_str[:10], font=font_val, fill=accent_alt + (255,))
        curr_y += 36

        draw.rounded_rectangle([40, curr_y, 40 + max_bar_w, curr_y + 34], radius=12, fill=(30, 41, 59, 255))
        fill_w = max(28, int(max_bar_w * ratio))
        draw.rounded_rectangle([40, curr_y, 40 + fill_w, curr_y + 34], radius=12, fill=b_col + (255,))
        curr_y += 58
        if curr_y > height - 55:
            break

    return card


def _prepare_brand_logo_overlay(logo_path, max_w=192, max_h=78):
    """
    Loads an optional user/channel Brand Logo (.png/.webp/.jpg/.gif) from either direct upload
    or WordPress Site Media Library URL and prepares both a PIL RGBA image (for thumbnails)
    and pre-multiplied BGR + alpha arrays (for fast per-frame video watermarking).
    Includes OpenCV fallback if PIL encounters unusual color profiles.
    """
    if not logo_path or not os.path.exists(logo_path):
        return None
    try:
        pil_logo = None
        try:
            img_open = Image.open(logo_path)
            if getattr(img_open, "is_animated", False):
                img_open.seek(0)
            pil_logo = img_open.convert("RGBA")
        except Exception:
            cv_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if cv_img is not None and cv_img.size > 0:
                if cv_img.ndim == 2:
                    pil_logo = Image.fromarray(cv_img).convert("RGBA")
                elif cv_img.shape[2] == 4:
                    rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
                    pil_logo = Image.fromarray(rgba)
                else:
                    rgb = cv2.cvtColor(cv_img[:, :, :3], cv2.COLOR_BGR2RGB)
                    pil_logo = Image.fromarray(rgb).convert("RGBA")

        if pil_logo is None:
            return None

        lw, lh = pil_logo.size
        if lw < 4 or lh < 4:
            return None
        sc = min(max_w / float(lw), max_h / float(lh))
        nw, nh = max(20, int(lw * sc)), max(20, int(lh * sc))
        pil_logo = pil_logo.resize((nw, nh), Image.Resampling.LANCZOS)

        # Check if logo has meaningful transparency; if fully opaque, frame it in a sleek rounded glass pill
        alpha_extrema = pil_logo.getchannel("A").getextrema()
        is_opaque = (alpha_extrema and alpha_extrema[0] > 250)

        pad_x, pad_y = 14, 8
        box_w, box_h = nw + pad_x * 2, nh + pad_y * 2
        badge = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        b_draw = ImageDraw.Draw(badge, "RGBA")
        b_draw.rounded_rectangle(
            [0, 0, box_w - 1, box_h - 1],
            radius=14,
            fill=(10, 16, 32, 205 if is_opaque else 145),
            outline=(255, 255, 255, 130),
            width=2
        )
        badge.paste(pil_logo, (pad_x, pad_y), pil_logo)

        arr = np.array(badge)
        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
        alpha = (arr[:, :, 3].astype(np.float32) / 255.0)[..., None]
        return {
            "pil": badge,
            "bgr": bgr,
            "alpha": alpha,
            "w": box_w,
            "h": box_h,
        }
    except Exception as e:
        print(f"Brand logo load note: {e}")
        return None


def _apply_brand_logo_to_frame_bgr(frame_bgr, logo_cache, pos="top_right"):
    """
    Fast NumPy alpha-blend of the optional Brand Logo onto `frame_bgr`.
    """
    if not logo_cache:
        return frame_bgr
    H, W = frame_bgr.shape[:2]
    lw, lh = logo_cache["w"], logo_cache["h"]
    if lw >= W - 40 or lh >= H - 40:
        return frame_bgr

    pos_clean = str(pos or "top_right").strip().lower()
    if pos_clean == "top_left":
        x0, y0 = 36, 24
    elif pos_clean == "bottom_right":
        x0, y0 = W - lw - 36, H - lh - 32
    elif pos_clean == "bottom_left":
        x0, y0 = 36, H - lh - 32
    else:
        # top_right default
        x0, y0 = W - lw - 36, 24

    roi = frame_bgr[y0:y0 + lh, x0:x0 + lw].astype(np.float32)
    a = logo_cache["alpha"]
    frame_bgr[y0:y0 + lh, x0:x0 + lw] = (roi * (1.0 - a) + logo_cache["bgr"] * a).astype(np.uint8)
    return frame_bgr


THUMBNAIL_PALETTES = [
    {"tint": (10, 16, 36), "primary": (56, 189, 248), "secondary": (250, 204, 21), "badge_bg": (239, 68, 68), "badge_fg": (255, 255, 255)},
    {"tint": (28, 10, 20), "primary": (244, 63, 94), "secondary": (253, 224, 71), "badge_bg": (250, 204, 21), "badge_fg": (15, 23, 42)},
    {"tint": (6, 32, 24), "primary": (16, 185, 129), "secondary": (251, 191, 36), "badge_bg": (16, 185, 129), "badge_fg": (6, 24, 18)},
    {"tint": (18, 12, 40), "primary": (168, 85, 247), "secondary": (34, 211, 238), "badge_bg": (34, 211, 238), "badge_fg": (15, 23, 42)},
    {"tint": (32, 18, 8), "primary": (249, 115, 22), "secondary": (254, 240, 138), "badge_bg": (249, 115, 22), "badge_fg": (255, 255, 255)},
    {"tint": (12, 26, 44), "primary": (59, 130, 246), "secondary": (134, 239, 172), "badge_bg": (234, 179, 8), "badge_fg": (15, 23, 42)},
]


def generate_youtube_thumbnail(
    raw_avatar_path,
    cf_image_path,
    lesson_title,
    presenter_name="",
    channel_name="",
    visual_template="broadcast_studio",
    output_jpg_path="thumbnail.jpg",
    logo_path=None,
    variant_seed=None
):
    """
    6-Variant Anti-Repetitive High-CTR 1280x720 YouTube Thumbnail Engine (.jpg):
    Automatically selects a unique layout & color palette per video based on
    `variant_seed` / `lesson_title` hash so no two lessons ever share a repetitive
    thumbnail!
      - Layout 0: Diagonal Split Action (Left Hook + Angled Slash + Right Cutout Portrait)
      - Layout 1: Left Hero Portrait + Right Stacked Ribbon Headlines
      - Layout 2: Editorial Masterclass Gold Frame + Floating Glass Hook Card
      - Layout 3: Top Giant Impact Banner + Dual Split Stage (Visual Left / Teacher Right)
      - Layout 4: Circular Spotlight Medallion Portrait + High-Contrast Left Hook
      - Layout 5: Full-Bleed Cinema Backdrop + Lower-Third Giant Headline & Left Portrait
    """
    import hashlib
    W, H = 1280, 720
    seed_str = f"{lesson_title}_{channel_name}_{visual_template}_{variant_seed or output_jpg_path}"
    h_val = int(hashlib.md5(seed_str.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
    layout_idx = h_val % 6
    pal = THUMBNAIL_PALETTES[(h_val // 6) % len(THUMBNAIL_PALETTES)]

    primary = pal["primary"]
    secondary = pal["secondary"]
    tint = pal["tint"]
    badge_bg = pal["badge_bg"]
    badge_fg = pal["badge_fg"]

    # 1. Prepare vivid high-contrast AI visual background
    if cf_image_path and os.path.exists(cf_image_path):
        try:
            bg_raw = Image.open(cf_image_path).convert("RGB")
            iw, ih = bg_raw.size
            sc = max(W / float(max(1, iw)), H / float(max(1, ih)))
            bg_raw = bg_raw.resize((int(iw * sc), int(ih * sc)), Image.Resampling.LANCZOS)
            lx, ly = (bg_raw.size[0] - W) // 2, (bg_raw.size[1] - H) // 2
            bg_raw = bg_raw.crop((lx, ly, lx + W, ly + H))
            dark = Image.new("RGB", (W, H), tint)
            bg = Image.blend(bg_raw, dark, alpha=0.52)
        except Exception:
            bg_raw = Image.new("RGB", (W, H), tint)
            bg = bg_raw.copy()
    else:
        bg_raw = Image.new("RGB", (W, H), tint)
        bg = bg_raw.copy()

    draw = ImageDraw.Draw(bg, "RGBA")

    # Determine whether portrait is on Right or Left for this layout variant
    portrait_on_left = layout_idx in (1, 5)

    # 2. Layout-specific geometric backdrops & glow accents
    if layout_idx == 0:
        # Layout 0: Diagonal Split Slash
        draw.polygon([(0, 0), (830, 0), (670, H), (0, H)], fill=tint + (228,))
        draw.line([(830, 0), (670, H)], fill=secondary + (255,), width=10)
        draw.line([(842, 0), (682, H)], fill=primary + (220,), width=6)
        draw.ellipse([720, 40, 1300, 680], fill=primary + (78,))
    elif layout_idx == 1:
        # Layout 1: Left Hero Portrait + Right Ribbon Stack
        draw.polygon([(460, 0), (W, 0), (W, H), (540, H)], fill=tint + (225,))
        draw.line([(460, 0), (540, H)], fill=primary + (255,), width=9)
        draw.ellipse([-60, 40, 540, 690], fill=secondary + (75,))
    elif layout_idx == 2:
        # Layout 2: Editorial Masterclass Frame + Floating Glass Card
        draw.rounded_rectangle([38, 38, 760, H - 38], radius=28, fill=tint + (236,), outline=secondary + (245,), width=5)
        draw.ellipse([740, 60, 1260, 660], fill=primary + (85,))
    elif layout_idx == 3:
        # Layout 3: Top Giant Impact Banner + Split Stage
        draw.rectangle([0, 0, W, 248], fill=tint + (244,))
        draw.line([(0, 248), (W, 248)], fill=secondary + (255,), width=7)
        draw.rounded_rectangle([42, 278, 748, H - 34], radius=24, fill=(10, 16, 32, 210), outline=primary + (235,), width=4)
    elif layout_idx == 4:
        # Layout 4: Circular Spotlight Medallion on Right + Bold Hook Left
        draw.rounded_rectangle([40, 44, 756, H - 44], radius=30, fill=tint + (232,), outline=primary + (235,), width=4)
        draw.ellipse([770, 95, 1245, 570], fill=secondary + (95,))
    else:
        # Layout 5: Left Portrait + Right/Bottom Cinema Headline Cards
        draw.polygon([(475, 0), (W, 0), (W, H), (430, H)], fill=tint + (218,))
        draw.line([(475, 0), (430, H)], fill=secondary + (255,), width=8)

    # 3. Render Presenter Portrait (Supports both Photo & Video Avatar files!)
    if raw_avatar_path and os.path.exists(raw_avatar_path):
        try:
            if layout_idx == 4:
                # Circular Medallion Portrait (430x430)
                pw, ph = 430, 430
                pres_bgr, _, _ = prepare_presenter_portrait(raw_avatar_path, pw, ph, device="cpu")
                pres_pil = Image.fromarray(cv2.cvtColor(pres_bgr, cv2.COLOR_BGR2RGB))
                px, py = 792, 118
                draw.ellipse([px - 10, py - 10, px + pw + 10, py + ph + 10], fill=secondary + (255,))
                draw.ellipse([px - 4, py - 4, px + pw + 4, py + ph + 4], fill=(15, 23, 42, 255))
                c_mask = Image.new("L", (pw, ph), 0)
                ImageDraw.Draw(c_mask).ellipse([0, 0, pw, ph], fill=255)
                bg.paste(pres_pil, (px, py), c_mask)
            elif layout_idx == 3:
                # Lower-right portrait below top banner
                pw, ph = 440, 412
                pres_bgr, _, _ = prepare_presenter_portrait(raw_avatar_path, pw, ph, device="cpu")
                pres_pil = Image.fromarray(cv2.cvtColor(pres_bgr, cv2.COLOR_BGR2RGB))
                px, py = 792, 276
                draw.rounded_rectangle([px - 6, py - 6, px + pw + 6, py + ph + 6], radius=26, fill=(15, 23, 42, 255), outline=secondary + (255,), width=6)
                r_mask = Image.new("L", (pw, ph), 0)
                ImageDraw.Draw(r_mask).rounded_rectangle([0, 0, pw, ph], radius=22, fill=255)
                bg.paste(pres_pil, (px, py), r_mask)
                # Paste crisp mini AI visual inside left stage box
                vis_mini = bg_raw.resize((694, 396), Image.Resampling.LANCZOS)
                v_mask = Image.new("L", (694, 396), 0)
                ImageDraw.Draw(v_mask).rounded_rectangle([0, 0, 694, 396], radius=18, fill=255)
                bg.paste(vis_mini, (48, 284), v_mask)
            else:
                pw, ph = 428, 592
                pres_bgr, _, _ = prepare_presenter_portrait(raw_avatar_path, pw, ph, device="cpu")
                pres_pil = Image.fromarray(cv2.cvtColor(pres_bgr, cv2.COLOR_BGR2RGB))
                px = 46 if portrait_on_left else 798
                py = 64
                draw.rounded_rectangle([px - 7, py - 7, px + pw + 7, py + ph + 7], radius=30, fill=(15, 23, 42, 255), outline=secondary + (255,), width=6)
                r_mask = Image.new("L", (pw, ph), 0)
                ImageDraw.Draw(r_mask).rounded_rectangle([0, 0, pw, ph], radius=24, fill=255)
                bg.paste(pres_pil, (px, py), r_mask)
        except Exception as e:
            print(f"Thumbnail portrait note: {e}")

    # 4. Render High-CTR Hook Typography & Badges according to Layout
    badge_txt = (channel_name or "COMPLETE LESSON").strip().upper()[:28]
    clean_title = (lesson_title or "VIDEO MASTERCLASS").strip().upper()

    if layout_idx == 3:
        # Top Full-Width Impact Banner
        font_top = _load_font(56, bold=True)
        top_lines = _wrap_text(draw, clean_title, font_top, W - 120)
        ty = 32 if len(top_lines) > 1 else 68
        for idx, tl in enumerate(top_lines[:2]):
            draw.text((64, ty + 4), tl, font=font_top, fill=(0, 0, 0, 235))
            col = secondary + (255,) if idx == 1 else (255, 255, 255, 255)
            draw.text((60, ty), tl, font=font_top, fill=col)
            ty += 68
        draw.rounded_rectangle([60, 182, 480, 232], radius=12, fill=badge_bg + (255,))
        font_b = _load_font(24, bold=True)
        draw.text((82, 194), badge_txt, font=font_b, fill=badge_fg + (255,))
    else:
        tx_start = 528 if portrait_on_left else 66
        max_tw = 685

        # Top Category / Channel Pill Badge
        draw.rounded_rectangle([tx_start, 54, tx_start + 440, 112], radius=14, fill=badge_bg + (250,))
        font_badge = _load_font(25, bold=True)
        draw.text((tx_start + 22, 68), badge_txt, font=font_badge, fill=badge_fg + (255,))

        font_hook = _load_font(66, bold=True)
        hook_lines = _wrap_text(draw, clean_title, font_hook, max_tw)
        hy = 138
        for idx, line in enumerate(hook_lines[:4]):
            if layout_idx == 1 and idx % 2 == 1:
                # Highlighted ribbon bar behind alternating hook lines
                bbox = draw.textbbox((tx_start, hy), line, font=font_hook)
                draw.rounded_rectangle([tx_start - 12, hy - 4, min(W - 36, bbox[2] + 18), hy + 72], radius=12, fill=primary + (235,))
                draw.text((tx_start, hy), line, font=font_hook, fill=(10, 15, 30, 255))
            else:
                draw.text((tx_start + 5, hy + 5), line, font=font_hook, fill=(0, 0, 0, 235))
                fill_col = secondary + (255,) if idx == len(hook_lines[:4]) - 1 else (255, 255, 255, 255)
                draw.text((tx_start, hy), line, font=font_hook, fill=fill_col)
            hy += 82

        if presenter_name:
            py_box = H - 114
            draw.rounded_rectangle([tx_start, py_box, tx_start + 580, py_box + 64], radius=16, fill=(15, 23, 42, 240), outline=secondary + (225,), width=3)
            font_pres = _load_font(27, bold=True)
            draw.text((tx_start + 24, py_box + 17), f"INSTRUCTOR: {presenter_name.upper()[:28]}", font=font_pres, fill=(255, 255, 255, 255))

    # 5. Optional Brand Logo Watermark on Thumbnail
    logo_cache = _prepare_brand_logo_overlay(logo_path, max_w=165, max_h=66)
    if logo_cache and logo_cache.get("pil"):
        l_pil = logo_cache["pil"]
        lx = 36 if not portrait_on_left else (W - l_pil.size[0] - 36)
        ly = H - l_pil.size[1] - 28 if layout_idx == 3 else 24
        if not portrait_on_left:
            lx = W - l_pil.size[0] - 32
            ly = 24
        bg.paste(l_pil, (lx, ly), l_pil)

    # Outer vibrant broadcast border frame
    draw.rectangle([5, 5, W - 6, H - 6], outline=primary + (235,), width=6)
    bg.save(output_jpg_path, "JPEG", quality=94)
    return output_jpg_path


def _draw_broadcast_branding_overlays(
    frame_bgr,
    t_local,
    t_global,
    total_video_dur,
    block_idx,
    total_blocks,
    lesson_title,
    slide_title,
    presenter_name="",
    channel_name="",
    theme=None,
    curr_mode="studio_lipsync",
    avatar_rect=None
):
    """
    Renders four high-retention broadcast television graphics onto `frame_bgr` while
    strictly respecting `avatar_rect` when `curr_mode == 'studio_lipsync'` so text
    or banners NEVER cover the presenter's face or body:
      1. Sleek Intro Title Badge during the first 2.6 seconds of Block 1
      2. Animated Slide-Out Lower-Third Teacher Nameplate (t = 1.2s .. 7.8s in Block 1)
      3. Periodic Big-Font Lesson Title / Topic Reminder Banner shown 3 TIMES across the
         video (~18%, ~52%, and ~84% of total video duration) placed strictly inside the
         Concept Card column when the avatar is on screen!
      4. Dynamic Multi-Variant Animated YouTube Call-To-Action Banner (LIKE • COMMENT • SHARE • SUBSCRIBE)
         during the final 4.2s of the last block
    """
    H, W = frame_bgr.shape[:2]
    accent = (theme or {}).get("accent", (56, 189, 248))
    accent_alt = (theme or {}).get("accent_alt", (250, 204, 21))
    presenter_side = (theme or {}).get("presenter_side", "left")
    avatar_showing = (curr_mode == "studio_lipsync") and (W >= H)

    # Safe horizontal zone away from avatar when avatar is showing in 16:9 Studio mode
    if avatar_showing:
        if presenter_side == "right":
            # Avatar is at x=1176..1856 -> keep banners in left Concept Card zone x=64..1136
            safe_x1, safe_x2 = 68, 1132
        else:
            # Avatar is at x=64..744 -> keep banners in right Concept Card zone x=784..1856
            safe_x1, safe_x2 = 788, 1852
    else:
        safe_x1, safe_x2 = 64, W - 64

    safe_w = max(320, safe_x2 - safe_x1)

    show_intro = (block_idx == 0 and t_local < 2.4)
    show_lower_third = (block_idx == 0 and 1.2 <= t_local <= 7.8 and (presenter_name or channel_name))
    time_remaining = max(0.0, total_video_dur - t_global)
    show_outro = (block_idx == total_blocks - 1 and time_remaining <= 4.2)

    dur_safe = max(12.0, float(total_video_dur))
    half_win = min(2.25, max(1.1, dur_safe * 0.06))
    reminder_centers = [dur_safe * 0.18, dur_safe * 0.52, dur_safe * 0.84]
    active_reminder_idx = -1
    reminder_prog = 0.0
    if not show_intro and not show_outro:
        for r_idx, c_sec in enumerate(reminder_centers):
            dist = abs(t_global - c_sec)
            if dist <= half_win:
                active_reminder_idx = r_idx
                edge_dist = half_win - dist
                reminder_prog = min(1.0, max(0.0, edge_dist / min(0.55, half_win * 0.35)))
                break

    show_topic_reminder = (active_reminder_idx >= 0 and reminder_prog > 0.02)

    if not (show_intro or show_lower_third or show_outro or show_topic_reminder):
        return frame_bgr

    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # 1. Animated Lower-Third Presenter Nameplate (t = 1.2s .. 7.8s)
    # Placed inside the safe Concept Card zone (or bottom desk strip) so it never covers the avatar face
    if show_lower_third:
        if t_local < 1.8:
            prog = (t_local - 1.2) / 0.6
        elif t_local > 7.2:
            prog = max(0.0, (7.8 - t_local) / 0.6)
        else:
            prog = 1.0
        ease = math.sin(prog * math.pi * 0.5)
        box_w = min(safe_w - 24, 580)
        x_left = safe_x1 + int((1.0 - ease) * 40)
        y_bot = 898 if W >= H else H - 280
        y_top = y_bot - 96

        draw.rounded_rectangle([x_left, y_top, x_left + box_w, y_bot], radius=18, fill=(10, 16, 34, int(238 * ease)), outline=accent + (int(240 * ease),), width=3)
        draw.rounded_rectangle([x_left, y_top, x_left + 16, y_bot], radius=8, fill=accent_alt + (int(255 * ease),))
        font_name = _load_font(29, bold=True)
        font_role = _load_font(20, bold=True)
        main_lbl = (presenter_name or channel_name or "Course Instructor").strip()
        sub_lbl = (channel_name if presenter_name and channel_name else (lesson_title or "Featured Masterclass")).strip()
        draw.text((x_left + 30, y_top + 14), main_lbl[:32], font=font_name, fill=(255, 255, 255, int(255 * ease)))
        draw.text((x_left + 30, y_top + 54), sub_lbl.upper()[:38], font=font_role, fill=accent_alt + (int(255 * ease),))

    # 2. Branded Intro Module Pill (first 2.4s of Block 1) — strictly in safe_x1..safe_x2
    if show_intro:
        alpha_i = int(235 * min(1.0, (2.4 - t_local) / 0.6))
        bw = min(safe_w - 20, 720)
        bx1 = safe_x1 + (safe_w - bw) // 2
        by1 = 92 if W >= H else 160
        draw.rounded_rectangle([bx1, by1, bx1 + bw, by1 + 60], radius=30, fill=(15, 23, 42, alpha_i), outline=accent_alt + (alpha_i,), width=3)
        font_ib = _load_font(24, bold=True)
        ch_prefix = f"{channel_name.upper()}  •  " if channel_name else "OFFICIAL LESSON  •  "
        draw.text((bx1 + 28, by1 + 16), f"{ch_prefix}{lesson_title[:34].upper()}", font=font_ib, fill=(255, 255, 255, alpha_i))

    # 3. Periodic Lesson Title / Topic Reminder Banner — strictly in safe_x1..safe_x2 when avatar is showing!
    if show_topic_reminder:
        ease_r = math.sin(reminder_prog * math.pi * 0.5)
        alpha_r = int(246 * ease_r)
        bw = min(safe_w - 16, 1040 if avatar_showing else 1340)
        bh = 118 if W >= H else 148
        bx1 = safe_x1 + (safe_w - bw) // 2
        target_y = 780 if avatar_showing else (92 if W >= H else 172)
        by1 = int(target_y + (1.0 - ease_r) * (24 if avatar_showing else -28))

        draw.rounded_rectangle(
            [bx1, by1, bx1 + bw, by1 + bh],
            radius=22,
            fill=(8, 14, 30, alpha_r),
            outline=accent_alt + (alpha_r,),
            width=3
        )
        draw.rounded_rectangle(
            [bx1, by1, bx1 + 18, by1 + bh],
            radius=9,
            fill=accent + (alpha_r,)
        )

        font_tag_r = _load_font(20, bold=True)
        tag_label = f"TOPIC REMINDER ({active_reminder_idx + 1}/3)"
        if slide_title and slide_title.lower() != lesson_title.lower():
            tag_label += f"  •  {slide_title.upper()[:32]}"
        draw.text((bx1 + 34, by1 + 14), tag_label, font=font_tag_r, fill=accent_alt + (alpha_r,))

        font_big_title = _load_font(36 if avatar_showing else (42 if W >= H else 34), bold=True)
        clean_big_topic = (lesson_title or "Interactive Video Lesson").strip().upper()
        wrapped_topic = _wrap_text(draw, clean_big_topic, font_big_title, bw - 60)
        headline_str = wrapped_topic[0] if wrapped_topic else clean_big_topic[:38]
        draw.text((bx1 + 37, by1 + 49), headline_str, font=font_big_title, fill=(0, 0, 0, alpha_r))
        draw.text((bx1 + 34, by1 + 46), headline_str, font=font_big_title, fill=(255, 255, 255, alpha_r))

    # 4. Dynamic Multi-Variant Animated YouTube Call-To-Action Banner (final 4.2s of last block)
    # Placed strictly inside safe_x1..safe_x2 when avatar is visible so it never touches the avatar!
    if show_outro:
        prog_o = min(1.0, (4.2 - time_remaining) / 0.55)
        ease_o = math.sin(prog_o * math.pi * 0.5)
        alpha_o = int(248 * ease_o)
        bw = min(safe_w - 16, 1020)
        bh = 104
        bx1 = safe_x1 + (safe_w - bw) // 2
        by1 = 796 if W >= H else H - 250

        cta_seed = int(hashlib.md5(f"{lesson_title}|{channel_name}".encode("utf-8", errors="ignore")).hexdigest()[:6], 16)
        cta_headlines = [
            f"ENJOYED THIS LESSON ON {lesson_title[:24].upper()}? JOIN {channel_name.upper() or 'OUR COMMUNITY'}!",
            f"FOUND THIS HELPFUL? SHARE WITH A COLLEAGUE & SUBSCRIBE FOR MORE!",
            f"HAVE A QUESTION ON {lesson_title[:26].upper()}? DROP A COMMENT BELOW!",
            f"DON'T MISS THE NEXT MASTERCLASS — LIKE, SHARE & SUBSCRIBE TODAY!",
        ]
        cta_top_txt = cta_headlines[cta_seed % len(cta_headlines)]

        draw.rounded_rectangle([bx1, by1, bx1 + bw, by1 + bh], radius=22, fill=(10, 16, 34, alpha_o), outline=accent_alt + (alpha_o,), width=3)
        font_cta_top = _load_font(21, bold=True)
        draw.text((bx1 + 26, by1 + 12), cta_top_txt[:64], font=font_cta_top, fill=accent_alt + (alpha_o,))

        # 4 Animated Action Pill Badges: LIKE, COMMENT, SHARE, SUBSCRIBE
        pills = [
            ("👍 LIKE", (37, 99, 235)),
            ("💬 COMMENT", (16, 185, 129)),
            ("↗ SHARE", (139, 92, 246)),
            ("🔔 SUBSCRIBE", (239, 68, 68)),
        ]
        active_pill = int((4.2 - time_remaining) * 2.2) % len(pills)
        px_curr = bx1 + 24
        py_pill = by1 + 44
        pill_w = (bw - 48 - 36) // 4
        font_pill = _load_font(22, bold=True)
        for p_idx, (p_lbl, p_col) in enumerate(pills):
            is_act = (p_idx == active_pill)
            p_y_off = -3 if is_act else 0
            out_c = (255, 255, 255, alpha_o) if is_act else (accent + (alpha_o,))
            draw.rounded_rectangle(
                [px_curr, py_pill + p_y_off, px_curr + pill_w, py_pill + 46 + p_y_off],
                radius=12,
                fill=p_col + (alpha_o,),
                outline=out_c,
                width=3 if is_act else 1
            )
            draw.text((px_curr + 22, py_pill + 10 + p_y_off), p_lbl, font=font_pill, fill=(255, 255, 255, alpha_o))
            px_curr += pill_w + 12

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _render_animated_powerpoint_slide_step(
    lesson_title,
    slide_title,
    slide_notes,
    cf_pil_img,
    block_idx=0,
    total_blocks=1,
    slide_num=1,
    anim_progress=1.0,
    anim_style="typewriter",
    visual_template="broadcast_studio",
    chart_data=None
):
    """
    Renders an animated, broadcast-grade 1920x1080 PowerPoint Presentation Slide frame
    for `anim_progress` in [0.0 .. 1.0].
    Supported `anim_style` modes (automatically rotated across blocks & slides):
      - 'typewriter'      : Live typing effect with glowing cursor `▍` on heading & bullet notes
      - 'staggered_cards' : Numbered summary cards slide & fade in sequentially with active highlight
      - 'focus_sweep'     : Clean executive deck where an active spotlight sweeps across each takeaway
      - 'kinetic_chart'   : Animated growing bar/process/formula chart alongside staggered notes
    All cards stay strictly within y=164..904 so bottom captions (y=930..1060) never overlap!
    """
    W, H = 1920, 1080
    prog = max(0.0, min(1.0, float(anim_progress)))
    theme = resolve_studio_template(visual_template, block_idx=block_idx)
    accent = theme["accent"]
    accent_alt = theme["accent_alt"]

    theme_variant = block_idx + (0 if slide_num == 1 else 1)
    use_dark = (theme_variant % 2 == 1) or (visual_template in ("ted_keynote", "academic_chalkboard", "cyber_stem_lab"))
    bg_col = theme["slide_bg_dark"] if use_dark else theme["slide_bg_light"]
    is_bg_dark = (int(bg_col[0]) + int(bg_col[1]) + int(bg_col[2])) < 380

    card_col = (theme["card_bg"][0], theme["card_bg"][1], theme["card_bg"][2], 255) if is_bg_dark else (255, 255, 255, 255)
    text_col = (255, 255, 255, 255) if is_bg_dark else (15, 23, 42, 255)
    border_col = accent + (220,)

    slide = Image.new("RGB", (W, H), bg_col)
    draw = ImageDraw.Draw(slide, "RGBA")

    # Top PowerPoint Header Banner (y=0..146) with animated progress bar
    draw.rectangle([0, 0, W, 146], fill=theme["card_header_bg"])
    draw.rectangle([0, 138, W, 146], fill=(30, 41, 59, 255))
    prog_bar_w = int(W * max(0.06, min(1.0, prog)))
    draw.rectangle([0, 138, prog_bar_w, 146], fill=accent + (255,))

    font_course = _load_font(24, bold=True)
    font_heading = _load_font(44, bold=True)
    style_badge_map = {
        "typewriter": "LIVE NOTES",
        "staggered_cards": "KEY TAKEAWAYS",
        "focus_sweep": "CONCEPT BREAKDOWN",
        "kinetic_chart": "DATA & ANALYTICS",
    }
    mode_tag = style_badge_map.get(anim_style, "KEY SUMMARY")
    draw.text(
        (72, 18),
        f"{lesson_title.upper()[:54]}   •   SLIDE {block_idx + 1}.{slide_num} ({mode_tag})",
        font=font_course,
        fill=accent_alt + (255,)
    )

    full_heading = (slide_title or lesson_title or f"Module {block_idx + 1}").strip()
    if anim_style == "typewriter" and prog < 0.22:
        h_frac = max(0.15, prog / 0.22)
        h_chars = max(1, int(round(len(full_heading) * h_frac)))
        shown_heading = full_heading[:h_chars] + " ▍"
    else:
        shown_heading = full_heading
    draw.text((72, 56), shown_heading[:46], font=font_heading, fill=(255, 255, 255, 255))

    img_on_left = (theme["slide_img_side"] == "left") if (slide_num == 1) else (theme["slide_img_side"] != "left")
    if img_on_left:
        img_card_x1, img_card_x2 = 64, 856
        img_paste_x = 80
        notes_card_x1, notes_card_x2 = 884, 1856
    else:
        notes_card_x1, notes_card_x2 = 64, 1036
        img_card_x1, img_card_x2 = 1064, 1856
        img_paste_x = 1080

    notes_w = notes_card_x2 - notes_card_x1
    card_top_y, card_bot_y = 166, 902  # Strictly above y=904 caption safe zone!

    draw.rounded_rectangle([notes_card_x1, card_top_y, notes_card_x2, card_bot_y], radius=26, fill=card_col, outline=border_col, width=3)
    font_sub = _load_font(26, bold=True)
    draw.text(
        (notes_card_x1 + 38, card_top_y + 24),
        "KEY SUMMARY NOTES & TAKEAWAYS",
        font=font_sub,
        fill=accent_alt + (255,) if is_bg_dark else accent + (255,)
    )
    draw.line([notes_card_x1 + 38, card_top_y + 66, notes_card_x2 - 38, card_top_y + 66], fill=border_col, width=2)

    notes_list = [str(n).strip() for n in (slide_notes or []) if str(n).strip()][:4]
    if not notes_list:
        notes_list = [full_heading]
    n_items = len(notes_list)

    font_bullet = _load_font(36, bold=True)
    font_num = _load_font(27, bold=True)
    curr_y = card_top_y + 88
    badge_colors = [accent, (16, 185, 129), (245, 158, 11), (139, 92, 246)]

    active_idx = min(n_items - 1, int(prog * n_items))

    for i, clean in enumerate(notes_list):
        b_col = badge_colors[i % len(badge_colors)]

        # Compute per-bullet animation state based on anim_style
        if anim_style == "typewriter":
            # Distribute typing window across the full slide scene [0.05 .. 0.93]
            t_start = 0.05 + (i / float(max(1, n_items))) * 0.88
            t_span = 0.88 / float(max(1, n_items))
            if prog < t_start:
                continue
            char_frac = min(1.0, (prog - t_start) / max(0.08, t_span))
            vis_chars = max(1, int(round(len(clean) * char_frac)))
            is_typing_now = (char_frac < 0.99)
            display_text = clean[:vis_chars] + (" ▍" if is_typing_now else "")
            item_alpha = 255
            x_offset = 0
            highlight_box = (i == active_idx or is_typing_now)
        elif anim_style == "staggered_cards":
            t_start = 0.04 + (i / float(max(1, n_items))) * 0.80
            if prog < t_start:
                continue
            local_p = min(1.0, (prog - t_start) / 0.18)
            ease_p = math.sin(local_p * math.pi * 0.5)
            display_text = clean
            item_alpha = int(255 * ease_p)
            x_offset = int((1.0 - ease_p) * 42)
            highlight_box = (i == active_idx)
        else:
            # focus_sweep or kinetic_chart
            local_p = min(1.0, max(0.0, (prog * (n_items + 0.5)) - i * 0.45))
            if local_p <= 0.02:
                continue
            display_text = clean
            item_alpha = int(255 * min(1.0, local_p * 1.5))
            x_offset = 0
            highlight_box = (i == active_idx)

        wrapped = _wrap_text(draw, display_text, font_bullet, notes_w - 165)
        wrapped = wrapped[:3]
        row_h = max(72, len(wrapped) * 46 + 22)
        if curr_y + row_h > card_bot_y - 18:
            break

        # Draw subtle card row background & active spotlight border
        row_x1 = notes_card_x1 + 26 + x_offset
        row_x2 = notes_card_x2 - 26
        if highlight_box:
            row_bg = (30, 48, 86, min(230, item_alpha)) if is_bg_dark else (239, 246, 255, min(240, item_alpha))
            draw.rounded_rectangle(
                [row_x1, curr_y - 6, row_x2, curr_y + row_h - 8],
                radius=16,
                fill=row_bg,
                outline=accent_alt + (item_alpha,),
                width=3
            )
        else:
            row_bg = (20, 30, 56, min(145, item_alpha)) if is_bg_dark else (248, 250, 252, min(200, item_alpha))
            draw.rounded_rectangle(
                [row_x1, curr_y - 6, row_x2, curr_y + row_h - 8],
                radius=14,
                fill=row_bg
            )

        # Numbered badge
        draw.rounded_rectangle(
            [row_x1 + 14, curr_y + 4, row_x1 + 66, curr_y + 54],
            radius=13,
            fill=b_col + (item_alpha,)
        )
        draw.text((row_x1 + 31, curr_y + 11), str(i + 1), font=font_num, fill=(255, 255, 255, item_alpha))

        t_col_alpha = (text_col[0], text_col[1], text_col[2], item_alpha)
        ty = curr_y + 2
        for line in wrapped:
            draw.text((row_x1 + 84, ty), line, font=font_bullet, fill=t_col_alpha)
            ty += 45

        curr_y += row_h + 10

    # High-Res Visual Illustration OR Animated Data Chart / Formula Card (760x704 at y=166..902)
    draw.rounded_rectangle([img_card_x1, card_top_y, img_card_x2, card_bot_y], radius=26, fill=(15, 23, 42, 255), outline=accent + (225,), width=3)
    target_w, target_h = 760, 704

    if isinstance(chart_data, dict) and (chart_data.get("labels") or chart_data.get("formula")):
        try:
            chart_pil = render_data_chart_or_formula_card(
                chart_data, target_w, target_h, theme, slide_title=slide_title, anim_progress=prog
            )
            mask = Image.new("L", (target_w, target_h), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, target_w, target_h], radius=20, fill=255)
            slide.paste(chart_pil, (img_paste_x, card_top_y + 16), mask)
        except Exception as e:
            print(f"Animated chart card render note: {e}")
    elif cf_pil_img is not None:
        try:
            mask = Image.new("L", (target_w, target_h), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, target_w, target_h], radius=20, fill=255)
            slide.paste(cf_pil_img, (img_paste_x, card_top_y + 16), mask)
        except Exception as e:
            print(f"Slide image paste note: {e}")

    img_box = (img_paste_x, card_top_y + 16, img_paste_x + target_w, card_top_y + 16 + target_h)
    return cv2.cvtColor(np.array(slide), cv2.COLOR_RGB2BGR), img_box


def render_powerpoint_slide_frame(
    lesson_title,
    slide_title,
    slide_notes,
    cf_image_path,
    raw_avatar_path,
    output_png,
    block_idx=0,
    total_blocks=1,
    reveal_all=True,
    theme_variant=0,
    visual_template="broadcast_studio",
    chart_data=None
):
    """
    Renders a static 1920x1080 PowerPoint Presentation Slide PNG (backward compatible)
    and returns (output_png, img_box_coords).
    """
    cf_pil = None
    if cf_image_path and os.path.exists(cf_image_path):
        try:
            ill = Image.open(cf_image_path).convert("RGB")
            iw, ih = ill.size
            target_w, target_h = 760, 704
            scale = max(target_w / float(max(1, iw)), target_h / float(max(1, ih)))
            nw, nh = int(iw * scale), int(ih * scale)
            ill = ill.resize((nw, nh), Image.Resampling.LANCZOS)
            left = (nw - target_w) // 2
            top = (nh - target_h) // 2
            cf_pil = ill.crop((left, top, left + target_w, top + target_h))
        except Exception:
            cf_pil = None

    slide_bgr, img_box = _render_animated_powerpoint_slide_step(
        lesson_title=lesson_title,
        slide_title=slide_title,
        slide_notes=slide_notes,
        cf_pil_img=cf_pil,
        block_idx=block_idx,
        total_blocks=total_blocks,
        slide_num=2 if reveal_all else 1,
        anim_progress=1.0 if reveal_all else 0.65,
        anim_style="staggered_cards",
        visual_template=visual_template,
        chart_data=chart_data
    )
    cv2.imwrite(output_png, slide_bgr)
    return output_png, img_box


# ==============================================================================
# 5B. WEBSITE STEP-BY-STEP TUTORIAL & HOW-TO COMPOSITOR (ANIMATED CURSOR & WEBCAM)
# ==============================================================================

def _draw_realistic_mouse_cursor(draw, cx, cy, is_clicking=False, click_progress=0.0):
    """
    Renders an authentic, crisp desktop pointer mouse cursor at (cx, cy)
    with smooth drop shadow, white interior, black outline, and an optional
    expanding radiant click ripple wave (Loom/Camtasia style) when clicking.
    """
    # If clicking, draw expanding concentric click pulse / ripple rings
    if is_clicking and click_progress > 0.0:
        max_r = 42
        r = int(6 + (max_r - 6) * click_progress)
        alpha = int(220 * (1.0 - click_progress))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(56, 189, 248, alpha), width=3)
        r_inner = max(2, int(r * 0.45))
        draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], fill=(255, 255, 255, int(alpha * 0.7)))

    scale = 0.90 if is_clicking else 1.0
    pts = [
        (0, 0),
        (0, int(26 * scale)),
        (int(7 * scale), int(20 * scale)),
        (int(13 * scale), int(29 * scale)),
        (int(17 * scale), int(27 * scale)),
        (int(11 * scale), int(18 * scale)),
        (int(19 * scale), int(18 * scale))
    ]
    # Drop shadow
    shadow_pts = [(cx + 2 + px, cy + 3 + py) for px, py in pts]
    draw.polygon(shadow_pts, fill=(0, 0, 0, 95))
    # Crisp pointer body
    cursor_pts = [(cx + px, cy + py) for px, py in pts]
    draw.polygon(cursor_pts, fill=(255, 255, 255, 255), outline=(15, 23, 42, 255))


def render_tutorial_walkthrough_screen(
    lesson_title,
    slide_title,
    slide_notes,
    tutorial_url="",
    tutorial_step=1,
    tutorial_action="",
    f_idx=0,
    num_frames=240,
    fps=24,
    pres_bgr=None,
    visual_template="broadcast_studio",
    theme=None,
    aspect_ratio="16:9",
    presenter_name="",
    channel_name="",
    step_image_pil=None,
    step_video_frame=None,
    step_image_path=None
):
    """
    Renders an authentic, broadcast-grade Website Step-by-Step Tutorial Walkthrough scene:
      1. Realistic Browser Window Mockup with MacOS/Windows traffic lights, active tab,
         SSL lock icon and clean step URL.
      2. Dynamic Top-to-Bottom Smooth Page Scroll: Viewport scrolls naturally down
         through the page context (header, hero, custom screenshots/video demos, notes)
         until it smoothly centers on the targeted interactive button/action!
      3. Media Integration (avoiding boredom): Displays the user's custom step screenshot
         or live recorded screen video demo directly in a high-tech studio media player frame.
      4. Crisp moving mouse cursor that glides smoothly toward the target interactive element,
         clicks with an expanding ripple wave, and triggers an active clicked button state!
      5. Studio Teacher Avatar webcam frame with glowing border ring and nameplate,
         perfectly lip-synced and speaking with the audio.
    """
    if theme is None:
        theme = resolve_studio_template(visual_template, block_idx=tutorial_step - 1)

    is_vertical = (aspect_ratio == "9:16")
    W = 1080 if is_vertical else 1920
    H = 1920 if is_vertical else 1080

    accent = theme.get("accent", (56, 189, 248))
    accent_alt = theme.get("accent_alt", (245, 158, 11))

    # Base background: deep studio backdrop
    bg_start = theme.get("bg_gradient_start", (15, 23, 42))
    bg_end = theme.get("bg_gradient_end", (30, 41, 59))
    base_bg = np.zeros((H, W, 3), dtype=np.uint8)
    for y_i in range(H):
        ratio = y_i / float(H)
        col = [
            int(bg_start[c] * (1.0 - ratio) + bg_end[c] * ratio)
            for c in range(3)
        ]
        base_bg[y_i, :] = col

    # Clean URL handling
    clean_url = str(tutorial_url or "https://example.com/portal/step-1").strip()
    if not re.match(r"^https?://", clean_url, re.I):
        clean_url = "https://" + clean_url
    domain_match = re.search(r"https?://([^/]+)", clean_url, re.I)
    domain_str = domain_match.group(1) if domain_match else "website.com"

    act_label = str(tutorial_action or "").strip()
    if not act_label:
        act_label = f"Click to Proceed with Step {tutorial_step}"

    font_xs = _load_font(18, bold=False)
    font_sm = _load_font(22, bold=True)
    font_med = _load_font(28, bold=True)
    font_lg = _load_font(34, bold=True)

    # Lazily load image if path provided and PIL not yet ready
    if step_image_pil is None and step_image_path and os.path.exists(step_image_path):
        try:
            step_image_pil = Image.open(step_image_path).convert("RGB")
        except Exception:
            step_image_pil = None

    if not is_vertical:
        # =========================================================================
        # 16:9 Widescreen Broadcast Layout
        # =========================================================================
        bx, by = 60, 48
        bw, bh = 1320, 864
        vx = bx + 2
        vy = by + 52
        vw = bw - 4      # 1316 px
        vh = bh - 54     # 810 px

        # Outer background
        out_bgr = base_bg.copy()

        # Browser Chrome Outer Card
        pil_frame = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
        draw_outer = ImageDraw.Draw(pil_frame, "RGBA")
        draw_outer.rounded_rectangle([bx, by, bx + bw, by + bh], radius=14, fill=(248, 250, 252, 255), outline=(71, 85, 105, 220), width=2)
        # Browser Header Bar
        draw_outer.rounded_rectangle([bx, by, bx + bw, by + 52], radius=14, fill=(30, 41, 59, 255))
        draw_outer.rectangle([bx, by + 36, bx + bw, by + 52], fill=(30, 41, 59, 255))

        # Traffic light dots
        draw_outer.ellipse([bx + 18, by + 20, bx + 30, by + 32], fill=(239, 68, 68, 255))
        draw_outer.ellipse([bx + 38, by + 20, bx + 50, by + 32], fill=(245, 158, 11, 255))
        draw_outer.ellipse([bx + 58, by + 20, bx + 70, by + 32], fill=(16, 185, 129, 255))

        # Active Tab
        draw_outer.rounded_rectangle([bx + 90, by + 10, bx + 360, by + 48], radius=8, fill=(15, 23, 42, 255))
        draw_outer.text((bx + 106, by + 16), f"🌐 {domain_str[:22]}", font=font_xs, fill=(226, 232, 240, 255))

        # URL Address Bar
        url_bx1 = bx + 380
        url_bx2 = bx + bw - 24
        draw_outer.rounded_rectangle([url_bx1, by + 10, url_bx2, by + 44], radius=8, fill=(15, 23, 42, 255), outline=(51, 65, 85, 255), width=1)
        draw_outer.text((url_bx1 + 14, by + 16), f"🔒 {clean_url[:64]}", font=font_xs, fill=(56, 189, 248, 255))

        out_bgr = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

        # -------------------------------------------------------------------------
        # Build Virtual Tall Webpage Canvas (1316 x 1460)
        # -------------------------------------------------------------------------
        pw = vw
        ph = 1460
        page_pil = Image.new("RGBA", (pw, ph), (248, 250, 252, 255))
        page_draw = ImageDraw.Draw(page_pil, "RGBA")

        # 1. Webpage Top Navigation Bar
        page_draw.rectangle([0, 0, pw, 58], fill=(255, 255, 255, 255), outline=(226, 232, 240, 255), width=1)
        page_draw.text((28, 14), f"⚡ {domain_str.upper()[:22]} PORTAL", font=font_med, fill=(15, 23, 42, 255))
        page_draw.text((360, 20), "Overview   •   Guides   •   Interactive Setup   •   Support", font=font_xs, fill=(100, 116, 139, 255))
        page_draw.rounded_rectangle([pw - 180, 12, pw - 24, 44], radius=6, fill=(240, 253, 244, 255), outline=(187, 247, 208, 255))
        page_draw.text((pw - 166, 18), "● Live Session", font=font_xs, fill=(22, 163, 74, 255))

        # 2. Hero Breadcrumbs & Title
        page_draw.text((36, 74), f"Home  /  Tutorial Guides  /  Step {tutorial_step}", font=font_xs, fill=(148, 163, 184, 255))
        page_draw.text((36, 100), str(slide_title or f"Step {tutorial_step}: Interactive Guide")[:50], font=font_lg, fill=(15, 23, 42, 255))
        page_draw.text((36, 142), f"Follow this step-by-step walkthrough on {domain_str} • Step {tutorial_step}", font=font_xs, fill=(100, 116, 139, 255))

        # 3. Rich Step Visual Area (Video / Picture Demo to avoid boredom!)
        mx, my = 36, 172
        mw, mh = pw - 72, 380
        page_draw.rounded_rectangle([mx, my, mx + mw, my + mh], radius=12, fill=(15, 23, 42, 255), outline=(51, 65, 85, 255), width=1)

        has_custom_visual = False
        if step_video_frame is not None:
            # Custom Walkthrough Screen Recording Video Frame
            try:
                vid_rgb = cv2.cvtColor(step_video_frame, cv2.COLOR_BGR2RGB)
                v_h, v_w = vid_rgb.shape[:2]
                max_vw, max_vh = mw - 16, mh - 50
                scale_v = min(max_vw / float(max(1, v_w)), max_vh / float(max(1, v_h)))
                target_vw, target_vh = int(v_w * scale_v), int(v_h * scale_v)
                vid_resized = cv2.resize(vid_rgb, (target_vw, target_vh), interpolation=cv2.INTER_LINEAR)
                vid_pil = Image.fromarray(vid_resized)
                pos_vx = mx + (mw - target_vw) // 2
                pos_vy = my + 34 + (max_vh - target_vh) // 2
                page_pil.paste(vid_pil, (pos_vx, pos_vy))

                # Video HUD Overlay
                page_draw.rounded_rectangle([mx + 16, my + 8, mx + 200, my + 30], radius=4, fill=(220, 38, 38, 230))
                page_draw.text((mx + 24, my + 10), "🔴 LIVE SCREEN RECORDING", font=font_xs, fill=(255, 255, 255, 255))
                # Progress scrubber
                v_prog = min(1.0, max(0.0, f_idx / float(max(1, num_frames - 1))))
                page_draw.rounded_rectangle([mx + 16, my + mh - 12, mx + mw - 16, my + mh - 8], radius=2, fill=(51, 65, 85, 255))
                page_draw.rounded_rectangle([mx + 16, my + mh - 12, mx + 16 + int((mw - 32) * v_prog), my + mh - 8], radius=2, fill=(239, 68, 68, 255))
                has_custom_visual = True
            except Exception:
                has_custom_visual = False

        elif step_image_pil is not None:
            # Custom Step Screenshot / Image
            try:
                img_w, img_h = step_image_pil.size
                max_iw, max_ih = mw - 16, mh - 50
                scale_i = min(max_iw / float(max(1, img_w)), max_ih / float(max(1, img_h)))
                target_iw, target_ih = int(img_w * scale_i), int(img_h * scale_i)
                img_resized = step_image_pil.resize((target_iw, target_ih), Image.Resampling.LANCZOS)
                pos_ix = mx + (mw - target_iw) // 2
                pos_iy = my + 34 + (max_ih - target_ih) // 2
                page_pil.paste(img_resized, (pos_ix, pos_iy))

                # Photo Badge
                page_draw.rounded_rectangle([mx + 16, my + 8, mx + 220, my + 30], radius=4, fill=(14, 165, 233, 230))
                page_draw.text((mx + 24, my + 10), "📷 STEP DEMONSTRATION", font=font_xs, fill=(255, 255, 255, 255))
                has_custom_visual = True
            except Exception:
                has_custom_visual = False

        if not has_custom_visual:
            # High-tech dynamic web application dashboard mockup
            page_draw.rounded_rectangle([mx + 16, my + 12, mx + 260, my + 36], radius=4, fill=accent + (220,))
            page_draw.text((mx + 24, my + 14), f"PORTAL INTERACTION VIEW • STEP {tutorial_step}", font=font_xs, fill=(255, 255, 255, 255))
            # 3 simulated metric cards
            card_w = (mw - 48) // 3
            for c_i, (c_lbl, c_val, c_color) in enumerate([
                ("Target Action", act_label[:20], accent),
                ("Current Step", f"Step {tutorial_step}", accent_alt),
                ("System Verification", "Live Verified", (34, 197, 94))
            ]):
                card_x = mx + 16 + c_i * (card_w + 8)
                card_y = my + 54
                page_draw.rounded_rectangle([card_x, card_y, card_x + card_w, card_y + 110], radius=8, fill=(30, 41, 59, 255), outline=(51, 65, 85, 255))
                page_draw.text((card_x + 14, card_y + 14), c_lbl, font=font_xs, fill=(148, 163, 184, 255))
                page_draw.text((card_x + 14, card_y + 44), c_val, font=font_med, fill=c_color + (255,))

            # Decorative wireframe mockup rows
            for r_i in range(3):
                ry = my + 185 + r_i * 44
                page_draw.rounded_rectangle([mx + 16, ry, mx + mw - 16, ry + 32], radius=6, fill=(30, 41, 59, 220))
                page_draw.rectangle([mx + 26, ry + 10, mx + 140, ry + 22], fill=(71, 85, 105, 255))
                page_draw.rectangle([mx + 160, ry + 12, mx + mw - 100, ry + 20], fill=(51, 65, 85, 255))
                page_draw.rounded_rectangle([mx + mw - 80, ry + 8, mx + mw - 24, ry + 24], radius=4, fill=accent + (180,))

        # 4. Contextual Checklist Notes
        page_draw.rounded_rectangle([36, 574, 380, 614], radius=6, fill=accent + (255,))
        page_draw.text((50, 582), f"STEP {tutorial_step} • ACTION CHECKLIST", font=font_sm, fill=(255, 255, 255, 255))

        notes_arr = [str(n).strip() for n in (slide_notes or []) if str(n).strip()]
        if not notes_arr:
            notes_arr = [act_label, f"Follow on-screen instructions on {domain_str}", "Confirm changes to complete step"]

        ny = 632
        for note_item in notes_arr[:3]:
            page_draw.rounded_rectangle([36, ny, pw - 36, ny + 44], radius=8, fill=(255, 255, 255, 255), outline=(226, 232, 240, 255))
            page_draw.rounded_rectangle([48, ny + 8, 76, ny + 36], radius=14, fill=(240, 253, 244, 255))
            page_draw.text((54, ny + 10), "✓", font=font_sm, fill=(22, 163, 74, 255))
            page_draw.text((88, ny + 10), note_item[:72], font=font_sm, fill=(51, 65, 85, 255))
            ny += 54

        # 5. Interactive Target Callout Card & Big Button
        cx1 = 36
        cy1 = 810
        cw = pw - 72
        ch = 180

        scroll_frames = max(18, int(num_frames * 0.42))
        click_frame = max(scroll_frames + 6, int(num_frames * 0.50))
        is_click_moment = (f_idx >= click_frame and f_idx <= click_frame + 20)
        is_after_click = (f_idx > click_frame + 20)

        # Highlight callout box
        callout_fill = (240, 253, 244, 255) if is_after_click else (240, 249, 255, 255)
        callout_border = (187, 247, 208, 255) if is_after_click else (186, 230, 253, 255)
        page_draw.rounded_rectangle([cx1, cy1, cx1 + cw, cy1 + ch], radius=12, fill=callout_fill, outline=callout_border, width=2)
        callout_badge = "✓ STEP ACTION VERIFIED & COMPLETED" if is_after_click else "👇 REQUIRED ACTION TO COMPLETE THIS STEP:"
        callout_badge_col = (22, 163, 74, 255) if is_after_click else (3, 105, 161, 255)
        page_draw.text((cx1 + 24, cy1 + 16), callout_badge, font=font_sm, fill=callout_badge_col)

        # Target Interactive Button
        tx = cx1 + 24
        ty = cy1 + 52
        tw = min(680, cw - 48)
        th = 76

        if is_click_moment or is_after_click:
            btn_fill = (22, 163, 74, 255)  # Success green
            btn_border = (187, 247, 208, 255)
            btn_text = f"✓ COMPLETED: {act_label[:40]}"
        else:
            btn_fill = (2, 132, 199, 255)  # Primary bright blue
            btn_border = (125, 211, 252, 255)
            btn_text = f"👉 CLICK HERE: {act_label[:40]} →"

        page_draw.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=10, fill=btn_fill, outline=btn_border, width=2)
        page_draw.text((tx + 24, ty + 20), btn_text, font=font_med, fill=(255, 255, 255, 255))
        page_draw.text((tx, ty + 84), f"Target URL: {clean_url[:60]} • Automatic Validation", font=font_xs, fill=(100, 116, 139, 255))

        # 6. Page Footer
        page_draw.rounded_rectangle([36, 1020, pw - 36, 1120], radius=8, fill=(255, 255, 255, 255), outline=(226, 232, 240, 255))
        page_draw.text((54, 1036), f"Next Module: Proceeding to Step {tutorial_step + 1} automatically...", font=font_sm, fill=(71, 85, 105, 255))
        page_draw.text((54, 1068), f"Interactive course syllabus verified • {domain_str}", font=font_xs, fill=(148, 163, 184, 255))

        # -------------------------------------------------------------------------
        # Dynamic Top-to-Bottom Smooth Page Scroll
        # -------------------------------------------------------------------------
        # Target scroll brings the button at ty=810 into center-view of viewport (vh=810)
        target_scroll_y = 390
        if f_idx < scroll_frames:
            scroll_prog = f_idx / float(scroll_frames)
            ease_scroll = 0.5 - 0.5 * math.cos(math.pi * scroll_prog)
            current_scroll_y = int(target_scroll_y * ease_scroll)
        else:
            current_scroll_y = target_scroll_y

        # Slicing the tall virtual webpage at current_scroll_y
        view_crop = page_pil.crop((0, current_scroll_y, vw, current_scroll_y + vh))
        view_bgr = cv2.cvtColor(np.array(view_crop), cv2.COLOR_RGBA2BGR)
        out_bgr[vy:vy + vh, vx:vx + vw] = view_bgr

        # Browser Scrollbar
        sb_x = vx + vw - 10
        sb_h = max(60, int(vh * (vh / float(ph))))
        sb_travel = vh - sb_h - 16
        sb_y = vy + 8 + int(sb_travel * (current_scroll_y / float(target_scroll_y or 1)))
        cv2.rectangle(out_bgr, (sb_x, sb_y), (sb_x + 6, sb_y + sb_h), (148, 163, 184), -1, lineType=cv2.LINE_AA)

        # -------------------------------------------------------------------------
        # Animated Mouse Cursor & Radiant Click Ripple
        # -------------------------------------------------------------------------
        pil_cursor = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
        draw_cur = ImageDraw.Draw(pil_cursor, "RGBA")

        btn_screen_cx = vx + tx + tw // 2
        btn_screen_cy = vy + (ty - current_scroll_y) + th // 2
        cur_start_x = vx + 160
        cur_start_y = vy + vh - 60

        cursor_reach = max(scroll_frames, int(num_frames * 0.46))

        if f_idx < cursor_reach:
            c_prog = f_idx / float(cursor_reach)
            c_ease = math.sin(c_prog * math.pi / 2.0)
            cx = cur_start_x + (btn_screen_cx - cur_start_x) * c_ease
            cy = cur_start_y + (btn_screen_cy - cur_start_y) * c_ease
            _draw_realistic_mouse_cursor(draw_cur, int(cx), int(cy), is_clicking=False)
        elif is_click_moment:
            click_prog = (f_idx - click_frame) / 20.0
            _draw_realistic_mouse_cursor(draw_cur, btn_screen_cx, btn_screen_cy, is_clicking=True, click_progress=click_prog)
        else:
            drift_x = 8.0 * math.sin((f_idx - click_frame) * 0.08)
            drift_y = 4.0 * math.cos((f_idx - click_frame) * 0.06)
            _draw_realistic_mouse_cursor(draw_cur, int(btn_screen_cx + drift_x), int(btn_screen_cy + drift_y), is_clicking=False)

        out_bgr = cv2.cvtColor(np.array(pil_cursor), cv2.COLOR_RGB2BGR)

        # -------------------------------------------------------------------------
        # Studio Teacher Avatar Frame (Right side)
        # -------------------------------------------------------------------------
        if pres_bgr is not None:
            pw_c, ph_c = 470, 560
            px_c = 1410
            py_c = 70

            pres_resized = cv2.resize(pres_bgr, (pw_c, ph_c), interpolation=cv2.INTER_LANCZOS4)
            mask_r = _create_rounded_mask(pw_c, ph_c, radius=20)
            alpha_3d = mask_r[:, :, None]

            card_bgr = np.full((ph_c, pw_c, 3), (15, 23, 42), dtype=np.uint8)
            comp_pres = (pres_resized.astype(np.float32) * alpha_3d + card_bgr.astype(np.float32) * (1.0 - alpha_3d)).astype(np.uint8)

            out_bgr[py_c:py_c + ph_c, px_c:px_c + pw_c] = comp_pres
            cv2.rectangle(out_bgr, (px_c, py_c), (px_c + pw_c, py_c + ph_c), (accent[2], accent[1], accent[0]), 3, lineType=cv2.LINE_AA)

            # Nameplate
            ny1 = py_c + ph_c + 14
            cv2.rectangle(out_bgr, (px_c, ny1), (px_c + pw_c, ny1 + 68), (15, 23, 42), -1)
            cv2.rectangle(out_bgr, (px_c, ny1), (px_c + pw_c, ny1 + 68), (71, 85, 105), 1)

            pres_disp = presenter_name if presenter_name else "Course Instructor"
            chan_disp = channel_name if channel_name else "Video Tutorial Studio"
            cv2.putText(out_bgr, pres_disp[:32], (px_c + 16, ny1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(out_bgr, f"LIVE WALKTHROUGH  •  {chan_disp[:24]}", (px_c + 16, ny1 + 54), cv2.FONT_HERSHEY_DUPLEX, 0.48, (accent[2], accent[1], accent[0]), 1, cv2.LINE_AA)

        return out_bgr

    else:
        # =========================================================================
        # 9:16 Vertical Shorts / TikTok Layout
        # =========================================================================
        # Top 42% shows Teacher Avatar speaking
        if pres_bgr is not None:
            pres_h = 760
            pres_w = W
            pres_resized = cv2.resize(pres_bgr, (pres_w, pres_h), interpolation=cv2.INTER_LANCZOS4)
            base_bg[0:pres_h, 0:pres_w] = pres_resized

        cv2.rectangle(base_bg, (0, 700), (W, 760), (15, 23, 42), -1)
        pres_disp = presenter_name if presenter_name else "Course Instructor"
        cv2.putText(base_bg, f"TUTORIAL: {slide_title[:38]}", (32, 742), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        bx, by = 36, 790
        bw, bh = W - 72, 980
        vx = bx + 2
        vy = by + 50
        vw = bw - 4
        vh = bh - 52

        out_bgr = base_bg.copy()
        pil_frame = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
        draw_outer = ImageDraw.Draw(pil_frame, "RGBA")
        draw_outer.rounded_rectangle([bx, by, bx + bw, by + bh], radius=14, fill=(248, 250, 252, 255), outline=(71, 85, 105, 220), width=2)
        draw_outer.rounded_rectangle([bx, by, bx + bw, by + 50], radius=14, fill=(30, 41, 59, 255))
        draw_outer.rectangle([bx, by + 34, bx + bw, by + 50], fill=(30, 41, 59, 255))
        draw_outer.ellipse([bx + 18, by + 20, bx + 28, by + 30], fill=(239, 68, 68, 255))
        draw_outer.ellipse([bx + 36, by + 20, bx + 46, by + 30], fill=(245, 158, 11, 255))
        draw_outer.ellipse([bx + 54, by + 20, bx + 64, by + 30], fill=(16, 185, 129, 255))
        draw_outer.text((bx + 80, by + 16), f"🔒 {clean_url[:42]}", font=font_xs, fill=(56, 189, 248, 255))
        out_bgr = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

        # Tall mobile webpage canvas
        pw = vw
        ph = 1500
        page_pil = Image.new("RGBA", (pw, ph), (248, 250, 252, 255))
        page_draw = ImageDraw.Draw(page_pil, "RGBA")

        page_draw.rounded_rectangle([20, 20, 280, 60], radius=6, fill=accent + (255,))
        page_draw.text((32, 28), f"STEP {tutorial_step} WALKTHROUGH", font=font_sm, fill=(255, 255, 255, 255))
        page_draw.text((20, 76), str(slide_title or f"Step {tutorial_step}")[:40], font=font_lg, fill=(15, 23, 42, 255))

        # Mobile Visual Demo (Video or Picture)
        mx, my = 20, 130
        mw, mh = pw - 40, 360
        page_draw.rounded_rectangle([mx, my, mx + mw, my + mh], radius=10, fill=(15, 23, 42, 255), outline=(51, 65, 85, 255))

        if step_video_frame is not None:
            try:
                vid_rgb = cv2.cvtColor(step_video_frame, cv2.COLOR_BGR2RGB)
                v_h, v_w = vid_rgb.shape[:2]
                sc_v = min((mw - 16) / float(max(1, v_w)), (mh - 40) / float(max(1, v_h)))
                tvw, tvh = int(v_w * sc_v), int(v_h * sc_v)
                v_resized = cv2.resize(vid_rgb, (tvw, tvh), interpolation=cv2.INTER_LINEAR)
                page_pil.paste(Image.fromarray(v_resized), (mx + (mw - tvw) // 2, my + 24 + (mh - 40 - tvh) // 2))
                page_draw.text((mx + 16, my + 8), "🔴 LIVE RECORDING", font=font_xs, fill=(239, 68, 68, 255))
            except Exception:
                pass
        elif step_image_pil is not None:
            try:
                iw, ih = step_image_pil.size
                sc_i = min((mw - 16) / float(max(1, iw)), (mh - 40) / float(max(1, ih)))
                tiw, tih = int(iw * sc_i), int(ih * sc_i)
                i_res = step_image_pil.resize((tiw, tih), Image.Resampling.LANCZOS)
                page_pil.paste(i_res, (mx + (mw - tiw) // 2, my + 24 + (mh - 40 - tih) // 2))
                page_draw.text((mx + 16, my + 8), "📷 STEP DEMO", font=font_xs, fill=(56, 189, 248, 255))
            except Exception:
                pass

        # Mobile Notes
        notes_arr = [str(n).strip() for n in (slide_notes or []) if str(n).strip()]
        ny = 510
        for n_item in (notes_arr or [act_label])[:3]:
            page_draw.rounded_rectangle([20, ny, pw - 20, ny + 46], radius=6, fill=(255, 255, 255, 255), outline=(226, 232, 240, 255))
            page_draw.text((32, ny + 12), f"• {n_item[:50]}", font=font_sm, fill=(51, 65, 85, 255))
            ny += 54

        # Target Button
        tx = 24
        ty = 710
        tw = pw - 48
        th = 84

        scroll_frames = max(18, int(num_frames * 0.42))
        click_frame = max(scroll_frames + 6, int(num_frames * 0.50))
        is_click_moment = (f_idx >= click_frame and f_idx <= click_frame + 20)
        is_after_click = (f_idx > click_frame + 20)

        if is_click_moment or is_after_click:
            page_draw.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=10, fill=(22, 163, 74, 255), outline=(187, 247, 208, 255), width=2)
            page_draw.text((tx + 20, ty + 26), f"✓ CLICKED: {act_label[:34]}", font=font_med, fill=(255, 255, 255, 255))
        else:
            page_draw.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=10, fill=(2, 132, 199, 255), outline=(125, 211, 252, 255), width=2)
            page_draw.text((tx + 20, ty + 26), f"👉 CLICK: {act_label[:34]} →", font=font_med, fill=(255, 255, 255, 255))

        # Dynamic scroll in vertical mode
        target_scroll_y = 300
        if f_idx < scroll_frames:
            s_prog = f_idx / float(scroll_frames)
            ease_s = 0.5 - 0.5 * math.cos(math.pi * s_prog)
            cur_scroll_y = int(target_scroll_y * ease_s)
        else:
            cur_scroll_y = target_scroll_y

        v_crop = page_pil.crop((0, cur_scroll_y, vw, cur_scroll_y + vh))
        out_bgr[vy:vy + vh, vx:vx + vw] = cv2.cvtColor(np.array(v_crop), cv2.COLOR_RGBA2BGR)

        # Cursor in vertical mode
        pil_vcur = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
        draw_vcur = ImageDraw.Draw(pil_vcur, "RGBA")
        btn_cx = vx + tx + tw // 2
        btn_cy = vy + (ty - cur_scroll_y) + th // 2
        start_cx, start_cy = vx + 100, vy + vh - 80
        if f_idx < click_frame:
            t_prog = f_idx / float(max(1, click_frame))
            ease = math.sin(t_prog * math.pi / 2.0)
            cur_x = start_cx + (btn_cx - start_cx) * ease
            cur_y = start_cy + (btn_cy - start_cy) * ease
            _draw_realistic_mouse_cursor(draw_vcur, int(cur_x), int(cur_y), is_clicking=False)
        elif is_click_moment:
            c_prog = (f_idx - click_frame) / 20.0
            _draw_realistic_mouse_cursor(draw_vcur, btn_cx, btn_cy, is_clicking=True, click_progress=c_prog)
        else:
            _draw_realistic_mouse_cursor(draw_vcur, btn_cx, btn_cy, is_clicking=False)

        return cv2.cvtColor(np.array(pil_vcur), cv2.COLOR_RGB2BGR)


# ==============================================================================
# 6. RESCALED MULTI-SCENE DYNAMIC DIRECTOR (WORKS FOR ANY VIDEO DURATION!)
# ==============================================================================

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


def _build_broll_overlay_rgba(lesson_title, slide_title, note_text, block_idx, total_blocks, visual_template="broadcast_studio"):
    W, H = 1920, 1080
    theme = resolve_studio_template(visual_template, block_idx=block_idx)
    accent = theme["accent"]
    accent_alt = theme["accent_alt"]

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov, "RGBA")

    # Top floating studio pill header (enlarged accessibility font)
    draw.rounded_rectangle([64, 22, 1856, 118], radius=22, fill=theme["header_bg"], outline=accent + (215,), width=3)
    font_tag = _load_font(23, bold=True)
    font_head = _load_font(39, bold=True)
    draw.text((96, 33), f"LIVE VISUAL CLIP   •   SECTION {block_idx + 1} OF {total_blocks}", font=font_tag, fill=accent_alt + (255,))
    draw.text((96, 63), (slide_title or lesson_title)[:48], font=font_head, fill=(255, 255, 255, 255))

    # Floating Key Note Callout Card kept above y=900 so bottom captions never collide
    if note_text:
        c_x1, c_x2 = 64, 1380
        draw.rounded_rectangle([c_x1, 718, c_x2, 898], radius=22, fill=(10, 15, 30, 236), outline=accent + (230,), width=3)
        draw.rounded_rectangle([c_x1, 718, c_x1 + 22, 898], radius=8, fill=accent_alt + (255,))
        font_note = _load_font(36, bold=True)
        lines = _wrap_text(draw, note_text, font_note, (c_x2 - c_x1) - 76)
        y = 740
        for line in lines[:3]:
            draw.text((c_x1 + 44, y), line, font=font_note, fill=(255, 255, 255, 255))
            y += 47

    ov_np = np.array(ov)
    rgb = ov_np[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    alpha = (ov_np[:, :, 3].astype(np.float32) / 255.0)[..., None]
    return bgr, alpha


def _overlay_pip_presenter(canvas_bgr, presenter_frame_bgr, pos="bottom_right", border_rgb=(56, 189, 248)):
    """
    Disabled small bottom Picture-in-Picture presenter box so the lip-sync / avatar
    never appears twice or as a small floating mouth at the bottom of the video.
    """
    return canvas_bgr


def _apply_organic_camera_breathing(frame_bgr, f_idx, fps=24):
    """
    Applies a subtle, natural 0.35% handheld camera focal drift across every frame
    so no pixel grid remains mathematically frozen—making the video look like a
    genuine human camera broadcast and preventing YouTube AI-static detection.
    """
    t = f_idx / float(max(1, fps))
    zoom = 1.004 + 0.003 * math.sin(2.0 * math.pi * 0.17 * t)
    tx = 2.2 * math.sin(2.0 * math.pi * 0.13 * t)
    ty = 1.4 * math.cos(2.0 * math.pi * 0.19 * t)
    H, W = frame_bgr.shape[:2]
    M = np.float32([
        [zoom, 0.0, (1.0 - zoom) * (W * 0.5) + tx],
        [0.0, zoom, (1.0 - zoom) * (H * 0.5) + ty]
    ])
    return cv2.warpAffine(frame_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _render_bokeh_particles(canvas_bgr, f_idx, fps=24):
    """
    Returns clean, unblemished B-roll footage without artificial particle dots/specks.
    """
    return canvas_bgr


def _compose_vertical_916_frame(widescreen_bgr, pres_bgr, lesson_title, slide_title, slide_notes, theme, block_idx, total_blocks):
    """
    Composes a 1080x1920 (9:16 Vertical Shorts / TikTok / Reels) broadcast layout:
      - Top header banner (y=0..116)
      - Upper stage (y=124..1016): Live Lip-Synced Teacher Portrait + Concept Card
      - Lower stage (y=1028..1640): Widescreen visual slide / B-Roll / Chart card
      - Bottom caption safe zone (y=1640..1920)
    """
    VW, VH = 1080, 1920
    canvas = np.full((VH, VW, 3), (theme["bg_tint"][2], theme["bg_tint"][1], theme["bg_tint"][0]), dtype=np.uint8)
    accent_bgr = (int(theme["accent"][2]), int(theme["accent"][1]), int(theme["accent"][0]))

    # Upper Presenter Stage (centered 520x628 portrait at top)
    pw, ph = 520, 628
    pres_resized = cv2.resize(pres_bgr, (pw, ph), interpolation=cv2.INTER_LINEAR)
    px = (VW - pw) // 2
    py = 136
    cv2.rectangle(canvas, (px - 4, py - 4), (px + pw + 4, py + ph + 4), accent_bgr, 4, cv2.LINE_AA)
    canvas[py:py + ph, px:px + pw] = pres_resized

    # Lower Visual / Slide Stage (1000x562 at y=800..1362)
    vw, vh = 1000, 562
    vis_resized = cv2.resize(widescreen_bgr, (vw, vh), interpolation=cv2.INTER_AREA)
    vx = (VW - vw) // 2
    vy = 804
    cv2.rectangle(canvas, (vx - 4, vy - 4), (vx + vw + 4, vy + vh + 4), accent_bgr, 3, cv2.LINE_AA)
    canvas[vy:vy + vh, vx:vx + vw] = vis_resized

    # Top Title Banner
    pil_c = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_c, "RGBA")
    draw.rounded_rectangle([36, 28, VW - 36, 112], radius=18, fill=theme["header_bg"], outline=theme["accent"] + (220,), width=3)
    f_top = _load_font(28, bold=True)
    draw.text((64, 52), (slide_title or lesson_title)[:34].upper(), font=f_top, fill=(255, 255, 255, 255))

    # Key Takeaway Card (y=1392..1630)
    if slide_notes:
        draw.rounded_rectangle([40, 1392, VW - 40, 1630], radius=22, fill=theme["card_bg"], outline=theme["accent"] + (220,), width=3)
        f_note = _load_font(33, bold=True)
        n_lines = _wrap_text(draw, str(slide_notes[0]), f_note, VW - 140)
        ny = 1424
        for nl in n_lines[:3]:
            draw.text((72, ny), nl, font=f_note, fill=theme["card_text"])
            ny += 46

    return cv2.cvtColor(np.array(pil_c), cv2.COLOR_RGB2BGR)


class _PexelsVideoStream:
    """
    Streams frames from a downloaded Pexels HD .mp4 video clip, automatically
    scaling/cropping to 1920x1080 (16:9) and looping seamlessly if the scene
    exceeds the clip duration.
    """
    def __init__(self, video_path, target_w=1920, target_h=1080):
        self.video_path = video_path
        self.target_w = target_w
        self.target_h = target_h
        self.cap = None
        if video_path and os.path.exists(video_path) and os.path.getsize(video_path) > 5000:
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    self.cap = cap
            except Exception as e:
                print(f"Pexels VideoCapture warning for {video_path}: {e}")

    def is_valid(self):
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self):
        if not self.is_valid():
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        ih, iw = frame.shape[:2]
        if iw == self.target_w and ih == self.target_h:
            return frame
        scale = max(self.target_w / float(max(1, iw)), self.target_h / float(max(1, ih)))
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x0 = max(0, (nw - self.target_w) // 2)
        y0 = max(0, (nh - self.target_h) // 2)
        return resized[y0:y0 + self.target_h, x0:x0 + self.target_w].copy()

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


def _compute_block_ckpt_r2_key(lesson_title, slide_title, block_idx, total_blocks, aspect_ratio, visual_template, audio_wav_path, suffix="raw"):
    wav_sz = 0
    if audio_wav_path and os.path.exists(audio_wav_path):
        try:
            wav_sz = os.path.getsize(audio_wav_path)
        except Exception:
            wav_sz = 0
    sig = f"v763|{lesson_title}|{slide_title}|{block_idx}|{total_blocks}|{aspect_ratio}|{visual_template}|{wav_sz}|{suffix}"
    sig_hash = hashlib.md5(sig.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"checkpoints/{sig_hash}/block_{block_idx}_{suffix}.mp4"


def render_dynamic_lesson_block(
    raw_avatar_path,
    audio_wav_path,
    output_mp4_path,
    lesson_title,
    slide_title,
    slide_notes,
    cf_images,
    pexels_video_clips=None,
    block_idx=0,
    total_blocks=1,
    block_start_sec=0.0,
    total_video_duration_sec=60.0,
    visual_blueprint="standard_30s",
    visual_template="broadcast_studio",
    chart_data=None,
    presenter_name="",
    channel_name="",
    aspect_ratio="16:9",
    logo_path=None,
    logo_position="top_right",
    fps=24,
    device="cuda",
    run_ffmpeg_fn=None,
    progress_cb=None,
    ass_subtitle_path=None,
    tutorial_url=None,
    tutorial_step=None,
    tutorial_action=None,
    flag_type="auto",
    step_image_path=None,
    step_video_path=None
):
    """
    Rescales the 10-Minute Video Blueprint proportionally to ANY video duration (60s, 90s, 180s, 600s),
    applies the user's selected Studio Visual Template (out of 6 distinct broadcast templates),
    renders optional Data Charts / Scientific Formulas onto slides, overlays branded Intro /
    Lower-Third / Outro graphics + optional Brand Logo, alternates 2-Camera Close-Up Punch-Ins
    with 100% Full-Screen Slide/B-Roll Cutaways (saving 15-25% GPU time), and supports both
    16:9 Widescreen and 9:16 Vertical Shorts/Reels.
    Includes automatic Cloudflare R2 per-block checkpointing so resumed jobs skip already-completed blocks.
    """
    ckpt_r2_key = None
    caller_s3 = None
    caller_bucket = None
    try:
        c_locals = sys._getframe(1).f_locals
        caller_s3 = c_locals.get("s3_client")
        caller_bucket = c_locals.get("bucket_name")
        if caller_s3 and caller_bucket:
            ckpt_suffix = "ass" if (ass_subtitle_path and os.path.exists(ass_subtitle_path)) else "raw"
            ckpt_r2_key = _compute_block_ckpt_r2_key(
                lesson_title=lesson_title,
                slide_title=slide_title,
                block_idx=block_idx,
                total_blocks=total_blocks,
                aspect_ratio=aspect_ratio,
                visual_template=visual_template,
                audio_wav_path=audio_wav_path,
                suffix=ckpt_suffix
            )
            caller_s3.head_object(Bucket=caller_bucket, Key=ckpt_r2_key)
            caller_s3.download_file(caller_bucket, ckpt_r2_key, output_mp4_path)
            if os.path.exists(output_mp4_path) and os.path.getsize(output_mp4_path) > 8192:
                print(f"[R2 CHECKPOINT HIT] Restored Block {block_idx+1}/{total_blocks} from R2 ({ckpt_r2_key}) in <0.5s!", flush=True)
                if callable(progress_cb):
                    try:
                        progress_cb(0.98, f"Restored cached Block {block_idx+1}/{total_blocks} from R2 checkpoint...")
                    except Exception:
                        pass
                return output_mp4_path
    except Exception:
        pass

    theme = resolve_studio_template(visual_template, block_idx=block_idx)
    accent_rgb = theme["accent"]
    pip_slide_pos = theme.get("pip_slide_pos", "top_right")
    pip_broll_pos = theme.get("pip_broll_pos", "bottom_right")
    is_vertical = str(aspect_ratio or "16:9").strip() == "9:16"
    out_w, out_h = (1080, 1920) if is_vertical else (1920, 1080)
    logo_cache = _prepare_brand_logo_overlay(logo_path, max_w=172, max_h=68)

    # Prioritize custom step media (avoiding boredom) for B-roll and PowerPoint slides
    if step_video_path and os.path.exists(step_video_path):
        if not pexels_video_clips:
            pexels_video_clips = [step_video_path]
        elif step_video_path not in pexels_video_clips:
            pexels_video_clips = [step_video_path] + list(pexels_video_clips)

    if step_image_path and os.path.exists(step_image_path):
        if not cf_images:
            cf_images = [step_image_path, step_image_path, step_image_path]
        elif step_image_path not in cf_images:
            cf_images = [step_image_path] + list(cf_images)

    img1 = cf_images[0] if cf_images and len(cf_images) > 0 else None
    img2 = cf_images[1] if cf_images and len(cf_images) > 1 else img1
    img3 = cf_images[2] if cf_images and len(cf_images) > 2 else img2

    vid_clip_1 = pexels_video_clips[0] if pexels_video_clips and len(pexels_video_clips) > 0 else None
    vid_clip_2 = pexels_video_clips[1] if pexels_video_clips and len(pexels_video_clips) > 1 else vid_clip_1

    pexels_stream_1 = _PexelsVideoStream(vid_clip_1, 1920, 1080)
    pexels_stream_2 = _PexelsVideoStream(vid_clip_2, 1920, 1080)

    # Initialize dedicated tutorial step video stream and PIL image (cached across frames)
    tut_step_image_pil = None
    if step_image_path and os.path.exists(step_image_path):
        try:
            tut_step_image_pil = Image.open(step_image_path).convert("RGB")
        except Exception as e_simg:
            print(f"Tutorial step image load note: {e_simg}")

    tut_step_video_stream = None
    if step_video_path and os.path.exists(step_video_path) and os.path.getsize(step_video_path) > 4000:
        try:
            tut_step_video_stream = _PexelsVideoStream(step_video_path, target_w=1280, target_h=720)
        except Exception as e_svid:
            print(f"Tutorial step video stream note: {e_svid}")

    # 1. Pre-compute talking presenter state (S3FD + Wav2Lip-GAN + Syllable Jaw + Eye Blink + Eyebrow Lift + Below-Neck Grids)
    talk_state = synthesize_talking_presenter_frames(raw_avatar_path, audio_wav_path, fps=fps, device=device)
    num_frames = talk_state["num_frames"]
    block_dur_sec = num_frames / float(fps)

    if callable(progress_cb):
        try:
            progress_cb(0.35, f"Wav2Lip-GAN neural lip-sync complete; building {theme['name']} slides...")
        except Exception:
            pass

    # 2. Build Studio Composite Background, Animated PowerPoint Slide Assets, and B-Roll Canvases
    presenter_w, presenter_h = 680, 820
    studio_bg_bgr, pres_slot_x, pres_slot_y = build_studio_Composite_background(
        raw_avatar_path=raw_avatar_path,
        lesson_title=lesson_title,
        slide_title=slide_title,
        slide_notes=slide_notes,
        cf_image_path=img1,
        block_idx=block_idx,
        total_blocks=total_blocks,
        visual_template=visual_template
    )
    rounded_card_mask = _create_rounded_mask(presenter_w, presenter_h, radius=22)
    card_alpha = (rounded_card_mask.astype(np.float32) / 255.0)[..., None]
    base_slot = studio_bg_bgr[pres_slot_y:pres_slot_y + presenter_h, pres_slot_x:pres_slot_x + presenter_w].astype(np.float32)

    def _load_cropped_slide_pil(img_p):
        if img_p and os.path.exists(img_p):
            try:
                ill = Image.open(img_p).convert("RGB")
                iw, ih = ill.size
                tw, th = 760, 704
                sc = max(tw / float(max(1, iw)), th / float(max(1, ih)))
                nw, nh = int(iw * sc), int(ih * sc)
                ill = ill.resize((nw, nh), Image.Resampling.LANCZOS)
                lx = (nw - tw) // 2
                ly = (nh - th) // 2
                return ill.crop((lx, ly, lx + tw, ly + th))
            except Exception:
                return None
        return None

    cf_slide_pil_1 = _load_cropped_slide_pil(img1)
    cf_slide_pil_2 = _load_cropped_slide_pil(img3 or img1)

    slide_anim_styles = ["typewriter", "staggered_cards", "focus_sweep"]
    slide1_anim_style = slide_anim_styles[block_idx % len(slide_anim_styles)]
    has_chart_dict = isinstance(chart_data, dict) and bool(chart_data.get("labels") or chart_data.get("formula"))
    slide2_anim_style = "kinetic_chart" if has_chart_dict else slide_anim_styles[(block_idx + 1) % len(slide_anim_styles)]

    broll_canvas_1 = _prepare_broll_canvas(img2, 1920, 1080)
    broll_canvas_2 = _prepare_broll_canvas(img3, 1920, 1080)
    notes_list = [str(n).strip() for n in (slide_notes or []) if str(n).strip()]
    broll_note_1 = notes_list[0] if len(notes_list) > 0 else slide_title
    broll_note_2 = notes_list[-1] if len(notes_list) > 1 else broll_note_1
    broll_ov1_bgr, broll_ov1_a = _build_broll_overlay_rgba(lesson_title, slide_title, broll_note_1, block_idx, total_blocks, visual_template=visual_template)
    broll_ov2_bgr, broll_ov2_a = _build_broll_overlay_rgba(lesson_title, slide_title, broll_note_2, block_idx, total_blocks, visual_template=visual_template)

    broll_bands = [(18, 122), (714, 902)]
    broll_ov1_strips = [
        (y1, y2, (broll_ov1_bgr[y1:y2] * broll_ov1_a[y1:y2]), (1.0 - broll_ov1_a[y1:y2]))
        for (y1, y2) in broll_bands
    ]
    broll_ov2_strips = [
        (y1, y2, (broll_ov2_bgr[y1:y2] * broll_ov2_a[y1:y2]), (1.0 - broll_ov2_a[y1:y2]))
        for (y1, y2) in broll_bands
    ]
    del broll_ov1_bgr, broll_ov1_a, broll_ov2_bgr, broll_ov2_a

    # 3. Rescale the 10-minute (600s) blueprint to the actual total video duration D
    D = max(block_dur_sec, float(total_video_duration_sec))
    t_intro_window = min(26.0, max(7.0, D * 0.12))
    t_outro_window = min(26.0, max(7.0, D * 0.12))
    is_doc_mode = (visual_blueprint == "documentary_cinema" or visual_template == "documentary_split")

    pattern_id = (block_idx * 5 + total_blocks * 3) % 6
    cut_jitter = ((block_idx * 17) % 9 - 4) * 0.015

    def _get_mode_for_frame(f_idx):
        if flag_type == "tutorial_walkthrough" or (tutorial_url and str(tutorial_url).strip()):
            return "tutorial_walkthrough"

        t_local = f_idx / float(fps)
        t_global = block_start_sec + t_local
        frac_local = t_local / max(1.0, block_dur_sec)

        # Explicit user-selected scene layouts from Step 2 Timeline Editor
        if flag_type == "slides":
            # PowerPoint presentation view:
            # Transition from slide 1 to slide 2 at the midpoint if charts or multi-notes exist
            if has_chart_dict or len(slide_notes or []) > 2:
                return "slide_1" if frac_local < 0.50 else "slide_2"
            return "slide_1"
        elif flag_type == "teacher_lipsync":
            return "studio_lipsync"
        elif flag_type == "b_roll":
            return "broll_1" if frac_local < 0.50 else "broll_2"

        if is_doc_mode:
            if block_idx == 0 and frac_local < 0.34:
                return "studio_lipsync"
            if block_idx == total_blocks - 1 and frac_local > 0.72:
                return "studio_lipsync"
            doc_pat = block_idx % 4
            if doc_pat == 0:
                if frac_local < 0.38 + cut_jitter:
                    return "broll_1"
                elif frac_local < 0.58 + cut_jitter:
                    return "slide_1"
                elif frac_local < 0.86:
                    return "broll_2"
                else:
                    return "studio_lipsync"
            elif doc_pat == 1:
                if frac_local < 0.24 + cut_jitter:
                    return "slide_1"
                elif frac_local < 0.62 + cut_jitter:
                    return "broll_1"
                elif frac_local < 0.82:
                    return "slide_2"
                else:
                    return "broll_2"
            elif doc_pat == 2:
                if frac_local < 0.18:
                    return "studio_lipsync"
                elif frac_local < 0.54 + cut_jitter:
                    return "broll_1"
                elif frac_local < 0.76 + cut_jitter:
                    return "slide_2"
                else:
                    return "broll_2"
            else:
                if frac_local < 0.42 + cut_jitter:
                    return "broll_2"
                elif frac_local < 0.66 + cut_jitter:
                    return "slide_1"
                else:
                    return "broll_1"

        if block_idx == 0 or t_global < t_intro_window:
            if frac_local < 0.52:
                return "studio_lipsync"
            elif frac_local < 0.78:
                return "broll_1"
            else:
                return "slide_1"

        if block_idx == total_blocks - 1 or t_global > (D - t_outro_window):
            if frac_local < 0.24:
                return "slide_2"
            elif frac_local < 0.50:
                return "broll_1"
            else:
                return "studio_lipsync"

        if visual_blueprint == "90s_avatar":
            if frac_local < 0.42 + cut_jitter:
                return "studio_lipsync"
            elif frac_local < 0.70 + cut_jitter:
                return "broll_1" if (block_idx % 2 == 0) else "slide_1"
            elif frac_local < 0.86:
                return "slide_2" if (block_idx % 2 == 0) else "broll_2"
            else:
                return "studio_lipsync"

        c1 = max(0.18, min(0.34, 0.26 + cut_jitter))
        c2 = max(0.44, min(0.62, 0.53 - cut_jitter * 0.5))
        c3 = max(0.72, min(0.86, 0.79 + cut_jitter * 0.5))

        if pattern_id == 0:
            if frac_local < c1: return "broll_1"
            elif frac_local < c2: return "slide_1"
            elif frac_local < c3: return "broll_2"
            else: return "slide_2"
        elif pattern_id == 1:
            if frac_local < c1: return "slide_1"
            elif frac_local < c2: return "broll_1"
            elif frac_local < c3: return "studio_lipsync"
            else: return "slide_2"
        elif pattern_id == 2:
            if frac_local < 0.22: return "studio_lipsync"
            elif frac_local < 0.52: return "broll_1"
            elif frac_local < 0.78: return "slide_2"
            else: return "broll_2"
        elif pattern_id == 3:
            if frac_local < 0.30: return "broll_1"
            elif frac_local < 0.56: return "slide_1"
            elif frac_local < 0.82: return "broll_2"
            else: return "studio_lipsync"
        elif pattern_id == 4:
            if frac_local < c1: return "slide_1"
            elif frac_local < c2: return "broll_2"
            elif frac_local < c3: return "slide_2"
            else: return "broll_1"
        else:
            if frac_local < 0.28: return "broll_1"
            elif frac_local < 0.48: return "studio_lipsync"
            elif frac_local < 0.76: return "broll_2"
            else: return "slide_2"

    modes = [_get_mode_for_frame(i) for i in range(num_frames)]

    # Pre-compute segment start & length for each slide appearance so slide animations
    # (typing effect, staggered card reveal, growing bar charts) progress from 0.0 -> 1.0
    # across the exact duration of that slide scene!
    slide_run_bounds = {}
    run_start = 0
    for idx_m in range(1, num_frames + 1):
        if idx_m == num_frames or modes[idx_m] != modes[run_start]:
            m_val = modes[run_start]
            if m_val in ("slide_1", "slide_2"):
                for k_f in range(run_start, idx_m):
                    slide_run_bounds[k_f] = (run_start, max(1, idx_m - run_start))
            run_start = idx_m

    last_broll_frame = {"broll_1": (-1, None), "broll_2": (-1, None)}
    last_slide_step = {"slide_1": (-1, None), "slide_2": (-1, None)}
    cam_dir = (block_idx % 4)

    def _render_single_mode_frame(mode_name, f_idx):
        prog = (f_idx % max(1, num_frames // 3)) / float(max(1, num_frames // 3))
        W, H = 1920, 1080
        use_closeup_cam = ((f_idx // max(1, int(fps * 5.5))) % 2 == 1)

        if mode_name == "tutorial_walkthrough":
            pres_bgr = get_animated_presenter_frame(talk_state, f_idx, camera_closeup=False)
            step_vid_frame = tut_step_video_stream.read_frame() if tut_step_video_stream else None
            wide_out = render_tutorial_walkthrough_screen(
                lesson_title=lesson_title,
                slide_title=slide_title,
                slide_notes=slide_notes,
                tutorial_url=tutorial_url,
                tutorial_step=tutorial_step if tutorial_step is not None else (block_idx + 1),
                tutorial_action=tutorial_action,
                f_idx=f_idx,
                num_frames=num_frames,
                fps=fps,
                pres_bgr=pres_bgr,
                visual_template=visual_template,
                theme=theme,
                aspect_ratio=aspect_ratio,
                presenter_name=presenter_name,
                channel_name=channel_name,
                step_image_pil=tut_step_image_pil,
                step_video_frame=step_vid_frame,
                step_image_path=step_image_path
            )

        elif mode_name == "studio_lipsync":
            pres_bgr = get_animated_presenter_frame(talk_state, f_idx, camera_closeup=use_closeup_cam)
            full_frame = studio_bg_bgr.copy()
            comp_slot = (pres_bgr.astype(np.float32) * card_alpha + base_slot * (1.0 - card_alpha)).astype(np.uint8)
            full_frame[pres_slot_y:pres_slot_y + presenter_h, pres_slot_x:pres_slot_x + presenter_w] = comp_slot
            wide_out = full_frame

        elif mode_name in ("slide_1", "slide_2"):
            s_start, s_len = slide_run_bounds.get(f_idx, (0, max(1, num_frames // 3)))
            scene_prog = max(0.0, min(1.0, (f_idx - s_start) / float(max(1, s_len - 1))))
            # Quantize to 16 smooth animation steps per slide scene so typing & chart growth
            # are fluid while caching rendered PIL slides across neighboring frames
            anim_q = int(round(scene_prog * 16.0))
            s_cached_step, s_cached_img = last_slide_step.get(mode_name, (-1, None))
            if s_cached_step == anim_q and s_cached_img is not None:
                base = s_cached_img.copy()
            else:
                is_s2 = (mode_name == "slide_2")
                base, slide_img_box = _render_animated_powerpoint_slide_step(
                    lesson_title=lesson_title,
                    slide_title=slide_title,
                    slide_notes=slide_notes,
                    cf_pil_img=cf_slide_pil_2 if is_s2 else cf_slide_pil_1,
                    block_idx=block_idx,
                    total_blocks=total_blocks,
                    slide_num=2 if is_s2 else 1,
                    anim_progress=anim_q / 16.0,
                    anim_style=slide2_anim_style if is_s2 else slide1_anim_style,
                    visual_template=visual_template,
                    chart_data=chart_data if is_s2 else None
                )
                if not (is_s2 and has_chart_dict):
                    ix1, iy1, ix2, iy2 = slide_img_box
                    iw, ih = ix2 - ix1, iy2 - iy1
                    zoom = (1.06 - 0.05 * scene_prog) if (cam_dir % 2 == 1 and is_s2) else (1.0 + 0.06 * scene_prog)
                    pan_x = (1.0 - zoom) * (iw * (0.46 + 0.08 * scene_prog if cam_dir < 2 else 0.54 - 0.08 * scene_prog))
                    pan_y = (1.0 - zoom) * (ih * 0.5)
                    sub_img = base[iy1:iy2, ix1:ix2]
                    M = np.float32([
                        [zoom, 0.0, pan_x],
                        [0.0, zoom, pan_y]
                    ])
                    base[iy1:iy2, ix1:ix2] = cv2.warpAffine(sub_img, M, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                last_slide_step[mode_name] = (anim_q, base.copy())
            wide_out = base

        else:
            is_second = (mode_name == "broll_2")
            ov_strips = broll_ov2_strips if is_second else broll_ov1_strips
            p_stream = pexels_stream_2 if is_second else pexels_stream_1

            has_vid_stream = p_stream.is_valid()
            cache_token = f_idx if has_vid_stream else (f_idx // 2)
            cached_idx, cached_img = last_broll_frame.get(mode_name, (-1, None))
            if cached_idx == cache_token and cached_img is not None:
                comp = cached_img.copy()
            else:
                pexels_bgr = p_stream.read_frame() if has_vid_stream else None
                if pexels_bgr is not None:
                    moved = pexels_bgr
                else:
                    canvas = broll_canvas_2 if is_second else broll_canvas_1
                    traj = (block_idx + (1 if is_second else 0)) % 4
                    if traj == 0:
                        zoom = 1.02 + 0.10 * prog
                        tx, ty = -20.0 * prog, -12.0 * prog
                    elif traj == 1:
                        zoom = 1.12 - 0.09 * prog
                        tx, ty = 18.0 * (1.0 - prog), 10.0 * (1.0 - prog)
                    elif traj == 2:
                        zoom = 1.04 + 0.08 * prog
                        tx, ty = 22.0 * (prog - 0.5), -14.0 * prog
                    else:
                        zoom = 1.10 - 0.07 * prog
                        tx, ty = -18.0 * (1.0 - prog), 12.0 * (prog - 0.5)
                    M = np.float32([
                        [zoom, 0.0, (1.0 - zoom) * (W * 0.5) + tx],
                        [0.0, zoom, (1.0 - zoom) * (H * 0.5) + ty]
                    ])
                    moved = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

                comp = moved.copy()
                for (sy1, sy2, premul_bgr, inv_a) in ov_strips:
                    comp[sy1:sy2] = (comp[sy1:sy2].astype(np.float32) * inv_a + premul_bgr).astype(np.uint8)
                last_broll_frame[mode_name] = (cache_token, comp.copy())

            wide_out = comp

        if is_vertical:
            if mode_name == "tutorial_walkthrough":
                return wide_out
            pres_bgr = get_animated_presenter_frame(talk_state, f_idx, camera_closeup=use_closeup_cam)
            return _compose_vertical_916_frame(wide_out, pres_bgr, lesson_title, slide_title, slide_notes, theme, block_idx, total_blocks)
        return wide_out

    silent_mp4 = f"{output_mp4_path}.silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_mp4, fourcc, float(fps), (out_w, out_h))

    transition_len = 7
    tick_33 = max(1, num_frames // 3)
    tick_66 = max(2, (num_frames * 2) // 3)
    avatar_rect_coords = (pres_slot_x, pres_slot_y, pres_slot_x + presenter_w, pres_slot_y + presenter_h)

    for f_idx in range(num_frames):
        curr_mode = modes[f_idx]
        frame_bgr = _render_single_mode_frame(curr_mode, f_idx)

        if f_idx >= transition_len:
            prev_mode = modes[f_idx - transition_len]
            if prev_mode != curr_mode:
                steps_since = 0
                for back in range(1, transition_len + 1):
                    if modes[f_idx - back] != curr_mode:
                        steps_since = back
                        break
                if 0 < steps_since <= transition_len:
                    alpha = steps_since / float(transition_len)
                    prev_bgr = _render_single_mode_frame(prev_mode, f_idx)
                    frame_bgr = cv2.addWeighted(frame_bgr, alpha, prev_bgr, 1.0 - alpha, 0.0)

        if block_idx > 0 and f_idx < 7:
            fade_a = (f_idx + 1) / 7.0
            dark_bg = np.full_like(frame_bgr, (15, 20, 34), dtype=np.uint8)
            frame_bgr = cv2.addWeighted(frame_bgr, fade_a, dark_bg, 1.0 - fade_a, 0.0)

        t_local = f_idx / float(fps)
        t_global = block_start_sec + t_local
        if curr_mode != "tutorial_walkthrough":
            frame_bgr = _draw_broadcast_branding_overlays(
                frame_bgr=frame_bgr,
                t_local=t_local,
                t_global=t_global,
                total_video_dur=D,
                block_idx=block_idx,
                total_blocks=total_blocks,
                lesson_title=lesson_title,
                slide_title=slide_title,
                presenter_name=presenter_name,
                channel_name=channel_name,
                theme=theme,
                curr_mode=curr_mode,
                avatar_rect=avatar_rect_coords
            )

        if logo_cache is not None:
            frame_bgr = _apply_brand_logo_to_frame_bgr(frame_bgr, logo_cache, pos=logo_position)

        writer.write(frame_bgr)

        if callable(progress_cb):
            if f_idx == tick_33:
                try:
                    progress_cb(0.58, "Compositing 1080p multi-scene video frames (35% of block)...")
                except Exception:
                    pass
            elif f_idx == tick_66:
                try:
                    progress_cb(0.78, "Compositing 1080p multi-scene video frames (70% of block)...")
                except Exception:
                    pass

    writer.release()
    pexels_stream_1.close()
    pexels_stream_2.close()
    if tut_step_video_stream:
        tut_step_video_stream.close()

    import gc
    del talk_state, cf_slide_pil_1, cf_slide_pil_2, broll_canvas_1, broll_canvas_2, studio_bg_bgr, last_broll_frame, last_slide_step
    gc.collect()

    if callable(progress_cb):
        try:
            progress_cb(0.92, "Encoding H.264 broadcast video stream & burning captions...")
        except Exception:
            pass

    mux_cmd = [
        "ffmpeg", "-y",
        "-i", silent_mp4,
        "-i", audio_wav_path,
    ]
    if ass_subtitle_path and os.path.exists(ass_subtitle_path):
        ass_escaped = ass_subtitle_path.replace("\\", "/").replace(":", "\\:")
        mux_cmd += ["-vf", f"ass='{ass_escaped}'"]
    mux_cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
        "-shortest",
        output_mp4_path
    ]
    if run_ffmpeg_fn:
        run_ffmpeg_fn(mux_cmd, desc="Mux dynamic multi-scene block with audio")
    else:
        import subprocess
        subprocess.run(mux_cmd, check=True)

    if os.path.exists(silent_mp4):
        try:
            os.remove(silent_mp4)
        except Exception:
            pass

    if caller_s3 and caller_bucket and ckpt_r2_key and os.path.exists(output_mp4_path):
        ckpt_copy_path = f"{output_mp4_path}.r2ckpt"
        try:
            import shutil
            shutil.copyfile(output_mp4_path, ckpt_copy_path)
            def _upload_block_ckpt(s3_c, b_name, r2_k, local_f):
                try:
                    s3_c.upload_file(local_f, b_name, r2_k, ExtraArgs={"ContentType": "video/mp4"})
                    print(f"[R2 CHECKPOINT SAVED] Uploaded {r2_k}", flush=True)
                except Exception as up_e:
                    print(f"R2 block checkpoint save note: {up_e}")
                finally:
                    if os.path.exists(local_f):
                        try:
                            os.remove(local_f)
                        except Exception:
                            pass
            threading.Thread(
                target=_upload_block_ckpt,
                args=(caller_s3, caller_bucket, ckpt_r2_key, ckpt_copy_path),
                daemon=True
            ).start()
        except Exception:
            pass

    return output_mp4_path


_ACTIVE_BLOCK_CTX = {"block_cached": False}


def _install_runtime_accelerators():
    """
    When hot-synced into an already-running RunPod container (via _hot_sync_lipsync_engine_from_github),
    wraps handler functions on __main__ so existing containers immediately benefit from:
      1. R2 TTS audio block checkpointing (Pass 1B)
      2. Instant skip of Pexels / Cloudflare AI network fetches when a block's video checkpoint is on R2
      3. Fast single-clip AI subject motion synthesis on multi-block lessons
      4. Ultrafast H.264 ASS subtitle burn-in
    """
    for mod_name in ("__main__", "handler"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue

        # 0. Wrap report_live_progress to detect on the main thread when a block is starting Pass 2
        orig_prog = getattr(mod, "report_live_progress", None)
        if callable(orig_prog) and not getattr(orig_prog, "_ailg_ctx_wrapped", False):
            def _ctx_prog_wrapper(job, s3_client, bucket_name, job_id, step, percent, time_left_sec=15, extra=None, _orig=orig_prog):
                try:
                    step_s = str(step or "")
                    if "Fetching HD video clips" in step_s and s3_client and bucket_name:
                        c_locals = sys._getframe(1).f_locals
                        item = c_locals.get("item") or {}
                        r_key = _compute_block_ckpt_r2_key(
                            lesson_title=c_locals.get("lesson_title", ""),
                            slide_title=item.get("slide_title") or c_locals.get("slide_title", ""),
                            block_idx=item.get("idx_counter", c_locals.get("idx_counter", 0)),
                            total_blocks=c_locals.get("total_blocks", 1),
                            aspect_ratio=c_locals.get("aspect_ratio", "16:9"),
                            visual_template=c_locals.get("visual_template", "broadcast_studio"),
                            audio_wav_path=item.get("audio_wav") or c_locals.get("output_wav", ""),
                            suffix="raw"
                        )
                        try:
                            s3_client.head_object(Bucket=bucket_name, Key=r_key)
                            _ACTIVE_BLOCK_CTX["block_cached"] = True
                        except Exception:
                            _ACTIVE_BLOCK_CTX["block_cached"] = False
                    elif "Mixing Teacher Voice" in step_s:
                        _ACTIVE_BLOCK_CTX["block_cached"] = False
                except Exception:
                    _ACTIVE_BLOCK_CTX["block_cached"] = False
                return _orig(job, s3_client, bucket_name, job_id, step, percent, time_left_sec=time_left_sec, extra=extra)

            _ctx_prog_wrapper._ailg_ctx_wrapped = True
            setattr(mod, "report_live_progress", _ctx_prog_wrapper)

        orig_px = getattr(mod, "fetch_pexels_video_clips_for_block", None)
        if callable(orig_px) and not getattr(orig_px, "_ailg_fast_wrapped", False):
            def _fast_px_wrapper(*args, _orig=orig_px, **kwargs):
                if _ACTIVE_BLOCK_CTX.get("block_cached"):
                    return []
                return _orig(*args, **kwargs)
            _fast_px_wrapper._ailg_fast_wrapped = True
            setattr(mod, "fetch_pexels_video_clips_for_block", _fast_px_wrapper)

        orig_cf = getattr(mod, "generate_cloudflare_ai_image", None)
        if callable(orig_cf) and not getattr(orig_cf, "_ailg_fast_wrapped", False):
            def _fast_cf_wrapper(*args, _orig=orig_cf, **kwargs):
                if _ACTIVE_BLOCK_CTX.get("block_cached"):
                    return None
                return _orig(*args, **kwargs)
            _fast_cf_wrapper._ailg_fast_wrapped = True
            setattr(mod, "generate_cloudflare_ai_image", _fast_cf_wrapper)

        # 1. Wrap synthesize_block_audio_with_pauses with Cloudflare R2 TTS audio caching
        orig_tts = getattr(mod, "synthesize_block_audio_with_pauses", None)
        if callable(orig_tts) and not getattr(orig_tts, "_ailg_r2_wrapped", False):
            def _fast_tts_wrapper(*args, _orig=orig_tts, **kwargs):
                output_wav = kwargs.get("output_wav") or (args[1] if len(args) > 1 else None)
                text = kwargs.get("text") or (args[0] if len(args) > 0 else "")
                tts_engine = kwargs.get("tts_engine", "omnivoice")
                target_lang = kwargs.get("target_lang", "en")
                edge_voice = kwargs.get("edge_voice", "")
                voice_style = kwargs.get("voice_style", "normal")
                speed_val = kwargs.get("speed_val", 1.0)
                pitch_val = kwargs.get("pitch_val", 0.0)
                whisper_mode = kwargs.get("whisper_mode", False)

                s3_c = None
                b_name = None
                tts_r2_key = None
                try:
                    c_locals = sys._getframe(1).f_locals
                    s3_c = c_locals.get("s3_client")
                    b_name = c_locals.get("bucket_name")
                    if s3_c and b_name and output_wav and text:
                        sig = f"v762_tts|{tts_engine}|{target_lang}|{edge_voice}|{voice_style}|{speed_val:.2f}|{pitch_val:.2f}|{whisper_mode}|{text}"
                        h = hashlib.md5(sig.encode("utf-8", errors="ignore")).hexdigest()[:18]
                        tts_r2_key = f"checkpoints/tts/{h}.wav"
                        s3_c.head_object(Bucket=b_name, Key=tts_r2_key)
                        s3_c.download_file(b_name, tts_r2_key, output_wav)
                        if os.path.exists(output_wav) and os.path.getsize(output_wav) > 1024:
                            print(f"[R2 TTS CHECKPOINT HIT] Restored narration audio ({tts_r2_key}) in <0.3s!", flush=True)
                            return output_wav
                except Exception:
                    pass

                res = _orig(*args, **kwargs)
                if s3_c and b_name and tts_r2_key and output_wav and os.path.exists(output_wav):
                    wav_copy = f"{output_wav}.r2ckpt"
                    try:
                        import shutil
                        shutil.copyfile(output_wav, wav_copy)
                        def _up_wav(c, b, k, f_path):
                            try:
                                c.upload_file(f_path, b, k, ExtraArgs={"ContentType": "audio/wav"})
                            except Exception:
                                pass
                            finally:
                                if os.path.exists(f_path):
                                    try:
                                        os.remove(f_path)
                                    except Exception:
                                        pass
                        threading.Thread(target=_up_wav, args=(s3_c, b_name, tts_r2_key, wav_copy), daemon=True).start()
                    except Exception:
                        pass
                return res

            _fast_tts_wrapper._ailg_r2_wrapped = True
            setattr(mod, "synthesize_block_audio_with_pauses", _fast_tts_wrapper)

        # 2. Replace synthesize_ai_subject_video_clip with clean, particle-free HD Ken-Burns motion
        orig_ai_clip = getattr(mod, "synthesize_ai_subject_video_clip", None)
        if callable(orig_ai_clip) and not getattr(orig_ai_clip, "_ailg_fast_wrapped", False):
            def _fast_ai_clip_wrapper(cf_images, output_mp4, subject_text="", duration_sec=8.0, fps=24, reverse_motion=False, block_seed=0, _orig=orig_ai_clip):
                if _ACTIVE_BLOCK_CTX.get("block_cached"):
                    return None
                try:
                    c_locals = sys._getframe(1).f_locals
                    total_blocks = int(c_locals.get("total_blocks", 1) or 1)
                    if reverse_motion and total_blocks >= 3:
                        existing_c1 = c_locals.get("ai_clip_1")
                        if existing_c1 and os.path.exists(existing_c1):
                            return existing_c1
                except Exception:
                    pass

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
                dur_clean = min(float(duration_sec or 5.0), 5.0)
                total_frames = max(48, int(round(dur_clean * fps)))
                n_scenes = len(valid_imgs)

                temp_raw_mp4 = output_mp4 + ".raw.mp4"
                writer = cv2.VideoWriter(temp_raw_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (target_w, target_h))
                if not writer.isOpened():
                    return None

                def _render_clean_cam(base_img, scene_idx, local_u):
                    ease_u = 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, local_u)))
                    mode = (scene_idx + int(block_seed) + (2 if reverse_motion else 0)) % 6
                    if mode == 0:
                        zoom, cx, cy = 1.03 + 0.11 * ease_u, 0.48 + 0.04 * ease_u, 0.47 - 0.03 * ease_u
                    elif mode == 1:
                        zoom, cx, cy = 1.15 - 0.10 * ease_u, 0.52 - 0.05 * ease_u, 0.48 + 0.03 * ease_u
                    elif mode == 2:
                        zoom, cx, cy = 1.10, 0.44 + 0.12 * ease_u, 0.49 + 0.015 * math.sin(ease_u * math.pi)
                    elif mode == 3:
                        zoom, cx, cy = 1.06 + 0.06 * ease_u, 0.50 + 0.02 * math.cos(ease_u * math.pi), 0.54 - 0.09 * ease_u
                    elif mode == 4:
                        zoom, cx, cy = 1.04 + 0.09 * ease_u, 0.46 + 0.08 * ease_u, 0.45 + 0.07 * ease_u
                    else:
                        zoom, cx, cy = 1.06 + 0.07 * math.sin(ease_u * math.pi), 0.50 + 0.05 * math.sin(ease_u * math.pi), 0.50 - 0.04 * math.cos(ease_u * math.pi)
                    crop_w = int(round(target_w / zoom))
                    crop_h = int(round(target_h / zoom))
                    x0 = max(0, min(target_w - crop_w, int(round(cx * target_w - crop_w * 0.5))))
                    y0 = max(0, min(target_h - crop_h, int(round(cy * target_h - crop_h * 0.5))))
                    cropped = base_img[y0:y0 + crop_h, x0:x0 + crop_w]
                    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

                for f_idx in range(total_frames):
                    pos = (f_idx / float(max(1, total_frames - 1))) * n_scenes
                    s_idx = min(n_scenes - 1, int(pos))
                    local_u = pos - s_idx
                    frame = _render_clean_cam(valid_imgs[s_idx], s_idx, local_u)
                    if s_idx < n_scenes - 1 and local_u > 0.78:
                        alpha = (local_u - 0.78) / 0.22
                        next_frame = _render_clean_cam(valid_imgs[s_idx + 1], s_idx + 1, 0.0)
                        frame = cv2.addWeighted(frame, 1.0 - alpha, next_frame, alpha, 0)
                    writer.write(frame)
                writer.release()

                run_ff = getattr(mod, "run_ffmpeg", None)
                try:
                    if callable(run_ff):
                        run_ff(
                            f"ffmpeg -y -i {temp_raw_mp4} -c:v libx264 -preset veryfast -crf 19 -pix_fmt yuv420p {output_mp4}",
                            desc="Encode clean HD AI Subject Motion Video Clip"
                        )
                    if os.path.exists(temp_raw_mp4):
                        os.remove(temp_raw_mp4)
                except Exception:
                    if os.path.exists(temp_raw_mp4):
                        import shutil
                        shutil.move(temp_raw_mp4, output_mp4)
                return output_mp4 if (os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 5000) else None

            _fast_ai_clip_wrapper._ailg_fast_wrapped = True
            setattr(mod, "synthesize_ai_subject_video_clip", _fast_ai_clip_wrapper)

        # 3. Wrap create_synced_lyrics_ass to keep subtitles compact in bottom safe strip (Fontsize=44, MarginV=24)
        orig_ass = getattr(mod, "create_synced_lyrics_ass", None)
        if callable(orig_ass) and not getattr(orig_ass, "_ailg_safe_wrapped", False):
            def _safe_ass_wrapper(*args, _orig=orig_ass, **kwargs):
                res_tuple = _orig(*args, **kwargs)
                try:
                    sub_path = res_tuple[0] if isinstance(res_tuple, tuple) else res_tuple
                    if sub_path and os.path.exists(sub_path):
                        with open(sub_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        content = content.replace("Liberation Sans,54,", "Liberation Sans,44,")
                        content = content.replace(",72,72,44,1", ",120,120,24,1")
                        with open(sub_path, "w", encoding="utf-8") as f:
                            f.write(content)
                except Exception:
                    pass
                return res_tuple

            _safe_ass_wrapper._ailg_safe_wrapped = True
            setattr(mod, "create_synced_lyrics_ass", _safe_ass_wrapper)

        # 4. Wrap build_youtube_seo_package to generate comprehensive SEO-friendly YouTube Description, Chapters & Tags
        orig_seo = getattr(mod, "build_youtube_seo_package", None)
        if callable(orig_seo) and not getattr(orig_seo, "_ailg_seo_wrapped", False):
            def _rich_seo_wrapper(lesson_title, synthesized_blocks, presenter_name="", channel_name="", _orig=orig_seo):
                import re
                import json
                fmt_ts = getattr(mod, "_format_youtube_chapter_ts", None)
                chapters = []
                keywords_bag = []
                module_summaries = []
                summary_bullets = []
                for idx, item in enumerate(synthesized_blocks or []):
                    st_sec = float(item.get("start_sec", 0.0) or 0.0)
                    if idx == 0:
                        ts_str = "00:00"
                    elif callable(fmt_ts):
                        ts_str = fmt_ts(st_sec)
                    else:
                        m_i, s_i = int(st_sec // 60), int(st_sec % 60)
                        ts_str = f"{m_i:02d}:{s_i:02d}"
                    stitle = str(item.get("slide_title") or f"Section {idx + 1}").strip()
                    chapters.append(f"{ts_str} - {stitle}")
                    notes = [str(n).strip() for n in (item.get("notes") or []) if str(n).strip()]
                    if notes:
                        summary_bullets.append(f"✅ {stitle}: {notes[0]}")
                        module_summaries.append(f"• Module {idx + 1} ({stitle}): " + "; ".join(notes[:3]))
                    for w in re.findall(r'[A-Za-z]{4,}', stitle + " " + " ".join(notes[:2])):
                        wl = w.lower()
                        if wl not in keywords_bag and wl not in ("section", "module", "slide", "lesson", "introduction", "conclusion", "with", "from", "that", "this", "your", "have", "will", "what", "when", "where", "into", "about"):
                            keywords_bag.append(wl)

                chapters_text = "\n".join(chapters) if chapters else "00:00 - Introduction & Overview"
                clean_title = str(lesson_title or "Complete Educational Masterclass").strip()
                seo_title = clean_title if len(clean_title) >= 42 else f"{clean_title} | Complete Step-by-Step Video Lesson & Guide"
                seo_title = seo_title[:98]

                ch_line = f"Presented by {presenter_name}" if presenter_name else (f"Published by {channel_name}" if channel_name else "Remedial Corner Educational Studio")
                top_kws = [k.title() for k in keywords_bag[:6]]
                kw_phrase = ", ".join(top_kws) if top_kws else clean_title

                hashtags = ["#" + re.sub(r'[^A-Za-z0-9]', '', w.title()) for w in ([clean_title] + top_kws[:5]) if re.sub(r'[^A-Za-z0-9]', '', w)]
                hashtags = list(dict.fromkeys(hashtags + ["#Education", "#OnlineLearning", "#VideoLesson", "#Masterclass"]))[:8]
                hashtags_line = " ".join(hashtags)

                tags_list = [
                    clean_title,
                    f"{clean_title} Explained",
                    f"{clean_title} Tutorial",
                    f"{clean_title} Guide",
                    "Video Lesson",
                    "Online Course",
                    "Educational Masterclass",
                    "Step by Step Guide",
                ]
                if channel_name:
                    tags_list.append(channel_name)
                if presenter_name:
                    tags_list.append(presenter_name)
                for kw in keywords_bag[:14]:
                    if kw.title() not in tags_list:
                        tags_list.append(kw.title())
                tags_list = tags_list[:20]
                tags_csv = ", ".join(tags_list)

                full_description = (
                    f"🎓 {seo_title}\n"
                    f"In this comprehensive, step-by-step video lesson on {clean_title}, we break down the essential principles, real-world applications, and key exam & practical takeaways covering {kw_phrase}. ({ch_line})\n\n"
                    f"📖 WHAT YOU WILL LEARN IN THIS VIDEO:\n"
                    + ("\n".join(summary_bullets[:8]) if summary_bullets else f"✅ Core concepts, practical examples, and complete breakdown of {clean_title}")
                    + "\n\n⏱️ YOUTUBE CHAPTER TIMESTAMPS (Click to Jump to Any Topic):\n"
                    + chapters_text
                    + "\n\n📚 DETAILED LESSON MODULE BREAKDOWN:\n"
                    + ("\n".join(module_summaries[:8]) if module_summaries else f"• Complete walkthrough of {clean_title}")
                    + f"\n\n💬 JOIN THE DISCUSSION:\n"
                    f"Which concept from today's lesson on {clean_title} stood out to you the most, or what topic would you like us to cover next? Drop your thoughts and questions in the comments below—we read and reply to every comment!\n\n"
                    f"🔔 SUPPORT THE CHANNEL:\n"
                    f"If this lesson helped clarify {clean_title}, please hit the 👍 LIKE button, ↗ SHARE this video with a classmate or colleague, and 🔔 SUBSCRIBE (with notifications turned on) so you never miss our upcoming educational masterclasses!\n\n"
                    f"🔖 TAGS & KEYWORDS:\n{tags_csv}\n\n"
                    f"{hashtags_line}\n"
                )

                return {
                    "title": seo_title,
                    "primary_title": seo_title,
                    "chapters_text": chapters_text,
                    "full_description": full_description,
                    "description": full_description,
                    "tags": tags_csv,
                    "tags_list": tags_list,
                    "hashtags": hashtags,
                }

            _rich_seo_wrapper._ailg_seo_wrapped = True
            setattr(mod, "build_youtube_seo_package", _rich_seo_wrapper)

        # 5. Wrap burn_presentation_slides to use -preset veryfast -crf 19 (crisp HD broadcast quality)
        orig_burn = getattr(mod, "burn_presentation_slides", None)
        if callable(orig_burn) and not getattr(orig_burn, "_ailg_fast_wrapped", False):
            def _fast_burn_wrapper(video_path, ass_path, output_path, duration_sec=10, aspect_ratio="16:9", _orig=orig_burn):
                ass_escaped = ass_path.replace("\\", "/").replace(":", "\\:")
                cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", f"ass='{ass_escaped}'",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
                    "-c:a", "copy", output_path
                ]
                run_ff = getattr(mod, "run_ffmpeg", None)
                if callable(run_ff):
                    run_ff(cmd, desc="Crisp HD burn captions")
                    return output_path
                return _orig(video_path, ass_path, output_path, duration_sec=duration_sec, aspect_ratio=aspect_ratio)

            _fast_burn_wrapper._ailg_fast_wrapped = True
            setattr(mod, "burn_presentation_slides", _fast_burn_wrapper)


try:
    _install_runtime_accelerators()
except Exception:
    pass


# Keep backward-compatible wrappers
def render_lipsync_teacher_block(raw_avatar_path, audio_wav_path, output_mp4_path, lesson_title, slide_title, slide_notes, cf_image_path=None, cf_image_2=None, block_idx=0, total_blocks=1, fps=24, device="cuda", run_ffmpeg_fn=None, include_mid_cutaway=False):
    return render_dynamic_lesson_block(
        raw_avatar_path=raw_avatar_path,
        audio_wav_path=audio_wav_path,
        output_mp4_path=output_mp4_path,
        lesson_title=lesson_title,
        slide_title=slide_title,
        slide_notes=slide_notes,
        cf_images=[cf_image_path, cf_image_2 or cf_image_path],
        block_idx=block_idx,
        total_blocks=total_blocks,
        fps=fps,
        device=device,
        run_ffmpeg_fn=run_ffmpeg_fn
    )


def render_slides_block(lesson_title, slide_title, slide_notes, cf_image_1, cf_image_2, raw_avatar_path, audio_wav_path, output_mp4_path, block_idx=0, total_blocks=1, fps=24, run_ffmpeg_fn=None):
    return render_dynamic_lesson_block(
        raw_avatar_path=raw_avatar_path,
        audio_wav_path=audio_wav_path,
        output_mp4_path=output_mp4_path,
        lesson_title=lesson_title,
        slide_title=slide_title,
        slide_notes=slide_notes,
        cf_images=[cf_image_1, cf_image_2 or cf_image_1],
        block_idx=block_idx,
        total_blocks=total_blocks,
        fps=fps,
        run_ffmpeg_fn=run_ffmpeg_fn
    )


def render_broll_motion_block(lesson_title, slide_title, slide_notes, cf_image_1, cf_image_2, raw_avatar_path, audio_wav_path, output_mp4_path, block_idx=0, total_blocks=1, fps=24, run_ffmpeg_fn=None):
    return render_dynamic_lesson_block(
        raw_avatar_path=raw_avatar_path,
        audio_wav_path=audio_wav_path,
        output_mp4_path=output_mp4_path,
        lesson_title=lesson_title,
        slide_title=slide_title,
        slide_notes=slide_notes,
        cf_images=[cf_image_1, cf_image_2 or cf_image_1],
        block_idx=block_idx,
        total_blocks=total_blocks,
        fps=fps,
        run_ffmpeg_fn=run_ffmpeg_fn
    )

