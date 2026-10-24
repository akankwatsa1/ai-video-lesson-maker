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


def prepare_presenter_portrait(raw_avatar_path, target_w=680, target_h=820, device="cuda"):
    """
    Takes any phone photo (vertical 9:16, square, or horizontal), detects the teacher's face
    using S3FD, and smartly crops/scales it into a clean presenter portrait (680x820).
    """
    if not raw_avatar_path or not os.path.exists(raw_avatar_path):
        canvas = np.full((target_h, target_w, 3), (45, 55, 75), dtype=np.uint8)
        return canvas, (target_w // 4, target_h // 4, 3 * target_w // 4, 3 * target_h // 4), False

    img = cv2.imread(raw_avatar_path)
    if img is None:
        pil_img = Image.open(raw_avatar_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    (fx1, fy1, fx2, fy2), _ = detect_face_bbox(img, device=device)
    face_cx = (fx1 + fx2) // 2
    face_cy = (fy1 + fy2) // 2

    target_aspect = target_w / float(target_h)
    img_aspect = w / float(h)

    if img_aspect > target_aspect:
        crop_h = h
        crop_w = int(h * target_aspect)
        x0 = max(0, min(w - crop_w, face_cx - crop_w // 2))
        y0 = 0
    else:
        crop_w = w
        crop_h = int(w / target_aspect)
        x0 = 0
        desired_y0 = face_cy - int(crop_h * 0.36)
        y0 = max(0, min(h - crop_h, desired_y0))

    cropped = img[y0:y0 + crop_h, x0:x0 + crop_w]
    presenter_bgr = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    bbox, found_final = detect_face_bbox(presenter_bgr, device=device)
    return presenter_bgr, bbox, found_final


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


def _apply_syllable_mouth_and_blink(face_bgr, energy, f_idx, fps=24):
    """
    Reinforces visible mouth/jaw articulation on spoken syllables, adds
    speech-synced eyebrow micro-lift, and performs natural single & occasional
    double eye blinks so even tight-lipped portrait photos look genuinely alive.
    """
    fh, fw = face_bgr.shape[:2]
    out = face_bgr.copy()
    t_sec = f_idx / float(max(1, fps))

    # 1. Eyebrow Micro-Lift on Emphasized Syllables (y: 16% -> 27%, x: 16% -> 84%)
    if energy > 0.14 and fh > 48 and fw > 48:
        brow_lift_px = min(4.2, float(energy) * 3.6 + 0.8 * max(0.0, math.sin(t_sec * 6.2)))
        by1, by2 = max(2, int(fh * 0.16)), min(fh - 2, int(fh * 0.27))
        bx1, bx2 = max(2, int(fw * 0.16)), min(fw - 2, int(fw * 0.84))
        if by2 > by1 + 6 and bx2 > bx1 + 12:
            brow_roi = out[by1:by2, bx1:bx2].copy()
            bh_r, bw_r = brow_roi.shape[:2]
            grid_x, grid_y = np.meshgrid(np.arange(bw_r, dtype=np.float32), np.arange(bh_r, dtype=np.float32))
            ny = grid_y / float(max(1, bh_r))
            nx = (grid_x - bw_r * 0.5) / float(max(1, bw_r * 0.5))
            weight = np.sin(ny * math.pi) * np.clip(1.0 - (nx ** 2), 0.0, 1.0)
            map_y = np.clip(grid_y + weight * brow_lift_px, 0, bh_r - 1).astype(np.float32)
            out[by1:by2, bx1:bx2] = cv2.remap(brow_roi, grid_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # 2. Syllable-driven lower-jaw & lip opening reinforcement
    if energy > 0.06 and fh > 40 and fw > 40:
        mouth_cx = fw // 2
        mouth_cy = int(fh * 0.74)
        rx = max(8, int(fw * 0.17 * (0.85 + 0.25 * energy)))
        ry = max(2, int(fh * 0.055 * min(1.15, energy * 1.35)))

        # Subtle downward shift of chin/lower lip below mouth_cy
        jaw_top = mouth_cy
        jaw_bot = int(fh * 0.96)
        jaw_h = jaw_bot - jaw_top
        if jaw_h > 8:
            shift_px = max(1, int(fh * 0.045 * energy))
            jaw_patch = out[jaw_top:jaw_bot, :].copy()
            stretched = cv2.resize(jaw_patch, (fw, jaw_h + shift_px), interpolation=cv2.INTER_LINEAR)
            x_grid = np.linspace(-1.0, 1.0, fw, dtype=np.float32)
            horiz_w = np.clip(1.0 - (x_grid ** 2) * 1.4, 0.0, 1.0)[None, :, None]
            out[jaw_top:jaw_bot, :] = (
                stretched[:jaw_h, :].astype(np.float32) * horiz_w
                + jaw_patch.astype(np.float32) * (1.0 - horiz_w)
            ).astype(np.uint8)

        # Subtle inner-mouth shading when syllable energy is strong
        if energy > 0.18:
            lip_mask = np.zeros((fh, fw), dtype=np.float32)
            cv2.ellipse(lip_mask, (mouth_cx, mouth_cy + 1), (rx, ry), 0, 0, 360, 1.0, -1)
            bk = max(3, (min(fw, fh) // 15) | 1)
            lip_mask = cv2.GaussianBlur(lip_mask, (bk, bk), 0)[..., None]
            cavity = np.full_like(out, (28, 22, 48), dtype=np.uint8)
            teeth_y1 = max(0, mouth_cy - max(1, ry // 2))
            teeth_y2 = min(fh, mouth_cy)
            cavity[teeth_y1:teeth_y2, mouth_cx - rx // 2:mouth_cx + rx // 2] = (210, 215, 225)
            alpha_lip = lip_mask * min(0.58, energy * 0.52)
            out = (out.astype(np.float32) * (1.0 - alpha_lip) + cavity.astype(np.float32) * alpha_lip).astype(np.uint8)

    # 3. Natural Single & Occasional Double Eye-Blinking (every ~3.8 seconds)
    blink_cycle = (t_sec + 0.75) % 3.8
    blink_strength = 0.0
    if blink_cycle < 0.15:
        blink_strength = math.sin((blink_cycle / 0.15) * math.pi)
    elif (int((t_sec + 0.75) // 3.8) % 2 == 1) and (0.24 <= blink_cycle < 0.38):
        # Occasional realistic double-blink
        blink_strength = 0.88 * math.sin(((blink_cycle - 0.24) / 0.14) * math.pi)

    if blink_strength > 0.05 and fh > 48:
        ey1, ey2 = int(fh * 0.26), int(fh * 0.41)
        eh = ey2 - ey1
        if eh > 4:
            upper_lid = out[ey1:ey1 + max(2, eh // 2), :]
            lid_stretched = cv2.resize(upper_lid, (fw, eh), interpolation=cv2.INTER_LINEAR)
            b_alpha = min(0.88, blink_strength * 0.88)
            blended_eyes = cv2.addWeighted(lid_stretched, b_alpha, out[ey1:ey2, :], 1.0 - b_alpha, 0.0)
            # Subtle upper eyelash fold line when eyelid descends
            lash_y = min(eh - 1, max(1, int(eh * 0.68 * blink_strength)))
            blended_eyes[lash_y, int(fw * 0.18):int(fw * 0.82)] = (
                blended_eyes[lash_y, int(fw * 0.18):int(fw * 0.82)].astype(np.float32) * 0.55
            ).astype(np.uint8)
            out[ey1:ey2, :] = blended_eyes

    return out


def synthesize_talking_presenter_frames(raw_avatar_path, audio_wav_path, fps=24, device="cuda"):
    """
    Pre-renders the lip-synced, blinking, head-swaying teacher portrait frames (680x820)
    for a block so they can be composited both in the Studio Split-Screen AND as a live
    Picture-in-Picture (PiP) presenter cam on PowerPoint slides and B-roll clips!
    """
    presenter_w, presenter_h = 680, 820
    presenter_bgr, (x1, y1, x2, y2), _ = prepare_presenter_portrait(raw_avatar_path, presenter_w, presenter_h, device=device)

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

    # Smooth energies slightly over 3-frame window for natural jaw physics
    if len(frame_energies) >= 3:
        smoothed = np.convolve(frame_energies, [0.25, 0.50, 0.25], mode="same")
        frame_energies = [float(x) for x in smoothed]

    face_crop_orig = presenter_bgr[y1:y2, x1:x2].copy()
    fh, fw = face_crop_orig.shape[:2]
    mouth_mask = _create_full_opacity_mouth_mask(fw, fh)

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

    return {
        "presenter_bgr": presenter_bgr,
        "bbox": (x1, y1, x2, y2),
        "face_crop_orig": face_crop_orig,
        "mouth_mask": mouth_mask,
        "predicted_faces": predicted_faces,
        "frame_energies": frame_energies,
        "num_frames": num_frames,
        "fps": fps,
    }


def get_animated_presenter_frame(talk_state, f_idx):
    """
    Returns the single 680x820 animated presenter frame for frame index `f_idx`
    with Wav2Lip-GAN mouth synthesis, syllable jaw reinforcement, eye blinking,
    and LivePortrait-style head sway.
    """
    presenter_bgr = talk_state["presenter_bgr"]
    x1, y1, x2, y2 = talk_state["bbox"]
    face_crop_orig = talk_state["face_crop_orig"]
    mouth_mask = talk_state["mouth_mask"]
    predicted_faces = talk_state["predicted_faces"]
    frame_energies = talk_state["frame_energies"]
    fps = talk_state["fps"]

    presenter_h, presenter_w = presenter_bgr.shape[:2]
    fh, fw = face_crop_orig.shape[:2]
    energy = frame_energies[min(f_idx, len(frame_energies) - 1)]

    frame_presenter = presenter_bgr.copy()
    if f_idx < len(predicted_faces) and fw > 16 and fh > 16:
        pred_resized = cv2.resize(predicted_faces[f_idx], (fw, fh), interpolation=cv2.INTER_LANCZOS4)
        # Sharpen synthesized mouth/teeth details
        blur = cv2.GaussianBlur(pred_resized, (0, 0), 2.0)
        pred_sharp = cv2.addWeighted(pred_resized, 1.35, blur, -0.35, 0)
        blended_face = (
            pred_sharp.astype(np.float32) * mouth_mask
            + face_crop_orig.astype(np.float32) * (1.0 - mouth_mask)
        ).astype(np.uint8)
    else:
        blended_face = face_crop_orig.copy()

    # Apply syllable lip/jaw reinforcement & natural eye blinking
    blended_face = _apply_syllable_mouth_and_blink(blended_face, energy, f_idx, fps=fps)
    frame_presenter[y1:y2, x1:x2] = blended_face

    # LivePortrait-style natural head sway, breathing & speech-driven nod
    t_sec = f_idx / float(fps)
    scale = 1.016 + 0.008 * math.sin(2.0 * math.pi * 0.38 * t_sec) + 0.007 * energy
    angle_deg = 1.15 * math.sin(2.0 * math.pi * 0.27 * t_sec) + 0.6 * math.cos(2.0 * math.pi * 0.65 * t_sec) * energy
    tx = 3.8 * math.sin(2.0 * math.pi * 0.24 * t_sec)
    ty = 2.5 * math.cos(2.0 * math.pi * 0.42 * t_sec) - 3.5 * energy

    M = cv2.getRotationMatrix2D((presenter_w * 0.5, presenter_h * 0.46), angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    animated_presenter = cv2.warpAffine(
        frame_presenter, M, (presenter_w, presenter_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
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

    if raw_avatar_path and os.path.exists(raw_avatar_path):
        bg_pil = Image.open(raw_avatar_path).convert("RGB").resize((W, H), Image.Resampling.BICUBIC)
        bg_pil = bg_pil.filter(ImageFilter.GaussianBlur(radius=38))
        tint_layer = Image.new("RGB", (W, H), tint_rgb)
        bg_pil = Image.blend(bg_pil, tint_layer, alpha=0.76)
    else:
        bg_pil = Image.new("RGB", (W, H), tint_rgb)

    draw = ImageDraw.Draw(bg_pil, "RGBA")

    # Top studio header bar (enlarged accessibility fonts)
    draw.rounded_rectangle([64, 18, 1856, 76], radius=16, fill=theme["header_bg"], outline=accent + (210,), width=3)
    font_header = _load_font(27, bold=True)
    font_badge = _load_font(22, bold=True)
    draw.text((88, 30), f"  {lesson_title.upper()[:54]}", font=font_header, fill=(255, 255, 255, 255))
    draw.text((1580, 33), f"MODULE {block_idx + 1} OF {total_blocks}", font=font_badge, fill=accent_alt + (255,))

    # Determine coordinates based on presenter_side ("left" vs "right")
    if presenter_side == "right":
        pres_x, pres_y = 1176, 88
        card_x1, card_x2 = 64, 1136
    else:
        pres_x, pres_y = 64, 88
        card_x1, card_x2 = 784, 1856

    # Concept Card (width = 1072, y = 88..912)
    draw.rounded_rectangle([card_x1, 88, card_x2, 912], radius=26, fill=theme["card_bg"], outline=accent + (230,), width=3)
    draw.rounded_rectangle([card_x1, 88, card_x2, 190], radius=26, fill=theme["card_header_bg"])
    draw.rectangle([card_x1, 158, card_x2, 190], fill=theme["card_header_bg"])
    draw.rectangle([card_x1, 182, card_x2, 190], fill=accent + (255,))

    # Enlarged accessibility heading (40px bold)
    font_slide_title = _load_font(40, bold=True)
    heading = (slide_title or lesson_title or f"Key Concept #{block_idx + 1}").strip()
    draw.text((card_x1 + 36, 114), heading[:40], font=font_slide_title, fill=(255, 255, 255, 255))

    notes_max_w = 990
    notes_start_y = 214
    if cf_image_path and os.path.exists(cf_image_path):
        try:
            cf_img = Image.open(cf_image_path).convert("RGB")
            tw, th = 992, 336
            iw, ih = cf_img.size
            sc = max(tw / float(iw), th / float(ih))
            cf_img = cf_img.resize((int(iw * sc), int(ih * sc)), Image.Resampling.LANCZOS)
            lx = (cf_img.size[0] - tw) // 2
            ly = (cf_img.size[1] - th) // 2
            cf_img = cf_img.crop((lx, ly, lx + tw, ly + th))
            mask = Image.new("L", (tw, th), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw, th], radius=18, fill=255)
            bg_pil.paste(cf_img, (card_x1 + 40, 204), mask)
            notes_start_y = 560
        except Exception as e:
            print(f"Studio CF image embed note: {e}")

    # Enlarged accessibility bullet font (35px bold with high contrast)
    font_bullet = _load_font(35, bold=True)
    curr_y = notes_start_y
    bullet_colors = [accent, accent_alt, (16, 185, 129), (59, 130, 246)]

    for idx, note in enumerate((slide_notes or [])[:4]):
        clean_note = str(note).strip()
        if not clean_note:
            continue
        lines = _wrap_text(draw, clean_note, font_bullet, notes_max_w - 72)
        b_col = bullet_colors[idx % len(bullet_colors)]
        draw.rounded_rectangle([card_x1 + 40, curr_y + 5, card_x1 + 78, curr_y + 43], radius=10, fill=b_col + (255,))
        for line in lines[:2]:
            draw.text((card_x1 + 94, curr_y), line, font=font_bullet, fill=theme["card_text"])
            curr_y += 45
        curr_y += 14
        if curr_y > 872:
            break

    # Presenter Frame glowing border
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


def render_data_chart_or_formula_card(chart_data, width, height, theme, slide_title=""):
    """
    Renders a crisp, broadcast-grade Data Chart, Comparison Table, Process Flowchart,
    or Scientific Formula Board onto a PIL RGB Image of size (width, height).
    Supported chart_data['type'] values:
      - 'bar'        : Horizontal/Vertical comparative metric bars with value badges
      - 'comparison' : Two-column or multi-row structured comparison matrix
      - 'formula'    : Highlighted mathematical / scientific equation board with variable legend
      - 'process'    : Step-by-step connected flowchart nodes
    """
    accent = theme["accent"]
    accent_alt = theme["accent_alt"]
    card = Image.new("RGB", (width, height), (13, 20, 38))
    draw = ImageDraw.Draw(card, "RGBA")

    # Subtle grid lines for technical broadcast look
    for gx in range(40, width, 60):
        draw.line([(gx, 0), (gx, height)], fill=(255, 255, 255, 13), width=1)
    for gy in range(40, height, 60):
        draw.line([(0, gy), (width, gy)], fill=(255, 255, 255, 13), width=1)

    ctype = str(chart_data.get("type") or "bar").strip().lower()
    ctitle = str(chart_data.get("title") or slide_title or "Key Data & Metrics").strip()
    unit = str(chart_data.get("unit") or "").strip()

    # Header bar inside chart card
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
        eq_lines = _wrap_text(draw, formula_str, font_eq, width - 110)
        ey = 176
        for el in eq_lines[:2]:
            draw.text((54, ey), el, font=font_eq, fill=(255, 255, 255, 255))
            ey += 54

        font_leg = _load_font(25, bold=True)
        ly = 338
        for idx, lbl in enumerate(labels[:5]):
            b_col = accent if idx % 2 == 0 else accent_alt
            draw.ellipse([40, ly + 6, 58, ly + 24], fill=b_col + (255,))
            w_lines = _wrap_text(draw, lbl, font_leg, width - 110)
            for wl in w_lines[:2]:
                draw.text((74, ly), wl, font=font_leg, fill=(235, 242, 255, 255))
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
        for i, st in enumerate(steps):
            col = palette[i % len(palette)]
            draw.rounded_rectangle([36, curr_y, width - 36, curr_y + box_h], radius=16, fill=(24, 36, 68, 245), outline=col + (235,), width=3)
            draw.ellipse([54, curr_y + (box_h - 44) // 2, 98, curr_y + (box_h + 44) // 2], fill=col + (255,))
            draw.text((69, curr_y + (box_h - 30) // 2), str(i + 1), font=font_num, fill=(15, 23, 42, 255))
            s_lines = _wrap_text(draw, st, font_step, width - 170)
            ty = curr_y + (box_h - len(s_lines[:2]) * 30) // 2
            for sl in s_lines[:2]:
                draw.text((116, ty), sl, font=font_step, fill=(255, 255, 255, 255))
                ty += 30
            if i < n_steps - 1:
                cx = width // 2
                draw.line([(cx, curr_y + box_h), (cx, curr_y + box_h + 16)], fill=accent_alt + (255,), width=4)
            curr_y += box_h + 18
        return card

    # Default: Horizontal Animated-Style Data Bar Chart
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
        ratio = max(0.12, min(1.0, abs(v) / max_v))
        b_col = bar_colors[idx % len(bar_colors)]
        val_str = f"{v:g}{unit}" if unit else f"{v:g}"

        draw.text((40, curr_y), lbl[:34], font=font_lbl, fill=(241, 245, 249, 255))
        draw.text((width - 145, curr_y), val_str[:10], font=font_val, fill=accent_alt + (255,))
        curr_y += 36

        # Track background + filled metric bar
        draw.rounded_rectangle([40, curr_y, 40 + max_bar_w, curr_y + 34], radius=12, fill=(30, 41, 59, 255))
        fill_w = max(28, int(max_bar_w * ratio))
        draw.rounded_rectangle([40, curr_y, 40 + fill_w, curr_y + 34], radius=12, fill=b_col + (255,))
        curr_y += 58
        if curr_y > height - 55:
            break

    return card


def generate_youtube_thumbnail(
    raw_avatar_path,
    cf_image_path,
    lesson_title,
    presenter_name="",
    channel_name="",
    visual_template="broadcast_studio",
    output_jpg_path="thumbnail.jpg"
):
    """
    Generates a high-CTR 1280x720 YouTube Thumbnail (.jpg) combining:
      - Full-bleed darkened & color-graded AI visual backdrop
      - Dramatic studio glow + cut-out framed Teacher Portrait on the right
      - Giant, high-contrast 3-6 word Hook Headline Banner on the left
      - Branded Channel / Presenter verification badge
    """
    W, H = 1280, 720
    theme = resolve_studio_template(visual_template, block_idx=0)
    accent = theme["accent"]
    accent_alt = theme["accent_alt"]

    if cf_image_path and os.path.exists(cf_image_path):
        try:
            bg = Image.open(cf_image_path).convert("RGB")
            iw, ih = bg.size
            sc = max(W / float(max(1, iw)), H / float(max(1, ih)))
            bg = bg.resize((int(iw * sc), int(ih * sc)), Image.Resampling.LANCZOS)
            lx, ly = (bg.size[0] - W) // 2, (bg.size[1] - H) // 2
            bg = bg.crop((lx, ly, lx + W, ly + H))
            dark = Image.new("RGB", (W, H), theme["bg_tint"])
            bg = Image.blend(bg, dark, alpha=0.68)
        except Exception:
            bg = Image.new("RGB", (W, H), theme["bg_tint"])
    else:
        bg = Image.new("RGB", (W, H), theme["bg_tint"])

    draw = ImageDraw.Draw(bg, "RGBA")

    # Radiant spotlight glow behind presenter & headline
    draw.ellipse([660, 40, 1340, 740], fill=accent + (88,))
    draw.ellipse([-140, -100, 680, 620], fill=(10, 15, 30, 210))

    # Left-side high-contrast Hook Typography
    badge_txt = (channel_name or "MASTERCLASS LESSON").strip().upper()[:28]
    draw.rounded_rectangle([54, 48, 490, 106], radius=14, fill=accent + (245,))
    font_badge = _load_font(25, bold=True)
    draw.text((76, 62), badge_txt, font=font_badge, fill=(10, 15, 30, 255))

    font_hook = _load_font(68, bold=True)
    clean_title = (lesson_title or "COMPLETE VIDEO MASTERCLASS").strip().upper()
    hook_lines = _wrap_text(draw, clean_title, font_hook, 690)
    hy = 136
    for idx, line in enumerate(hook_lines[:4]):
        # Drop shadow
        draw.text((60, hy + 5), line, font=font_hook, fill=(0, 0, 0, 230))
        fill_col = accent_alt + (255,) if idx == len(hook_lines[:4]) - 1 else (255, 255, 255, 255)
        draw.text((54, hy), line, font=font_hook, fill=fill_col)
        hy += 82

    if presenter_name:
        draw.rounded_rectangle([54, H - 112, 640, H - 46], radius=16, fill=(15, 23, 42, 235), outline=accent_alt + (220,), width=3)
        font_pres = _load_font(28, bold=True)
        draw.text((80, H - 94), f"WITH {presenter_name.upper()[:32]}", font=font_pres, fill=(255, 255, 255, 255))

    # Right-side Framed Presenter Portrait (x=780..1228, y=54..666)
    if raw_avatar_path and os.path.exists(raw_avatar_path):
        try:
            pres_bgr, _, _ = prepare_presenter_portrait(raw_avatar_path, 440, 590, device="cpu")
            pres_rgb = cv2.cvtColor(pres_bgr, cv2.COLOR_BGR2RGB)
            pres_pil = Image.fromarray(pres_rgb)
            draw.rounded_rectangle([778, 56, 1230, 658], radius=30, fill=(15, 23, 42, 255), outline=accent_alt + (255,), width=6)
            mask = Image.new("L", (440, 590), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, 440, 590], radius=24, fill=255)
            bg.paste(pres_pil, (784, 62), mask)
        except Exception as e:
            print(f"Thumbnail portrait note: {e}")

    # Outer broadcast border frame
    draw.rectangle([6, 6, W - 7, H - 7], outline=accent + (220,), width=6)
    bg.save(output_jpg_path, "JPEG", quality=93)
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
    theme=None
):
    """
    Renders three high-retention broadcast television graphics onto `frame_bgr`:
      1. Sleek Intro Title Badge during the first 2.6 seconds of Block 1
      2. Animated Slide-Out Lower-Third Teacher Nameplate (t = 1.2s .. 7.8s in Block 1)
      3. Branded YouTube Outro End-Screen Call-To-Action Banner during the final 3.6s of the last block
    """
    H, W = frame_bgr.shape[:2]
    accent = (theme or {}).get("accent", (56, 189, 248))
    accent_alt = (theme or {}).get("accent_alt", (250, 204, 21))

    show_intro = (block_idx == 0 and t_local < 2.4)
    show_lower_third = (block_idx == 0 and 1.2 <= t_local <= 7.8 and (presenter_name or channel_name))
    time_remaining = max(0.0, total_video_dur - t_global)
    show_outro = (block_idx == total_blocks - 1 and time_remaining <= 3.6)

    if not (show_intro or show_lower_third or show_outro):
        return frame_bgr

    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # 1. Animated Lower-Third Presenter Nameplate (t = 1.2s .. 7.8s)
    if show_lower_third:
        if t_local < 1.8:
            prog = (t_local - 1.2) / 0.6
        elif t_local > 7.2:
            prog = max(0.0, (7.8 - t_local) / 0.6)
        else:
            prog = 1.0
        ease = math.sin(prog * math.pi * 0.5)
        box_w = min(W - 80, 620)
        x_left = int(-box_w + (box_w + 68) * ease)
        y_bot = H - 148 if W >= H else H - 280
        y_top = y_bot - 108

        draw.rounded_rectangle([x_left, y_top, x_left + box_w, y_bot], radius=18, fill=(10, 16, 34, 238), outline=accent + (240,), width=3)
        draw.rounded_rectangle([x_left, y_top, x_left + 18, y_bot], radius=8, fill=accent_alt + (255,))
        font_name = _load_font(31, bold=True)
        font_role = _load_font(22, bold=True)
        main_lbl = (presenter_name or channel_name or "Course Instructor").strip()
        sub_lbl = (channel_name if presenter_name and channel_name else (lesson_title or "Featured Masterclass")).strip()
        draw.text((x_left + 34, y_top + 18), main_lbl[:34], font=font_name, fill=(255, 255, 255, 255))
        draw.text((x_left + 34, y_top + 60), sub_lbl.upper()[:42], font=font_role, fill=accent_alt + (255,))

    # 2. Branded Intro Module Pill (first 2.4s of Block 1)
    if show_intro:
        alpha_i = int(235 * min(1.0, (2.4 - t_local) / 0.6))
        bw = min(W - 80, 740)
        bx1 = (W - bw) // 2
        by1 = 92 if W >= H else 160
        draw.rounded_rectangle([bx1, by1, bx1 + bw, by1 + 64], radius=32, fill=(15, 23, 42, alpha_i), outline=accent_alt + (alpha_i,), width=3)
        font_ib = _load_font(25, bold=True)
        ch_prefix = f"{channel_name.upper()}  •  " if channel_name else "OFFICIAL LESSON  •  "
        draw.text((bx1 + 32, by1 + 17), f"{ch_prefix}{lesson_title[:36].upper()}", font=font_ib, fill=(255, 255, 255, alpha_i))

    # 3. YouTube Outro End-Screen Call-To-Action Banner (final 3.6s of last block)
    if show_outro:
        prog_o = min(1.0, (3.6 - time_remaining) / 0.5)
        alpha_o = int(242 * prog_o)
        bw = min(W - 80, 920)
        bx1 = (W - bw) // 2
        by1 = H - 134 if W >= H else H - 240
        draw.rounded_rectangle([bx1, by1, bx1 + bw, by1 + 88], radius=22, fill=(12, 18, 38, alpha_o), outline=accent_alt + (alpha_o,), width=3)
        draw.rounded_rectangle([bx1 + 20, by1 + 18, bx1 + 230, by1 + 70], radius=14, fill=(239, 68, 68, alpha_o))
        font_sub = _load_font(25, bold=True)
        font_cta = _load_font(27, bold=True)
        draw.text((bx1 + 44, by1 + 30), "SUBSCRIBE", font=font_sub, fill=(255, 255, 255, alpha_o))
        cta_msg = f"Thanks for watching {channel_name or 'this lesson'}! Like & Share for more."
        draw.text((bx1 + 254, by1 + 28), cta_msg[:46], font=font_cta, fill=(255, 255, 255, alpha_o))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


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
    Renders a dynamic, high-contrast 1920x1080 PowerPoint Presentation Slide with:
      - Enlarged accessibility typography (40px bold bullets, 50px bold headers) for visually impaired learners
      - Real Data Chart / Formula / Process Diagram rendering when `chart_data` is provided on Slide 2
      - Template-specific color palette & left/right column swapping
      - Returns (output_png, img_box_coords) so Ken-Burns zoom targets the exact illustration box
    """
    W, H = 1920, 1080
    theme = resolve_studio_template(visual_template, block_idx=block_idx)
    accent = theme["accent"]
    accent_alt = theme["accent_alt"]

    use_dark = (theme_variant % 2 == 1) or (visual_template in ("ted_keynote", "academic_chalkboard", "cyber_stem_lab"))
    bg_col = theme["slide_bg_dark"] if use_dark else theme["slide_bg_light"]
    is_bg_dark = (int(bg_col[0]) + int(bg_col[1]) + int(bg_col[2])) < 380

    card_col = (theme["card_bg"][0], theme["card_bg"][1], theme["card_bg"][2], 255) if is_bg_dark else (255, 255, 255, 255)
    text_col = (255, 255, 255, 255) if is_bg_dark else (15, 23, 42, 255)
    border_col = accent + (220,)

    slide = Image.new("RGB", (W, H), bg_col)
    draw = ImageDraw.Draw(slide, "RGBA")

    # Top PowerPoint Header Banner (y=0..154) with enlarged accessibility text
    draw.rectangle([0, 0, W, 154], fill=theme["card_header_bg"])
    draw.rectangle([0, 145, W, 154], fill=accent + (255,))

    font_course = _load_font(25, bold=True)
    font_heading = _load_font(48, bold=True)
    draw.text((72, 20), f"{lesson_title.upper()[:60]}   •   SLIDE {block_idx + 1} OF {total_blocks}", font=font_course, fill=accent_alt + (255,))
    draw.text((72, 60), (slide_title or lesson_title)[:44], font=font_heading, fill=(255, 255, 255, 255))

    img_on_left = (theme["slide_img_side"] == "left")
    if img_on_left:
        img_card_x1, img_card_x2 = 64, 856
        img_paste_x = 80
        notes_card_x1, notes_card_x2 = 884, 1856
    else:
        notes_card_x1, notes_card_x2 = 64, 1036
        img_card_x1, img_card_x2 = 1064, 1856
        img_paste_x = 1080

    # Summarized Bullet Points Card (y=176..924)
    draw.rounded_rectangle([notes_card_x1, 176, notes_card_x2, 924], radius=26, fill=card_col, outline=border_col, width=3)
    font_sub = _load_font(28, bold=True)
    draw.text((notes_card_x1 + 40, 204), "KEY SUMMARY NOTES & TAKEAWAYS", font=font_sub, fill=accent_alt + (255,) if is_bg_dark else accent + (255,))
    draw.line([notes_card_x1 + 40, 248, notes_card_x2 - 40, 248], fill=border_col, width=2)

    notes_list = [str(n).strip() for n in (slide_notes or []) if str(n).strip()]
    if not reveal_all and len(notes_list) > 1:
        notes_list = notes_list[:max(1, (len(notes_list) + 1) // 2)]

    # Enlarged 40px bold bullet font for visually impaired learners
    font_bullet = _load_font(39, bold=True)
    font_num = _load_font(29, bold=True)
    curr_y = 274
    badge_colors = [accent, (16, 185, 129), (245, 158, 11), (139, 92, 246)]

    for i, clean in enumerate(notes_list[:4]):
        b_col = badge_colors[i % len(badge_colors)]
        draw.rounded_rectangle([notes_card_x1 + 40, curr_y + 4, notes_card_x1 + 94, curr_y + 56], radius=14, fill=b_col + (255,))
        draw.text((notes_card_x1 + 57, curr_y + 11), str(i + 1), font=font_num, fill=(255, 255, 255, 255))

        wrapped = _wrap_text(draw, clean, font_bullet, 790)
        for line in wrapped[:3]:
            draw.text((notes_card_x1 + 114, curr_y + 2), line, font=font_bullet, fill=text_col)
            curr_y += 50
        curr_y += 26
        if curr_y > 872:
            break

    # High-Res Visual Illustration OR Data Chart / Formula Card (760x716)
    draw.rounded_rectangle([img_card_x1, 176, img_card_x2, 924], radius=26, fill=(15, 23, 42, 255), outline=accent + (225,), width=3)
    target_w, target_h = 760, 716

    if isinstance(chart_data, dict) and (chart_data.get("labels") or chart_data.get("formula")):
        try:
            chart_pil = render_data_chart_or_formula_card(chart_data, target_w, target_h, theme, slide_title=slide_title)
            mask = Image.new("L", (target_w, target_h), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, target_w, target_h], radius=20, fill=255)
            slide.paste(chart_pil, (img_paste_x, 192), mask)
        except Exception as e:
            print(f"Chart card render note: {e}")
    elif cf_image_path and os.path.exists(cf_image_path):
        try:
            ill = Image.open(cf_image_path).convert("RGB")
            iw, ih = ill.size
            scale = max(target_w / float(iw), target_h / float(ih))
            nw, nh = int(iw * scale), int(ih * scale)
            ill = ill.resize((nw, nh), Image.Resampling.LANCZOS)
            left = (nw - target_w) // 2
            top = (nh - target_h) // 2
            ill = ill.crop((left, top, left + target_w, top + target_h))

            mask = Image.new("L", (target_w, target_h), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, target_w, target_h], radius=20, fill=255)
            slide.paste(ill, (img_paste_x, 192), mask)
        except Exception as e:
            print(f"Slide image paste note: {e}")

    slide.save(output_png, "PNG")
    return output_png, (img_paste_x, 192, img_paste_x + 760, 192 + 716)


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

    # Floating Key Note Callout Card with enlarged 38px bold font for visually impaired learners
    if note_text:
        pip_pos = theme.get("pip_broll_pos", "bottom_right")
        c_x1 = 410 if pip_pos == "bottom_left" else 64
        c_x2 = 1650 if pip_pos == "bottom_left" else 1300
        draw.rounded_rectangle([c_x1, 718, c_x2, 916], radius=22, fill=(10, 15, 30, 236), outline=accent + (230,), width=3)
        draw.rounded_rectangle([c_x1, 718, c_x1 + 22, 916], radius=8, fill=accent_alt + (255,))
        font_note = _load_font(37, bold=True)
        lines = _wrap_text(draw, note_text, font_note, (c_x2 - c_x1) - 76)
        y = 744
        for line in lines[:3]:
            draw.text((c_x1 + 44, y), line, font=font_note, fill=(255, 255, 255, 255))
            y += 49

    ov_np = np.array(ov)
    rgb = ov_np[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    alpha = (ov_np[:, :, 3].astype(np.float32) / 255.0)[..., None]
    return bgr, alpha


def _overlay_pip_presenter(canvas_bgr, presenter_frame_bgr, pos="bottom_right", border_rgb=(56, 189, 248)):
    """
    Overlays a live Lip-Synced Teacher Picture-in-Picture (PiP) rounded card
    onto PowerPoint Slides and B-Roll Clips at top_right, bottom_right, top_left, or bottom_left.
    """
    pip_w, pip_h = 250, 302
    pip_img = cv2.resize(presenter_frame_bgr, (pip_w, pip_h), interpolation=cv2.INTER_AREA)

    if pos == "top_right":
        x0, y0 = 1920 - pip_w - 48, 162
    elif pos == "top_left":
        x0, y0 = 48, 162
    elif pos == "bottom_left":
        x0, y0 = 48, 1080 - pip_h - 148
    else:
        x0, y0 = 1920 - pip_w - 48, 1080 - pip_h - 148

    border_bgr = (int(border_rgb[2]), int(border_rgb[1]), int(border_rgb[0]))
    cv2.rectangle(canvas_bgr, (x0 - 4, y0 - 4), (x0 + pip_w + 4, y0 + pip_h + 4), border_bgr, 4, cv2.LINE_AA)
    canvas_bgr[y0:y0 + pip_h, x0:x0 + pip_w] = pip_img
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
    Adds subtle moving cinema bokeh light particles to B-roll video clips so they
    look like dynamic motion footage rather than static images.
    """
    t = f_idx / float(fps)
    overlay = canvas_bgr.copy()
    H, W = canvas_bgr.shape[:2]
    particles = [
        (320, 280, 42, 0.35, 0.25, (248, 210, 120)),
        (860, 480, 58, -0.28, 0.30, (255, 230, 160)),
        (1420, 340, 36, 0.40, -0.22, (248, 189, 56)),
        (1620, 620, 52, -0.32, -0.28, (200, 240, 255)),
        (580, 680, 46, 0.25, -0.35, (255, 220, 140)),
    ]
    for idx, (bx, by, rad, vx, vy, col) in enumerate(particles):
        px = int((bx + math.sin(t * vx * 2.0 + idx) * 140) % max(1, W))
        py = int((by + math.cos(t * vy * 2.0 + idx) * 90) % max(1, H))
        cv2.circle(overlay, (px, py), rad, col, -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.12, canvas_bgr, 0.88, 0.0)


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
    fps=24,
    device="cuda",
    run_ffmpeg_fn=None,
    progress_cb=None
):
    """
    Rescales the 10-Minute Video Blueprint proportionally to ANY video duration (60s, 90s, 180s, 600s),
    applies the user's selected Studio Visual Template (out of 6 distinct broadcast templates),
    renders optional Data Charts / Scientific Formulas onto slides, overlays branded Intro /
    Lower-Third / Outro graphics, and supports both 16:9 Widescreen and 9:16 Vertical Shorts/Reels.
    """
    theme = resolve_studio_template(visual_template, block_idx=block_idx)
    accent_rgb = theme["accent"]
    pip_slide_pos = theme.get("pip_slide_pos", "top_right")
    pip_broll_pos = theme.get("pip_broll_pos", "bottom_right")
    is_vertical = str(aspect_ratio or "16:9").strip() == "9:16"
    out_w, out_h = (1080, 1920) if is_vertical else (1920, 1080)

    img1 = cf_images[0] if cf_images and len(cf_images) > 0 else None
    img2 = cf_images[1] if cf_images and len(cf_images) > 1 else img1
    img3 = cf_images[2] if cf_images and len(cf_images) > 2 else img2

    vid_clip_1 = pexels_video_clips[0] if pexels_video_clips and len(pexels_video_clips) > 0 else None
    vid_clip_2 = pexels_video_clips[1] if pexels_video_clips and len(pexels_video_clips) > 1 else vid_clip_1

    pexels_stream_1 = _PexelsVideoStream(vid_clip_1, 1920, 1080)
    pexels_stream_2 = _PexelsVideoStream(vid_clip_2, 1920, 1080)

    # 1. Pre-compute talking presenter state (S3FD + Wav2Lip-GAN + Syllable Jaw + Eye Blink + Eyebrow Lift)
    talk_state = synthesize_talking_presenter_frames(raw_avatar_path, audio_wav_path, fps=fps, device=device)
    num_frames = talk_state["num_frames"]
    block_dur_sec = num_frames / float(fps)

    if callable(progress_cb):
        try:
            progress_cb(0.35, f"Wav2Lip-GAN neural lip-sync complete; building {theme['name']} slides...")
        except Exception:
            pass

    # 2. Build Studio Composite Background, PowerPoint Slide 1 & 2 (with optional Chart/Formula), and B-Roll Canvases
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

    p1_png = f"{output_mp4_path}.s1.png"
    p2_png = f"{output_mp4_path}.s2.png"
    _, slide_img_box = render_powerpoint_slide_frame(
        lesson_title=lesson_title,
        slide_title=slide_title,
        slide_notes=slide_notes,
        cf_image_path=img1,
        raw_avatar_path=raw_avatar_path,
        output_png=p1_png,
        block_idx=block_idx,
        total_blocks=total_blocks,
        reveal_all=False,
        theme_variant=block_idx,
        visual_template=visual_template,
        chart_data=None
    )
    render_powerpoint_slide_frame(
        lesson_title=lesson_title,
        slide_title=slide_title,
        slide_notes=slide_notes,
        cf_image_path=img3,
        raw_avatar_path=raw_avatar_path,
        output_png=p2_png,
        block_idx=block_idx,
        total_blocks=total_blocks,
        reveal_all=True,
        theme_variant=block_idx + 1,
        visual_template=visual_template,
        chart_data=chart_data
    )
    slide1_bgr = cv2.imread(p1_png)
    slide2_bgr = cv2.imread(p2_png)

    broll_canvas_1 = _prepare_broll_canvas(img2, 1920, 1080)
    broll_canvas_2 = _prepare_broll_canvas(img3, 1920, 1080)
    notes_list = [str(n).strip() for n in (slide_notes or []) if str(n).strip()]
    broll_note_1 = notes_list[0] if len(notes_list) > 0 else slide_title
    broll_note_2 = notes_list[-1] if len(notes_list) > 1 else broll_note_1
    broll_ov1_bgr, broll_ov1_a = _build_broll_overlay_rgba(lesson_title, slide_title, broll_note_1, block_idx, total_blocks, visual_template=visual_template)
    broll_ov2_bgr, broll_ov2_a = _build_broll_overlay_rgba(lesson_title, slide_title, broll_note_2, block_idx, total_blocks, visual_template=visual_template)

    # 3. Rescale the 10-minute (600s) blueprint to the actual total video duration D
    D = max(block_dur_sec, float(total_video_duration_sec))
    t_intro_window = min(30.0, max(8.0, D * 0.15))
    t_outro_window = min(30.0, max(8.0, D * 0.15))

    def _get_mode_for_frame(f_idx):
        t_local = f_idx / float(fps)
        t_global = block_start_sec + t_local
        frac_local = t_local / max(1.0, block_dur_sec)

        if block_idx == 0 or t_global < t_intro_window:
            if frac_local < 0.56:
                return "studio_lipsync"
            elif frac_local < 0.78:
                return "broll_1"
            else:
                return "slide_1"

        if block_idx == total_blocks - 1 or t_global > (D - t_outro_window):
            if frac_local < 0.22:
                return "slide_2"
            elif frac_local < 0.44:
                return "broll_1"
            else:
                return "studio_lipsync"

        if visual_blueprint == "90s_avatar":
            if frac_local < 0.45:
                return "studio_lipsync"
            elif frac_local < 0.72:
                return "broll_1"
            elif frac_local < 0.88:
                return "slide_2"
            else:
                return "studio_lipsync"

        if frac_local < 0.28:
            return "slide_1"
        elif frac_local < 0.56:
            return "broll_1"
        elif frac_local < 0.80:
            return "slide_2"
        else:
            return "broll_2" if (block_idx % 2 == 0 or pexels_stream_2.is_valid()) else "studio_lipsync"

    last_broll_frame = {"broll_1": (-1, None), "broll_2": (-1, None)}

    def _render_single_mode_frame(mode_name, f_idx):
        pres_bgr = get_animated_presenter_frame(talk_state, f_idx)
        prog = (f_idx % max(1, num_frames // 3)) / float(max(1, num_frames // 3))
        W, H = 1920, 1080

        if mode_name == "studio_lipsync":
            full_frame = studio_bg_bgr.copy()
            comp_slot = (pres_bgr.astype(np.float32) * card_alpha + base_slot * (1.0 - card_alpha)).astype(np.uint8)
            full_frame[pres_slot_y:pres_slot_y + presenter_h, pres_slot_x:pres_slot_x + presenter_w] = comp_slot
            wide_out = _apply_organic_camera_breathing(full_frame, f_idx, fps=fps)

        elif mode_name in ("slide_1", "slide_2"):
            base = (slide1_bgr if mode_name == "slide_1" else slide2_bgr).copy()
            ix1, iy1, ix2, iy2 = slide_img_box
            iw, ih = ix2 - ix1, iy2 - iy1
            zoom = 1.0 + 0.06 * prog
            sub_img = base[iy1:iy2, ix1:ix2]
            M = np.float32([
                [zoom, 0.0, (1.0 - zoom) * (iw * 0.5)],
                [0.0, zoom, (1.0 - zoom) * (ih * 0.5)]
            ])
            base[iy1:iy2, ix1:ix2] = cv2.warpAffine(sub_img, M, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            base = _overlay_pip_presenter(base, pres_bgr, pos=pip_slide_pos, border_rgb=accent_rgb)
            wide_out = _apply_organic_camera_breathing(base, f_idx, fps=fps)

        else:
            is_second = (mode_name == "broll_2")
            ov_bgr = broll_ov2_bgr if is_second else broll_ov1_bgr
            ov_a = broll_ov2_a if is_second else broll_ov1_a
            p_stream = pexels_stream_2 if is_second else pexels_stream_1

            cached_idx, cached_img = last_broll_frame.get(mode_name, (-1, None))
            if cached_idx == f_idx and cached_img is not None:
                moved = cached_img
            else:
                pexels_bgr = p_stream.read_frame() if p_stream.is_valid() else None
                if pexels_bgr is not None:
                    moved = pexels_bgr
                else:
                    canvas = broll_canvas_2 if is_second else broll_canvas_1
                    zoom = (1.14 - 0.10 * prog) if is_second else (1.02 + 0.12 * prog)
                    tx = (22.0 * (1.0 - prog)) if is_second else (-24.0 * prog)
                    ty = (12.0 * (1.0 - prog)) if is_second else (-14.0 * prog)
                    M = np.float32([
                        [zoom, 0.0, (1.0 - zoom) * (W * 0.5) + tx],
                        [0.0, zoom, (1.0 - zoom) * (H * 0.5) + ty]
                    ])
                    moved = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                    moved = _render_bokeh_particles(moved, f_idx, fps=fps)
                last_broll_frame[mode_name] = (f_idx, moved)

            comp = (moved.astype(np.float32) * (1.0 - ov_a) + ov_bgr * ov_a).astype(np.uint8)
            comp = _overlay_pip_presenter(comp, pres_bgr, pos=pip_broll_pos, border_rgb=accent_rgb)
            wide_out = comp

        if is_vertical:
            return _compose_vertical_916_frame(wide_out, pres_bgr, lesson_title, slide_title, slide_notes, theme, block_idx, total_blocks)
        return wide_out

    silent_mp4 = f"{output_mp4_path}.silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_mp4, fourcc, float(fps), (out_w, out_h))

    modes = [_get_mode_for_frame(i) for i in range(num_frames)]
    transition_len = 9
    tick_33 = max(1, num_frames // 3)
    tick_66 = max(2, (num_frames * 2) // 3)

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

        # Smooth 0.30s soft entrance fade at the start of every block (>0) for broadcast polish
        if block_idx > 0 and f_idx < 7:
            fade_a = (f_idx + 1) / 7.0
            dark_bg = np.full_like(frame_bgr, (15, 20, 34), dtype=np.uint8)
            frame_bgr = cv2.addWeighted(frame_bgr, fade_a, dark_bg, 1.0 - fade_a, 0.0)

        # Apply Intro Title Badge, Animated Lower-Third Teacher Nameplate & YouTube Outro End-Screen
        t_local = f_idx / float(fps)
        t_global = block_start_sec + t_local
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
            theme=theme
        )

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

    if callable(progress_cb):
        try:
            progress_cb(0.92, "Encoding H.264 video stream & burning active karaoke captions...")
        except Exception:
            pass

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
        run_ffmpeg_fn(mux_cmd, desc="Mux dynamic multi-scene block with audio")
    else:
        import subprocess
        subprocess.run(mux_cmd, check=True)

    for tmp in (silent_mp4, p1_png, p2_png):
        if os.path.exists(tmp):
            os.remove(tmp)

    return output_mp4_path


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

