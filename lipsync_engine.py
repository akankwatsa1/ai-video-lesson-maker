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
    Reinforces visible mouth/jaw articulation on spoken syllables and adds
    natural periodic eye blinks so even tight-lipped portrait photos visibly talk and blink.
    """
    fh, fw = face_bgr.shape[:2]
    out = face_bgr.copy()

    # 1. Syllable-driven lower-jaw & lip opening reinforcement
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
            # Feather horizontally so only central chin moves
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
            # Warm dark inner oral cavity tint + subtle upper teeth highlight
            cavity = np.full_like(out, (28, 22, 48), dtype=np.uint8)
            teeth_y1 = max(0, mouth_cy - max(1, ry // 2))
            teeth_y2 = min(fh, mouth_cy)
            cavity[teeth_y1:teeth_y2, mouth_cx - rx // 2:mouth_cx + rx // 2] = (210, 215, 225)
            alpha_lip = lip_mask * min(0.58, energy * 0.52)
            out = (out.astype(np.float32) * (1.0 - alpha_lip) + cavity.astype(np.float32) * alpha_lip).astype(np.uint8)

    # 2. Periodic natural eye blink every ~3.6 seconds (duration 4 frames)
    cycle_frames = int(3.6 * fps)
    phase = (f_idx + 19) % max(1, cycle_frames)
    if phase < 4 and fh > 48:
        blink_strength = 1.0 - abs(phase - 1.5) / 2.0
        ey1, ey2 = int(fh * 0.27), int(fh * 0.41)
        eh = ey2 - ey1
        if eh > 4:
            upper_lid = out[ey1:ey1 + max(2, eh // 2), :]
            lid_stretched = cv2.resize(upper_lid, (fw, eh), interpolation=cv2.INTER_LINEAR)
            b_alpha = min(0.85, blink_strength * 0.85)
            out[ey1:ey2, :] = cv2.addWeighted(lid_stretched, b_alpha, out[ey1:ey2, :], 1.0 - b_alpha, 0.0)

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
# 5. ENHANCED POWERPOINT SLIDES, STUDIO CARDS & PROCEDURAL INFOGRAPHICS
# ==============================================================================

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
      - Blurred & color-graded ambient backdrop from the teacher's photo
      - Left Frame (x=64, y=88, w=680, h=820): Rounded presenter studio slot
      - Right Card (x=784, y=88, w=1072, h=820): PowerPoint-style concept card with
        Cloudflare/Wikimedia illustration + enhanced bold typography
    """
    W, H = 1920, 1080
    if raw_avatar_path and os.path.exists(raw_avatar_path):
        bg_pil = Image.open(raw_avatar_path).convert("RGB").resize((W, H), Image.Resampling.BICUBIC)
        bg_pil = bg_pil.filter(ImageFilter.GaussianBlur(radius=38))
        navy_tint = Image.new("RGB", (W, H), (12, 18, 36))
        bg_pil = Image.blend(bg_pil, navy_tint, alpha=0.74)
    else:
        bg_pil = Image.new("RGB", (W, H), (12, 18, 36))

    draw = ImageDraw.Draw(bg_pil, "RGBA")

    # Top studio header bar
    draw.rounded_rectangle([64, 20, 1856, 74], radius=16, fill=(15, 23, 42, 225), outline=(56, 189, 248, 190), width=2)
    font_header = _load_font(24, bold=True)
    font_badge = _load_font(20, bold=True)
    draw.text((88, 33), f"  {lesson_title.upper()[:60]}", font=font_header, fill=(248, 250, 252, 255))
    draw.text((1610, 35), f"MODULE {block_idx + 1} OF {total_blocks}", font=font_badge, fill=(56, 189, 248, 255))

    # Right PowerPoint Concept Card (x=784..1856, y=88..908)
    draw.rounded_rectangle([784, 88, 1856, 908], radius=26, fill=(255, 255, 255, 250), outline=(56, 189, 248, 220), width=3)
    draw.rounded_rectangle([784, 88, 1856, 184], radius=26, fill=(15, 23, 42, 255))
    draw.rectangle([784, 154, 1856, 184], fill=(15, 23, 42, 255))
    draw.rectangle([784, 178, 1856, 184], fill=(56, 189, 248, 255))

    font_slide_title = _load_font(36, bold=True)
    heading = (slide_title or lesson_title or f"Key Concept #{block_idx + 1}").strip()
    draw.text((820, 114), heading[:44], font=font_slide_title, fill=(255, 255, 255, 255))

    notes_max_w = 990
    notes_start_y = 212
    if cf_image_path and os.path.exists(cf_image_path):
        try:
            cf_img = Image.open(cf_image_path).convert("RGB")
            tw, th = 992, 350
            iw, ih = cf_img.size
            sc = max(tw / float(iw), th / float(ih))
            cf_img = cf_img.resize((int(iw * sc), int(ih * sc)), Image.Resampling.LANCZOS)
            lx = (cf_img.size[0] - tw) // 2
            ly = (cf_img.size[1] - th) // 2
            cf_img = cf_img.crop((lx, ly, lx + tw, ly + th))
            mask = Image.new("L", (tw, th), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw, th], radius=18, fill=255)
            bg_pil.paste(cf_img, (824, 202), mask)
            notes_start_y = 574
        except Exception as e:
            print(f"Studio CF image embed note: {e}")

    font_bullet = _load_font(31, bold=True)
    curr_y = notes_start_y
    bullet_colors = [(37, 99, 235), (16, 185, 129), (245, 158, 11), (139, 92, 246)]

    for idx, note in enumerate((slide_notes or [])[:4]):
        clean_note = str(note).strip()
        if not clean_note:
            continue
        lines = _wrap_text(draw, clean_note, font_bullet, notes_max_w - 65)
        b_col = bullet_colors[idx % len(bullet_colors)]
        draw.rounded_rectangle([824, curr_y + 4, 858, curr_y + 38], radius=10, fill=b_col + (255,))
        for line in lines[:2]:
            draw.text((874, curr_y), line, font=font_bullet, fill=(15, 23, 42, 255))
            curr_y += 42
        curr_y += 14
        if curr_y > 868:
            break

    # Left Presenter Frame glowing border
    draw.rounded_rectangle([58, 82, 750, 914], radius=26, fill=(15, 23, 42, 190), outline=(56, 189, 248, 235), width=4)

    return cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)


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

    # Rich layered geometric diagram
    draw.ellipse([W - 480, -140, W + 200, 540], fill=c2 + (165,))
    draw.ellipse([-180, H - 420, 460, H + 220], fill=accent + (80,))

    # Draw visual node network / infographic bars on right half
    cx, cy = 920, 360
    nodes = [(cx, cy - 160), (cx + 170, cy - 40), (cx + 110, cy + 150), (cx - 110, cy + 150), (cx - 170, cy - 40)]
    for nx, ny in nodes:
        draw.line([cx, cy, nx, ny], fill=accent + (210,), width=5)
        draw.ellipse([nx - 36, ny - 36, nx + 36, ny + 36], fill=c2 + (240,), outline=gold + (255,), width=4)
    draw.ellipse([cx - 58, cy - 58, cx + 58, cy + 58], fill=gold + (255,), outline=(255, 255, 255, 255), width=5)

    # Left visual concept card
    draw.rounded_rectangle([64, 64, 680, H - 64], radius=28, fill=(15, 23, 42, 225), outline=accent + (220,), width=3)
    font_tag = _load_font(22, bold=True)
    font_title = _load_font(42, bold=True)
    font_body = _load_font(27, bold=True)

    draw.text((104, 108), "VISUAL CONCEPT DIAGRAM", font=font_tag, fill=gold + (255,))
    t_lines = _wrap_text(draw, title_text or "Core Lesson Concept", font_title, 530)
    y = 160
    for line in t_lines[:3]:
        draw.text((104, y), line, font=font_title, fill=(255, 255, 255, 255))
        y += 54

    draw.line([104, y + 12, 620, y + 12], fill=accent + (190,), width=3)
    y += 38
    p_lines = _wrap_text(draw, prompt_text or "", font_body, 530)
    for line in p_lines[:5]:
        draw.text((104, y), line, font=font_body, fill=(226, 232, 240, 255))
        y += 38

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
    reveal_all=True,
    theme_variant=0
):
    """
    Renders a dynamic, high-contrast 1920x1080 PowerPoint Presentation Slide with:
      - Enhanced bold typography (Ubuntu/Liberation/DejaVu Bold)
      - Alternating dark/light executive slide themes so slides never look repetitive
      - Left Column: Numbered gradient cards with summarized bullet notes
      - Right Column: Large 760x700 Cloudflare Workers AI / Wikimedia illustration
    """
    W, H = 1920, 1080
    dark_mode = (theme_variant % 2 == 1)
    bg_col = (15, 23, 42) if dark_mode else (241, 245, 249)
    card_col = (30, 41, 59, 255) if dark_mode else (255, 255, 255, 255)
    text_col = (248, 250, 252, 255) if dark_mode else (15, 23, 42, 255)
    border_col = (56, 189, 248, 190) if dark_mode else (203, 213, 225, 255)

    slide = Image.new("RGB", (W, H), bg_col)
    draw = ImageDraw.Draw(slide, "RGBA")

    # Top PowerPoint Header Banner (y=0..146)
    draw.rectangle([0, 0, W, 146], fill=(12, 18, 34, 255))
    draw.rectangle([0, 138, W, 146], fill=(56, 189, 248, 255))

    font_course = _load_font(22, bold=True)
    font_heading = _load_font(44, bold=True)
    draw.text((72, 22), f"{lesson_title.upper()[:65]}   •   SLIDE {block_idx + 1} OF {total_blocks}", font=font_course, fill=(250, 204, 21, 255))
    draw.text((72, 60), (slide_title or lesson_title)[:48], font=font_heading, fill=(255, 255, 255, 255))

    # Left Card: Summarized Bullet Points (x=64..1036, y=176..920)
    draw.rounded_rectangle([64, 176, 1036, 920], radius=26, fill=card_col, outline=border_col, width=3)
    font_sub = _load_font(25, bold=True)
    draw.text((104, 206), "KEY SUMMARY NOTES & TAKEAWAYS", font=font_sub, fill=(56, 189, 248, 255) if dark_mode else (37, 99, 235, 255))
    draw.line([104, 246, 996, 246], fill=border_col, width=2)

    notes_list = [str(n).strip() for n in (slide_notes or []) if str(n).strip()]
    if not reveal_all and len(notes_list) > 1:
        notes_list = notes_list[:max(1, (len(notes_list) + 1) // 2)]

    font_bullet = _load_font(35, bold=True)
    font_num = _load_font(26, bold=True)
    curr_y = 274
    badge_colors = [(37, 99, 235), (16, 185, 129), (245, 158, 11), (139, 92, 246)]

    for i, clean in enumerate(notes_list[:4]):
        b_col = badge_colors[i % len(badge_colors)]
        draw.rounded_rectangle([104, curr_y + 4, 154, curr_y + 52], radius=14, fill=b_col + (255,))
        draw.text((120, curr_y + 11), str(i + 1), font=font_num, fill=(255, 255, 255, 255))

        wrapped = _wrap_text(draw, clean, font_bullet, 800)
        for line in wrapped[:3]:
            draw.text((174, curr_y + 2), line, font=font_bullet, fill=text_col)
            curr_y += 46
        curr_y += 28
        if curr_y > 868:
            break

    # Right Card: High-Res Visual Illustration (x=1064..1856, y=176..920)
    draw.rounded_rectangle([1064, 176, 1856, 920], radius=26, fill=(15, 23, 42, 255), outline=(56, 189, 248, 210), width=3)
    if cf_image_path and os.path.exists(cf_image_path):
        try:
            ill = Image.open(cf_image_path).convert("RGB")
            target_w, target_h = 760, 712
            iw, ih = ill.size
            scale = max(target_w / float(iw), target_h / float(ih))
            nw, nh = int(iw * scale), int(ih * scale)
            ill = ill.resize((nw, nh), Image.Resampling.LANCZOS)
            left = (nw - target_w) // 2
            top = (nh - target_h) // 2
            ill = ill.crop((left, top, left + target_w, top + target_h))

            mask = Image.new("L", (target_w, target_h), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, target_w, target_h], radius=20, fill=255)
            slide.paste(ill, (1080, 192), mask)
        except Exception as e:
            print(f"Slide image paste note: {e}")

    slide.save(output_png, "PNG")
    return output_png


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


def _build_broll_overlay_rgba(lesson_title, slide_title, note_text, block_idx, total_blocks):
    W, H = 1920, 1080
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov, "RGBA")

    # Top floating studio pill header
    draw.rounded_rectangle([64, 24, 1856, 112], radius=22, fill=(12, 18, 34, 210), outline=(56, 189, 248, 200), width=2)
    font_tag = _load_font(20, bold=True)
    font_head = _load_font(34, bold=True)
    draw.text((96, 36), f"LIVE VISUAL CLIP   •   SECTION {block_idx + 1} OF {total_blocks}", font=font_tag, fill=(250, 204, 21, 255))
    draw.text((96, 64), (slide_title or lesson_title)[:52], font=font_head, fill=(255, 255, 255, 255))

    # Floating Key Note Callout Card on lower-left
    if note_text:
        draw.rounded_rectangle([64, 735, 1220, 910], radius=22, fill=(12, 18, 34, 225), outline=(56, 189, 248, 210), width=3)
        draw.rounded_rectangle([64, 735, 84, 910], radius=8, fill=(250, 204, 21, 255))
        font_note = _load_font(32, bold=True)
        lines = _wrap_text(draw, note_text, font_note, 1080)
        y = 762
        for line in lines[:3]:
            draw.text((108, y), line, font=font_note, fill=(248, 250, 252, 255))
            y += 44

    ov_np = np.array(ov)
    rgb = ov_np[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    alpha = (ov_np[:, :, 3].astype(np.float32) / 255.0)[..., None]
    return bgr, alpha


def _overlay_pip_presenter(canvas_bgr, presenter_frame_bgr, pos="bottom_right"):
    """
    Overlays a live Lip-Synced Teacher Picture-in-Picture (PiP) rounded card
    onto PowerPoint Slides and B-Roll Clips so the teacher is visibly speaking
    even during visual cutaway scenes.
    """
    pip_w, pip_h = 240, 290
    pip_img = cv2.resize(presenter_frame_bgr, (pip_w, pip_h), interpolation=cv2.INTER_AREA)

    if pos == "top_right":
        x0, y0 = 1920 - pip_w - 48, 132
    else:
        x0, y0 = 1920 - pip_w - 48, 1080 - pip_h - 148

    # Draw glowing cyan border behind PiP card
    cv2.rectangle(canvas_bgr, (x0 - 3, y0 - 3), (x0 + pip_w + 3, y0 + pip_h + 3), (248, 189, 56), 3, cv2.LINE_AA)
    canvas_bgr[y0:y0 + pip_h, x0:x0 + pip_w] = pip_img
    return canvas_bgr


def _render_bokeh_particles(canvas_bgr, f_idx, fps=24):
    """
    Adds subtle moving cinema bokeh light particles to B-roll video clips so they
    look like dynamic motion footage rather than static images.
    """
    t = f_idx / float(fps)
    overlay = canvas_bgr.copy()
    particles = [
        (320, 280, 42, 0.35, 0.25, (248, 210, 120)),
        (860, 480, 58, -0.28, 0.30, (255, 230, 160)),
        (1420, 340, 36, 0.40, -0.22, (248, 189, 56)),
        (1620, 620, 52, -0.32, -0.28, (200, 240, 255)),
        (580, 680, 46, 0.25, -0.35, (255, 220, 140)),
    ]
    for idx, (bx, by, rad, vx, vy, col) in enumerate(particles):
        px = int((bx + math.sin(t * vx * 2.0 + idx) * 140) % 1920)
        py = int((by + math.cos(t * vy * 2.0 + idx) * 90) % 1080)
        cv2.circle(overlay, (px, py), rad, col, -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.12, canvas_bgr, 0.88, 0.0)


def render_dynamic_lesson_block(
    raw_avatar_path,
    audio_wav_path,
    output_mp4_path,
    lesson_title,
    slide_title,
    slide_notes,
    cf_images,
    block_idx=0,
    total_blocks=1,
    block_start_sec=0.0,
    total_video_duration_sec=60.0,
    visual_blueprint="standard_30s",
    fps=24,
    device="cuda",
    run_ffmpeg_fn=None
):
    """
    Rescales the 10-Minute Video Blueprint proportionally to ANY video duration (60s, 90s, 180s, 600s)
    and dynamically switches camera/visual modes every 6-8 seconds inside EVERY block:
      - Scaled Intro Window [0 .. T_intro] & Scaled Outro Window [D - T_outro .. D]:
        Full Lip-Synced Teacher Studio Shot (with right-side PowerPoint card & Cloudflare image)
      - Middle Timeline (rescaled across every block):
        Interchanges:
          1. PowerPoint Slide Deck (Part 1) + Cloudflare Image #1 + Live Lip-Sync Teacher PiP
          2. 30s-scaled B-Roll Video Motion Clip (Ken Burns 3D pan/zoom + cinema bokeh + Cloudflare Image #2 + PiP)
          3. PowerPoint Slide Deck (Part 2 - Full Summary) + Cloudflare Image #3 + Live Lip-Sync Teacher PiP
          4. Lip-Synced Teacher Studio Check-in
    """
    img1 = cf_images[0] if len(cf_images) > 0 else None
    img2 = cf_images[1] if len(cf_images) > 1 else img1
    img3 = cf_images[2] if len(cf_images) > 2 else img2

    # 1. Pre-compute talking presenter state (S3FD + Wav2Lip-GAN + Syllable Jaw + Eye Blink)
    talk_state = synthesize_talking_presenter_frames(raw_avatar_path, audio_wav_path, fps=fps, device=device)
    num_frames = talk_state["num_frames"]
    block_dur_sec = num_frames / float(fps)

    # 2. Build Studio Composite Background, PowerPoint Slide 1 & 2, and B-Roll Canvases
    presenter_w, presenter_h = 680, 820
    studio_bg_bgr = build_studio_Composite_background(
        raw_avatar_path=raw_avatar_path,
        lesson_title=lesson_title,
        slide_title=slide_title,
        slide_notes=slide_notes,
        cf_image_path=img1,
        block_idx=block_idx,
        total_blocks=total_blocks
    )
    rounded_card_mask = _create_rounded_mask(presenter_w, presenter_h, radius=22)
    card_alpha = (rounded_card_mask.astype(np.float32) / 255.0)[..., None]
    base_slot = studio_bg_bgr[88:88 + presenter_h, 64:64 + presenter_w].astype(np.float32)

    p1_png = f"{output_mp4_path}.s1.png"
    p2_png = f"{output_mp4_path}.s2.png"
    render_powerpoint_slide_frame(
        lesson_title=lesson_title,
        slide_title=slide_title,
        slide_notes=slide_notes,
        cf_image_path=img1,
        raw_avatar_path=raw_avatar_path,
        output_png=p1_png,
        block_idx=block_idx,
        total_blocks=total_blocks,
        reveal_all=False,
        theme_variant=block_idx
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
        theme_variant=block_idx + 1
    )
    slide1_bgr = cv2.imread(p1_png)
    slide2_bgr = cv2.imread(p2_png)

    broll_canvas_1 = _prepare_broll_canvas(img2, 1920, 1080)
    broll_canvas_2 = _prepare_broll_canvas(img3, 1920, 1080)
    notes_list = [str(n).strip() for n in (slide_notes or []) if str(n).strip()]
    broll_note_1 = notes_list[0] if len(notes_list) > 0 else slide_title
    broll_note_2 = notes_list[-1] if len(notes_list) > 1 else broll_note_1
    broll_ov1_bgr, broll_ov1_a = _build_broll_overlay_rgba(lesson_title, slide_title, broll_note_1, block_idx, total_blocks)
    broll_ov2_bgr, broll_ov2_a = _build_broll_overlay_rgba(lesson_title, slide_title, broll_note_2, block_idx, total_blocks)

    # 3. Rescale the 10-minute (600s) blueprint to the actual total video duration D
    D = max(block_dur_sec, float(total_video_duration_sec))
    # Intro & Outro Lip-Sync Teacher windows: 15% of total video each (min 8s, max 30s)
    t_intro_window = min(30.0, max(8.0, D * 0.15))
    t_outro_window = min(30.0, max(8.0, D * 0.15))

    def _get_mode_for_frame(f_idx):
        t_local = f_idx / float(fps)
        t_global = block_start_sec + t_local
        frac_local = t_local / max(1.0, block_dur_sec)

        # Always show Full Studio Lip-Sync Teacher during the rescaled Intro and Outro windows
        if t_global < t_intro_window or t_global > (D - t_outro_window):
            # Even in a long 30s intro/outro, briefly flash the full-screen slide for 5s around 65% so it stays dynamic
            if block_dur_sec >= 22.0 and 0.62 <= frac_local <= 0.80:
                return "slide_1"
            return "studio_lipsync"

        if visual_blueprint == "90s_avatar":
            # Extended Teacher Presence: 65% Studio Lip-Sync, 20% B-Roll Clip, 15% Full Slide
            if frac_local < 0.45:
                return "studio_lipsync"
            elif frac_local < 0.72:
                return "broll_1"
            elif frac_local < 0.88:
                return "slide_2"
            else:
                return "studio_lipsync"

        # Default Cost-Saving Dynamic Blueprint (rescaled for any duration):
        # Every block dynamically cycles through Slide 1 -> B-Roll Motion Clip 1 -> Slide 2 -> B-Roll Motion Clip 2 / Studio Check-in
        if frac_local < 0.28:
            return "slide_1"
        elif frac_local < 0.56:
            return "broll_1"
        elif frac_local < 0.82:
            return "slide_2"
        else:
            return "broll_2" if (block_idx % 2 == 0) else "studio_lipsync"

    def _render_single_mode_frame(mode_name, f_idx):
        pres_bgr = get_animated_presenter_frame(talk_state, f_idx)
        prog = (f_idx % max(1, num_frames // 3)) / float(max(1, num_frames // 3))
        W, H = 1920, 1080

        if mode_name == "studio_lipsync":
            full_frame = studio_bg_bgr.copy()
            comp_slot = (pres_bgr.astype(np.float32) * card_alpha + base_slot * (1.0 - card_alpha)).astype(np.uint8)
            full_frame[88:88 + presenter_h, 64:64 + presenter_w] = comp_slot
            return full_frame

        elif mode_name in ("slide_1", "slide_2"):
            base = (slide1_bgr if mode_name == "slide_1" else slide2_bgr).copy()
            # Dynamic zoom on the right-hand illustration card (x=1080..1840, y=192..904)
            ix1, iy1, ix2, iy2 = 1080, 192, 1840, 904
            iw, ih = ix2 - ix1, iy2 - iy1
            zoom = 1.0 + 0.06 * prog
            sub_img = base[iy1:iy2, ix1:ix2]
            M = np.float32([
                [zoom, 0.0, (1.0 - zoom) * (iw * 0.5)],
                [0.0, zoom, (1.0 - zoom) * (ih * 0.5)]
            ])
            base[iy1:iy2, ix1:ix2] = cv2.warpAffine(sub_img, M, (iw, ih), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            # Overlay live Lip-Synced Teacher PiP in top-right corner
            base = _overlay_pip_presenter(base, pres_bgr, pos="top_right")
            return base

        else:
            # 'broll_1' or 'broll_2' - Full-Screen Dynamic B-Roll Motion Clip
            is_second = (mode_name == "broll_2")
            canvas = broll_canvas_2 if is_second else broll_canvas_1
            ov_bgr = broll_ov2_bgr if is_second else broll_ov1_bgr
            ov_a = broll_ov2_a if is_second else broll_ov1_a

            zoom = (1.14 - 0.10 * prog) if is_second else (1.02 + 0.12 * prog)
            tx = (22.0 * (1.0 - prog)) if is_second else (-24.0 * prog)
            ty = (12.0 * (1.0 - prog)) if is_second else (-14.0 * prog)
            M = np.float32([
                [zoom, 0.0, (1.0 - zoom) * (W * 0.5) + tx],
                [0.0, zoom, (1.0 - zoom) * (H * 0.5) + ty]
            ])
            moved = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            moved = _render_bokeh_particles(moved, f_idx, fps=fps)
            comp = (moved.astype(np.float32) * (1.0 - ov_a) + ov_bgr * ov_a).astype(np.uint8)
            comp = _overlay_pip_presenter(comp, pres_bgr, pos="bottom_right")
            return comp

    silent_mp4 = f"{output_mp4_path}.silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_mp4, fourcc, float(fps), (1920, 1080))

    modes = [_get_mode_for_frame(i) for i in range(num_frames)]
    transition_len = 8

    for f_idx in range(num_frames):
        curr_mode = modes[f_idx]
        frame_bgr = _render_single_mode_frame(curr_mode, f_idx)

        # Smooth crossfade transition when switching scene modes
        if f_idx >= transition_len:
            prev_mode = modes[f_idx - transition_len]
            if prev_mode != curr_mode:
                # Find how many frames since mode changed
                steps_since = 0
                for back in range(1, transition_len + 1):
                    if modes[f_idx - back] != curr_mode:
                        steps_since = back
                        break
                if 0 < steps_since <= transition_len:
                    alpha = steps_since / float(transition_len)
                    prev_bgr = _render_single_mode_frame(prev_mode, f_idx)
                    frame_bgr = cv2.addWeighted(frame_bgr, alpha, prev_bgr, 1.0 - alpha, 0.0)

        writer.write(frame_bgr)

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
