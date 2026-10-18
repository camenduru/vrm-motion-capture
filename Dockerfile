# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# VRM Motion Capture - fully self-contained image.
#
# Baked in (nothing is fetched at runtime):
#   * app/ + the built three.js bundle (compiled here from web/)
#   * rtmlib (vendored, editable install)
#   * models/            RTMPose/YOLOX ONNX + MediaPipe face model  (~0.7 GB)
#   * MuseTalk/          patched v1.5 checkout + its weights        (~4.1 GB)
#   * vrm_models/        avatar .vrm and idle_loop.vrma
#   * videos/            the demo pose clips
#   * two Python envs:   /opt/app (onnxruntime-gpu) and /opt/musetalk (torch)
#
# Build (from the project root, which must contain models/ and MuseTalk/):
#   docker build -t camenduru/vrm-motion-capture .
#
# Run (GPU + the UI on http://localhost:8000):
#   docker run --rm -it --gpus all -p 8000:8000 camenduru/vrm-motion-capture
#
# Push:
#   docker push camenduru/vrm-motion-capture:latest
#
# Expect a ~12-14 GB image; the build context is ~5 GB.
# ---------------------------------------------------------------------------

# ---- frontend: build the three.js bundle from source ----------------------
FROM node:22-alpine AS frontend
WORKDIR /src
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- runtime --------------------------------------------------------------
# CUDA 12.8 + cuDNN 9 matches onnxruntime-gpu 1.26; Ubuntu 24.04 brings Py 3.12.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# MUSETALK_PYTHON is the interpreter app/lipsync.py looks for;
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE hard-guarantee no runtime downloads.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MUSETALK_PYTHON=/opt/musetalk/bin/python \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
        ffmpeg \
        libgl1 libglib2.0-0 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# app code, models, MuseTalk (code + weights), rtmlib, videos, vrm_models
COPY . /app/
# always use the bundle built in the stage above
COPY --from=frontend /src/dist /app/web/dist

# ---- app environment ------------------------------------------------------
# requirements.txt ships the CPU onnxruntime; swap it for the CUDA build that
# matches this base image (CUDA 12.8 / cuDNN 9).
RUN python3.12 -m venv /opt/app \
    && /opt/app/bin/pip install --upgrade pip wheel \
    && /opt/app/bin/pip install -r requirements.txt \
    && /opt/app/bin/pip uninstall -y onnxruntime \
    && /opt/app/bin/pip install "onnxruntime-gpu>=1.26,<1.27"

# ---- MuseTalk environment -------------------------------------------------
# Same interpreter major/minor so one base image serves both; kept separate
# because the two stacks pin different torch/diffusers versions.
RUN python3.12 -m venv /opt/musetalk \
    && /opt/musetalk/bin/pip install --upgrade pip wheel \
    && /opt/musetalk/bin/pip install \
         torch torchvision torchaudio \
         --index-url https://download.pytorch.org/whl/cu126 \
    && /opt/musetalk/bin/pip install \
         diffusers accelerate transformers omegaconf librosa einops \
         soundfile imageio imageio-ffmpeg ffmpeg-python \
         opencv-python-headless mediapipe huggingface_hub

# mediapipe loads EGL/GLES at runtime (the face landmarker); the CUDA runtime
# base image does not ship them. Kept as its own late layer so the heavy
# venv layers above stay cached when this changes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         libegl1 libgles2 libgbm1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---- wiring + baked-in checks --------------------------------------------
RUN printf '%s\n' /opt/musetalk/bin/python > /app/MuseTalk/.musetalk_python \
    && mkdir -p /app/animations /app/.cache/lipsync

RUN test -f /app/models/yolox_x_humanart.onnx \
    && test -f /app/models/yolox_tiny_humanart.onnx \
    && test -f /app/models/rtmpose_x_body7_384x288.onnx \
    && test -f /app/models/rtmpose_s_body7_256x192.onnx \
    && test -f /app/models/face_landmarker.task \
    && test -f /app/MuseTalk/models/musetalkV15/unet.pth \
    && test -f /app/MuseTalk/models/sd-vae/diffusion_pytorch_model.bin \
    && test -f /app/MuseTalk/models/whisper/pytorch_model.bin \
    && test -f /app/vrm_models/AvatarSample_B.vrm \
    && test -f /app/web/dist/index.html \
    && echo "all models and assets baked in"

# import checks: GPU provider visible, app modules load, mediapipe can open
# the baked face model (this is what needs EGL/GLES in the image)
RUN /opt/app/bin/python - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
assert 'CUDAExecutionProvider' in providers, providers
import app.server          # noqa: F401  (fastapi, cv2, rtmlib)
import app.offline         # noqa: F401
from app.face_engine import FaceEngine
face = FaceEngine()
assert face.open(), face.error      # mediapipe + face_landmarker.task
face.close()
print('app env ok, providers:', providers, '| face model ok')
PY

RUN /opt/musetalk/bin/python - <<'PY'
import torch, transformers, diffusers, mediapipe  # noqa: F401
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
opts = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(
        model_asset_path='/app/models/face_landmarker.task'),
    running_mode=vision.RunningMode.IMAGE)
vision.FaceLandmarker.create_from_options(opts).close()
print('musetalk env ok, torch', torch.__version__,
      '| mediapipe landmarker ok')
PY

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/status > /dev/null || exit 1

CMD ["/opt/app/bin/python", "-m", "app.server", "--host", "0.0.0.0", "--port", "8000"]
