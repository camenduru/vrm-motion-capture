"""Optional facial capture: MediaPipe FaceLandmarker on the live frames.

Produces a head rotation (a quaternion in camera space) plus the 52 ARKit-style
blendshape coefficients, which the web layer maps onto VRM expressions. It runs
on the same mirrored frame as the body tracker, so "the person's left" already
lands where the preview shows it and the avatar can use it directly.

Every failure mode (mediapipe not installed, model missing, a decode error) is
reported once and then the layer switches itself off - the app keeps running
with body tracking only.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FACE_MODEL = ROOT / 'models' / 'face_landmarker.task'

# MediaPipe logs a wall of warnings on import; keep the app log readable
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Only coefficients above this are forwarded; a resting face measures noise
# around 0.01 and sending 52 zeros every frame is wasted bandwidth.
SEND_THRESHOLD = 0.005


def _quaternion(matrix) -> list:
    """Rotation part of a 4x4 transform as [x, y, z, w]."""
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return [x / norm, y / norm, z / norm, w / norm]


class FaceEngine:
    """Thin wrapper around MediaPipe's FaceLandmarker task."""

    def __init__(self, model_path: Path = FACE_MODEL):
        self.model_path = Path(model_path)
        self.error: Optional[str] = None
        self.frames = 0
        self.hits = 0
        self._landmarker = None
        self._mp = None
        self._start = time.perf_counter()
        self._last_ts = -1

    # ------------------------------------------------------------------ setup
    def open(self) -> bool:
        """Load the model. Returns False (and records why) instead of raising."""
        if self._landmarker is not None:
            return True
        if not self.model_path.exists():
            self.error = (f'missing {self.model_path.name} - run '
                          'python scripts/download_models.py --set face')
            return False
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception as exc:  # noqa: BLE001
            self.error = f'mediapipe unavailable ({exc})'
            return False

        try:
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(
                options)
            self._mp = mp
        except Exception as exc:  # noqa: BLE001
            self.error = f'face model failed to load ({exc})'
            return False

        self.error = None
        print('[face] mediapipe face landmarker ready', flush=True)
        return True

    # --------------------------------------------------------------- per frame
    def process(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """Return {'q': [...], 'blend': {...}} for the frame, or None."""
        if self._landmarker is None and not self.open():
            return None

        # VIDEO mode wants strictly increasing timestamps and the source may
        # loop, so the clock is wall time rather than the frame index
        ts = int((time.perf_counter() - self._start) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        self.frames += 1
        if not result.face_landmarks:
            return None
        self.hits += 1

        out: dict = {}
        if result.facial_transformation_matrixes:
            out['q'] = _quaternion(result.facial_transformation_matrixes[0])
        if result.face_blendshapes:
            scores = {
                category.category_name: round(float(category.score), 3)
                for category in result.face_blendshapes[0]
                if category.score >= SEND_THRESHOLD
            }
            out['blend'] = scores
        return out

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # noqa: BLE001
                pass
            self._landmarker = None
