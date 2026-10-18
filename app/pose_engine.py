"""Webcam capture + RTMPose inference running in a background thread.

The engine owns the video source and the rtmlib pose tracker. Every frame it
infers COCO-17 keypoints, draws a skeleton preview and exposes the latest
normalized pose over a thread-safe snapshot that the web server / websocket
serves to the browser.

Sources:
    camera:0        default webcam (backend auto-detected)
    demo            synthetic skeleton animation, no camera or model needed
    videos/foo.mp4  any clip in ./videos (looped) - the "demo live pose" set
    path/to/file    image (looped) or video (looped)
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / 'models'
VIDEOS_DIR = ROOT / 'videos'
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXT = {'.mp4', '.webm', '.mov', '.mkv', '.avi', '.m4v'}

# The rtmlib checkout lives in ./rtmlib next to this app. The project root also
# contains a directory with that name, which Python would otherwise resolve as
# an empty namespace package when running `python -m app.server` from the root.
# Put the actual package parent first so the real package always wins.
_RTMLIB_SRC = ROOT / 'rtmlib'
if (_RTMLIB_SRC / 'rtmlib' / '__init__.py').exists():
    sys.path.insert(0, str(_RTMLIB_SRC))

COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
    'right_knee', 'left_ankle', 'right_ankle',
]

# COCO left/right keypoint pairs. Every frame is mirrored (cv2.flip) *before*
# inference so the preview reads like a selfie, but RTMPose labels by what it
# sees: mirroring a frame moves the person's left side to the right of the
# image, and the model then calls it "right". Its left/right labels therefore
# land on the opposite side of the real body, which is what made the avatar's
# arms cross (move your left arm, the avatar moves the other one). Swap the
# pairs back so the indices describe the real person again - the coordinates
# stay in the mirrored frame, which is what the web retargeter and the
# synthetic demo both expect.
LR_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8),
            (9, 10), (11, 12), (13, 14), (15, 16))


def _unmirror_labels(keypoints, scores):
    """Undo the left/right swap a mirrored frame imposes on the model."""
    order = np.arange(len(keypoints))
    for a, b in LR_PAIRS:
        if b < len(order):
            order[a], order[b] = order[b], order[a]
    return keypoints[order], scores[order]


# Detector / pose estimator per quality preset. Both sets are downloaded into
# models/ by scripts/download_models.py.
MODEL_SETS = {
    'performance': {
        'det': ('yolox_x_humanart.onnx', (640, 640)),
        'pose': ('rtmpose_x_body7_384x288.onnx', (288, 384)),
    },
    'balanced': {
        'det': ('yolox_m_humanart.onnx', (640, 640)),
        'pose': ('rtmpose_m_body7_256x192.onnx', (192, 256)),
    },
    'lightweight': {
        'det': ('yolox_tiny_humanart.onnx', (416, 416)),
        'pose': ('rtmpose_s_body7_256x192.onnx', (192, 256)),
    },
}

# `mode='auto'` picks a model pair the chosen device can drive in real time:
# the most accurate pair on a GPU, the small one on CPU.
AUTO_MODES = {'cuda': 'performance', 'cpu': 'lightweight'}

PREVIEW_WIDTH = 640
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720


def _plan_mode(mode: str, device: str) -> str:
    """Resolve 'auto' to the model size that suits this device."""
    if mode != 'auto':
        return mode
    return AUTO_MODES['cuda' if 'cuda' in device else 'cpu']


def _resolve_device(device: str) -> str:
    """Turn 'auto' into a concrete onnxruntime device id."""
    requested = (device or 'auto').strip().lower()
    if requested != 'auto':
        return requested
    try:
        import onnxruntime as ort
    except ImportError:
        return 'cpu'
    return ('cuda' if 'CUDAExecutionProvider' in ort.get_available_providers()
            else 'cpu')


# Handles from os.add_dll_directory must stay open for the process to keep
# using those directories, so they are parked here for the whole run.
_DLL_DIR_HANDLES: list = []


def _add_nvidia_dll_dirs() -> int:
    """Expose the pip-installed CUDA/cuDNN DLLs before onnxruntime loads them.

    onnxruntime-gpu bundles no CUDA runtime, and cuDNN 9 is split across
    several DLLs (cudnn_graph64_9, cudnn_engines_tensor_ir64_9, ...) that
    Windows resolves from the ordinary search path. The ``nvidia/*/bin``
    directories therefore have to be on it *before* the CUDA provider starts:
    otherwise the first Conv fails inside the cuDNN frontend and onnxruntime
    silently rebuilds the session on the CPU.
    """
    dirs: list = []
    for entry in sys.path:
        root = Path(entry or '.') / 'nvidia'
        if not root.is_dir():
            continue
        for package in sorted(root.iterdir()):
            bin_dir = package / 'bin'
            if bin_dir.is_dir() and bin_dir not in dirs:
                dirs.append(bin_dir)

    for bin_dir in dirs:
        path = os.environ.get('PATH', '')
        if str(bin_dir) not in path.split(os.pathsep):
            os.environ['PATH'] = str(bin_dir) + os.pathsep + path
        if hasattr(os, 'add_dll_directory'):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(bin_dir)))
            except OSError as exc:
                print(f'[pose] cannot expose {bin_dir}: {exc}', flush=True)
    return len(dirs)


def _preload_cuda_dlls() -> None:
    """Make the GPU runtime loadable, then let onnxruntime preload it."""
    count = _add_nvidia_dll_dirs()
    try:
        import onnxruntime as ort
    except ImportError:
        return
    preload = getattr(ort, 'preload_dlls', None)
    if preload is None:
        return
    try:
        preload()
    except Exception as exc:  # noqa: BLE001
        print(f'[pose] CUDA dll preload skipped: {exc}', flush=True)
    else:
        if count:
            print(f'[pose] CUDA runtime: {count} nvidia dirs on the DLL path',
                  flush=True)


def _session_providers(tracker) -> list:
    """Providers the tracker's sessions actually run on (ORT may fall back)."""
    for attr in ('pose_model', 'det_model'):
        session = getattr(getattr(tracker, attr, None), 'session', None)
        if session is not None:
            try:
                return [str(p) for p in session.get_providers()]
            except Exception:  # noqa: BLE001
                pass
    return []


def _open_camera(index: int):
    """Open a webcam, preferring the DirectShow backend on Windows."""
    backends = [cv2.CAP_ANY]
    if hasattr(cv2, 'CAP_DSHOW'):
        backends.insert(0, cv2.CAP_DSHOW)

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = False
        for _ in range(5):  # warm up; some drivers return empty first frames
            ok, _frame = cap.read()
            if ok:
                break
        if ok:
            print(f'[camera] opened index={index} backend={backend}', flush=True)
            return cap
        cap.release()
    return None


class DemoSource:
    """Synthetic COCO-17 animation used to preview the app without a camera."""

    def __init__(self):
        self.width = CAPTURE_WIDTH
        self.height = CAPTURE_HEIGHT
        self.t0 = time.perf_counter()

    def read(self):
        t = time.perf_counter() - self.t0

        sway = 0.012 * math.sin(t * 0.7)
        breathe = 0.006 * math.sin(t * 1.6)
        cx = 0.5 + sway
        sh_y = 0.36 + breathe
        hip_y = 0.60 + breathe * 0.5
        sh_half = 0.082
        hip_half = 0.046

        def arm(shoulder_x, phase, waving):
            sx = shoulder_x
            sy = sh_y
            if waving:
                ex = sx + 0.115 + 0.012 * math.sin(phase)
                ey = sy + 0.035 + 0.02 * math.cos(phase * 0.7)
                wx = sx + 0.155 + 0.03 * math.sin(phase * 1.4)
                wy = sy - 0.075 + 0.045 * math.sin(phase * 1.4)
            else:
                ex = sx - 0.006 + 0.012 * math.sin(phase)
                ey = sy + 0.105
                wx = sx - 0.012 + 0.02 * math.sin(phase + 0.4)
                wy = sy + 0.205
            return (ex, ey), (wx, wy)

        r_elbow, r_wrist = arm(cx + sh_half, t * 2.4, True)
        l_elbow, l_wrist = arm(cx - sh_half, t * 1.6, False)

        kp = np.zeros((17, 2), dtype=np.float32)
        # face
        head_x = cx + 0.006 * math.sin(t * 0.9)
        kp[0] = (head_x, 0.263)
        kp[1] = (head_x - 0.016, 0.247)
        kp[2] = (head_x + 0.016, 0.247)
        kp[3] = (head_x - 0.036, 0.253)
        kp[4] = (head_x + 0.036, 0.253)
        # arms, mirrored so the avatar (facing us) moves like a mirror
        kp[5] = (cx - sh_half, sh_y)  # person left = screen left
        kp[6] = (cx + sh_half, sh_y)  # person right = screen right
        kp[7], kp[9] = l_elbow, l_wrist
        kp[8], kp[10] = r_elbow, r_wrist
        # torso + legs
        kp[11] = (cx - hip_half, hip_y)
        kp[12] = (cx + hip_half, hip_y)
        kp[13] = (cx - hip_half - 0.006, hip_y + 0.135)
        kp[14] = (cx + hip_half + 0.006, hip_y + 0.135)
        kp[15] = (cx - hip_half - 0.01, 0.895)
        kp[16] = (cx + hip_half + 0.01, 0.895)
        scores = np.full(17, 0.95, dtype=np.float32)
        return kp, scores


class PoseEngine:

    def __init__(self,
                 source: str = 'camera:0',
                 mode: str = 'auto',
                 device: str = 'auto',
                 det_frequency: int = 6,
                 face: bool = True,
                 face_every: int = 2):
        if mode != 'auto' and mode not in MODEL_SETS:
            raise ValueError(f'Unknown mode {mode!r}. '
                             f'Choose from {sorted(MODEL_SETS)} or auto')
        self.mode = mode
        self.device = device
        # `mode`/`device` are the requested preferences; these are what the
        # models actually ended up on, which is what the UI reports
        self._active_mode = mode
        self._active_device = device
        self.det_frequency = max(1, det_frequency)
        self.face_enabled = face
        self.face_every = max(1, face_every)
        self._face_engine = None
        self._face_tick = 0
        self._face_cache: Optional[dict] = None
        self.source = source
        self.is_demo = source == 'demo'

        # state shared with the web layer
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._frame_seq = 0
        self._pose: dict = {
            'seq': 0,
            't': 0.0,
            'fps': 0.0,
            'detected': False,
            'keypoints': [],
            'scores': [],
            'width': 0,
            'height': 0,
        }
        self._status = {
            'state': 'starting',
            'error': None,
            'source': source,
            'mode': mode,
            'device': device,
            'device_pref': device,
            'gpu_available': _resolve_device('auto') == 'cuda',
            'face': False,
            'width': 0,
            'height': 0,
            'fps': 0.0,
            'detected': False,
        }

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._tracker = None
        self._demo = None
        self._cap = None
        self._image = None
        self._pending_source: Optional[str] = None
        self._pending_device: Optional[str] = None
        self._frame_interval = 0.0
        self._frame_due = 0.0
        self._source_ready = False

    # ------------------------------------------------------------------ setup
    def _build_tracker(self):
        from functools import partial

        from rtmlib import Body, PoseTracker

        device = _resolve_device(self.device)
        if 'cuda' in device:
            _preload_cuda_dlls()

        # (device, mode) pairs to try, best first; CPU always works
        attempts = [(device, _plan_mode(self.mode, device))]
        if device != 'cpu':
            attempts.append(('cpu', _plan_mode(self.mode, 'cpu')))

        last_exc: Optional[Exception] = None
        for attempt_device, mode in attempts:
            det_file, det_size = MODEL_SETS[mode]['det']
            pose_file, pose_size = MODEL_SETS[mode]['pose']
            det_path = MODELS_DIR / det_file
            pose_path = MODELS_DIR / pose_file

            missing = [str(p) for p in (det_path, pose_path)
                       if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    'Missing pose models: ' + ', '.join(missing) +
                    '. Run: python scripts/download_models.py')

            print(f'[pose] loading {mode} models on {attempt_device}...',
                  flush=True)
            solution = partial(
                Body,
                det=str(det_path),
                det_input_size=det_size,
                pose=str(pose_path),
                pose_input_size=pose_size,
            )
            try:
                tracker = PoseTracker(
                    solution,
                    det_frequency=self.det_frequency,
                    mode=mode,
                    to_openpose=False,
                    backend='onnxruntime',
                    device=attempt_device,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt_device == 'cpu':
                    raise
                print(f'[pose] {attempt_device} unavailable ({exc}); '
                      'falling back to cpu', flush=True)
                continue

            providers = _session_providers(tracker)
            if attempt_device != 'cpu' \
                    and not any('CUDA' in p for p in providers):
                # onnxruntime silently falls back to CPU if the GPU runtime
                # cannot load, so trust the session, not the request
                print(f'[pose] {attempt_device} started on {providers}; '
                      'falling back to cpu', flush=True)
                continue

            if attempt_device != 'cpu':
                # a half-broken GPU runtime can also fail on the first
                # inference instead of at session creation
                probe = np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3),
                                 dtype=np.uint8)
                try:
                    tracker(probe)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    print(f'[pose] {attempt_device} failed its first '
                          'inference; falling back to cpu', flush=True)
                    continue
                # a blank frame leaves the tracker's IoU state inconsistent:
                # frame_cnt stops advancing, which freezes det_frequency and
                # quietly disables person detection for the whole session
                tracker.reset()

            self._active_device = attempt_device
            self._active_mode = mode
            with self._lock:
                self._status.update(device=attempt_device, mode=mode,
                                    device_pref=self.device)
            print(f'[pose] ready: {mode} on {attempt_device} '
                  f'({", ".join(providers) or "unknown provider"})',
                  flush=True)
            return tracker

        raise RuntimeError('No usable pose execution provider') from last_exc

    # --------------------------------------------------------------- sources
    @staticmethod
    def resolve_source(source: str) -> Optional[Path]:
        """Map a source id to a file path inside the project, if it is one."""
        if source in ('demo', '') or source.startswith('camera'):
            return None
        path = (ROOT / source).resolve()
        if ROOT not in path.parents or not path.exists():
            return None
        return path

    def available_sources(self) -> list:
        items = [
            {'id': 'camera:0', 'label': 'Webcam 0', 'kind': 'camera'},
            {'id': 'camera:1', 'label': 'Webcam 1', 'kind': 'camera'},
            {'id': 'camera:2', 'label': 'Webcam 2', 'kind': 'camera'},
            {'id': 'demo', 'label': 'Synthetic demo', 'kind': 'demo'},
        ]
        if VIDEOS_DIR.exists():
            for path in sorted(VIDEOS_DIR.iterdir()):
                if path.suffix.lower() in VIDEO_EXT:
                    items.append({
                        'id': f'videos/{path.name}',
                        'label': path.stem.replace('_', ' '),
                        'kind': 'video',
                    })
        if not any(item['id'] == self.source for item in items):
            items.insert(0, {'id': self.source, 'label': self.source,
                             'kind': 'custom'})
        return items

    def set_source(self, source: str) -> None:
        """Queue a different camera / demo clip / video for the engine."""
        if source != 'demo' and not source.startswith('camera') \
                and self.resolve_source(source) is None:
            raise ValueError(f'Unknown source: {source}')
        with self._lock:
            self._pending_source = source
            self._status.update(state='switching', error=None)

    def set_device(self, device: str) -> None:
        """Queue a different compute device: auto | cpu | cuda | mps.

        The pose models are rebuilt on the new device on the next frame while
        the source keeps running. With ``mode='auto'`` this also swaps the
        preset, so cuda gets the accurate models and cpu the light ones.
        """
        preference = (device or 'auto').strip().lower()
        if preference not in ('auto', 'cpu', 'cuda', 'mps') \
                and not preference.startswith('cuda:'):
            raise ValueError(f'Unknown device: {device}')
        with self._lock:
            if preference == self._status.get('device_pref'):
                return
            self._pending_device = preference
            self._status.update(state='switching', error=None)

    def _close_source(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._image = None
        self._demo = None
        self._frame_interval = 0.0
        self._frame_due = 0.0
        self._face_cache = None
        self._face_tick = 0
        if self._tracker is not None and hasattr(self._tracker, 'reset'):
            self._tracker.reset()

    def _prepare_source(self):
        """Open the current source and record its playback cadence."""
        self._close_source()
        self.is_demo = self.source == 'demo'

        if self.is_demo:
            self._demo = DemoSource()
            self._frame_interval = 1.0 / 30.0
            return

        if self.source.startswith('camera'):
            try:
                index = int(self.source.split(':', 1)[1])
            except (IndexError, ValueError):
                index = 0
            self._cap = _open_camera(index)
            if self._cap is None:
                raise RuntimeError(
                    f'Could not open camera index {index}. Check that a '
                    'webcam is connected, has camera permission and is not '
                    'used by another app.')
            return

        path = self.resolve_source(self.source)
        if path is None:
            raise FileNotFoundError(f'Source not found: {self.source}')
        if path.suffix.lower() in IMAGE_EXT:
            self._image = cv2.imread(str(path))
            if self._image is None:
                raise RuntimeError(f'Could not read image: {path}')
            self._frame_interval = 1.0 / 30.0
            return

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f'Could not open video: {path}')
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_interval = 1.0 / fps if 1 <= fps <= 120 else 1.0 / 30.0
        print(f'[source] {path.name} @ {fps:.1f} fps', flush=True)

    # ------------------------------------------------------------------ loop
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name='pose-engine')
        self._thread.start()

    def _ensure_tracker(self):
        if self._tracker is None:
            self._set_status('loading', None)
            self._tracker = self._build_tracker()
        return self._tracker

    def _ensure_face(self):
        """Start the facial layer once; disable it if it cannot start."""
        if not self.face_enabled or self._face_engine is not None:
            return
        from .face_engine import FaceEngine

        print('[face] loading mediapipe face landmarker...', flush=True)
        engine = FaceEngine()
        if engine.open():
            self._face_engine = engine
            with self._lock:
                self._status['face'] = True
        else:
            print(f'[face] off: {engine.error}', flush=True)
            self.face_enabled = False

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap is not None:
            self._cap.release()
        if self._face_engine is not None:
            self._face_engine.close()
            self._face_engine = None

    def _set_status(self, state: str, error: Optional[str]):
        with self._lock:
            self._status['state'] = state
            self._status['error'] = error

    def _loop(self):
        prev_t = time.perf_counter()
        frame_times = deque(maxlen=30)
        fps = 0.0
        last_log = 0.0

        while self._running:
            # pick up source / device switches requested from the web UI
            with self._lock:
                pending = self._pending_source
                self._pending_source = None
                pending_device = self._pending_device
                self._pending_device = None
            if pending is not None:
                self.source = pending
                with self._lock:
                    self._status['source'] = pending
                self._source_ready = False

            if pending_device is not None:
                print(f'[pose] device preference -> {pending_device}',
                      flush=True)
                self.device = pending_device
                with self._lock:
                    self._status['device_pref'] = pending_device
                # drop the models so they are rebuilt on the new device; the
                # source keeps running, only the tracker is replaced
                if self._tracker is not None:
                    self._tracker = None

            if not self._source_ready:
                try:
                    self._prepare_source()
                    self._source_ready = True
                    if self.is_demo:
                        self._set_status('running', None)
                except Exception as exc:  # noqa: BLE001
                    # keep retrying, e.g. a camera that is busy right now
                    print(f'[source] {exc}', flush=True)
                    self._set_status('error', str(exc))
                    time.sleep(2.0)
                    continue

            if not self.is_demo and self._tracker is None:
                try:
                    self._ensure_tracker()
                    self._set_status('running', None)
                except Exception as exc:  # noqa: BLE001
                    print(f'[pose] {exc}', flush=True)
                    self._set_status('error', str(exc))
                    time.sleep(2.0)
                    continue

            if not self.is_demo:
                self._ensure_face()

            frame = self._next_frame()
            if frame is None:
                if self._cap is not None:
                    time.sleep(0.02)
                continue

            now = time.perf_counter()
            dt = max(1e-3, now - prev_t)
            prev_t = now
            frame_times.append(dt)
            # average over a short window: no ramp-up spike on the first frames
            fps = len(frame_times) / sum(frame_times)

            keypoints = None
            scores = None
            detected = False

            if self.is_demo:
                kp_norm, sc = self._demo.read()
                keypoints = kp_norm * np.array([frame.shape[1],
                                                frame.shape[0]],
                                               dtype=np.float32)
                scores = sc
                detected = True
            elif self._tracker is not None:
                try:
                    all_kp, all_scores = self._tracker(frame)
                    keypoints, scores = self._pick_person(all_kp, all_scores)
                    if keypoints is not None:
                        # frame was mirrored for the selfie preview, so the
                        # model's left/right labels are flipped relative to the
                        # real person; put them back before anything reads them
                        keypoints, scores = _unmirror_labels(keypoints, scores)
                    detected = keypoints is not None
                except Exception as exc:  # noqa: BLE001
                    print(f'[pose] inference failed: {exc}', flush=True)
                    time.sleep(0.05)

            face = self._face_cache
            if self._face_engine is not None:
                self._face_tick += 1
                if self._face_tick % self.face_every == 0:
                    try:
                        face = self._face_engine.process(frame)
                        self._face_cache = face
                    except Exception as exc:  # noqa: BLE001
                        print(f'[face] disabled: {exc}', flush=True)
                        self._face_engine.close()
                        self._face_engine = None
                        self.face_enabled = False
                        face = None
                        self._face_cache = None
                        with self._lock:
                            self._status['face'] = False

            preview = self._render_preview(frame, keypoints, scores)
            ok, encoded = cv2.imencode(
                '.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 78])
            jpeg = encoded.tobytes() if ok else None

            with self._lock:
                if jpeg is not None:
                    self._jpeg = jpeg
                    self._frame_seq += 1
                h, w = frame.shape[:2]
                self._pose = {
                    'seq': self._frame_seq,
                    't': time.time(),
                    'fps': round(fps, 1),
                    'detected': detected,
                    'keypoints': (keypoints / np.array([w, h],
                                                       dtype=np.float32)
                                  ).round(5).tolist()
                    if keypoints is not None else [],
                    'scores': np.round(scores, 4).tolist()
                    if scores is not None else [],
                    'face': face,
                    'width': w,
                    'height': h,
                }
                self._status.update(state='running',
                                    error=None,
                                    width=w,
                                    height=h,
                                    fps=round(fps, 1),
                                    detected=detected)

            if now - last_log > 5.0:
                print(f'[engine] {fps:4.1f} fps  detected={detected}',
                      flush=True)
                last_log = now

            if self._frame_interval > 0:
                # demo clips and stills play at their own cadence; advance an
                # absolute deadline each frame so timer overshoot is paid back
                # on the next one (sleeping "interval - worked" drifts below
                # the source rate and never recovers)
                if self._frame_due <= 0.0:
                    self._frame_due = now + self._frame_interval
                else:
                    self._frame_due += self._frame_interval
                    if self._frame_due < now:
                        # stalled (source hiccup, long detection): resync
                        self._frame_due = now + self._frame_interval
                remaining = self._frame_due - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

    def _next_frame(self):
        if self._demo is not None:
            canvas = np.zeros((self._demo.height, self._demo.width, 3),
                              dtype=np.uint8)
            canvas[:] = (18, 16, 26)
            return canvas
        if self._image is not None:
            return cv2.flip(self._image, 1)
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            # loop video files, ignore transient webcam failures
            if self._cap.get(cv2.CAP_PROP_FRAME_COUNT) > 1:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None
        return cv2.flip(frame, 1)  # mirror for a natural "selfie" view

    @staticmethod
    def _pick_person(all_kp, all_scores):
        if all_kp is None or len(all_kp) == 0:
            return None, None
        scores = np.asarray(all_scores)
        if scores.ndim == 1:
            scores = scores[None, :]
        # prefer the most confident person in frame
        stable = [5, 6, 11, 12, 7, 8, 13, 14]
        conf = scores[:, stable].mean(axis=1)
        idx = int(np.argmax(conf))
        if conf[idx] < 0.25:
            return None, None
        return np.asarray(all_kp[idx], dtype=np.float32), scores[idx]

    @staticmethod
    def _render_preview(frame, keypoints, scores):
        h, w = frame.shape[:2]
        out_w = min(PREVIEW_WIDTH, w)
        out_h = int(round(h * out_w / w))
        preview = cv2.resize(frame, (out_w, out_h),
                             interpolation=cv2.INTER_AREA)
        if keypoints is not None and len(keypoints):
            from rtmlib import draw_skeleton
            scale = out_w / w
            overlay = keypoints.astype(np.float32) * scale
            preview = draw_skeleton(preview, overlay[None],
                                    np.asarray(scores)[None], kpt_thr=0.35)
        if keypoints is None:
            cv2.putText(preview, 'NO POSE', (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 120), 1,
                        cv2.LINE_AA)
        return preview

    # --------------------------------------------------------------- snapshots
    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def snapshot(self):
        with self._lock:
            return self._jpeg, self._frame_seq

    def pose(self) -> dict:
        with self._lock:
            return self._pose
