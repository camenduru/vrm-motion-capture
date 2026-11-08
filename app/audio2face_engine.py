"""NVIDIA Audio2Face-3D inference on the regression models (v2.3).

Runs the public Hugging Face weights (e.g. nvidia/Audio2Face-3D-v2.3.1-Claire)
directly with ONNX Runtime and re-implements the post-processing stages of the
Audio2Face-3D SDK (github.com/NVIDIA/Audio2Face-3D-SDK) in numpy:

    audio window  -> network.onnx -> coefficients
                  -> skin PCA reconstruction
                  -> skin animator (face-mask split double EMA smoothing)
                  -> blendshape solve (ARKit weights, bounded least squares)
                  -> eyeball rotations (+ saccade table)

The result is a per-frame ARKit blendshape track the web layer feeds into the
same VRM expression mapping used by the webcam face tracker.

Every failure mode (model missing, onnxruntime missing, ffmpeg missing) is
reported through `availability()` instead of raising, so the server can answer
requests with a clear message.
"""

from __future__ import annotations

import json
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / 'models' / 'audio2face'

# The regression networks run at 60 fps with a sliding audio window.
FPS = 60

# Weights below this never move a VRM expression; dropping them keeps the
# stored track (and the JSON payload) small.
STORE_THRESHOLD = 0.002

_FRAMES_PER_BATCH = 32


def availability() -> dict:
    """What is installed and usable, for the UI's feature checks."""
    models = sorted(
        p.name for p in MODELS_DIR.glob('*')
        if (p / 'model.json').exists() and (p / 'network.onnx').exists())
    try:
        import onnxruntime  # noqa: F401
        ort_ok = True
    except Exception:  # noqa: BLE001
        ort_ok = False
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True,
                       check=True)
        ffmpeg_ok = True
    except Exception:  # noqa: BLE001
        ffmpeg_ok = False
    return {
        'models': models,
        'onnxruntime': ort_ok,
        'ffmpeg': ffmpeg_ok,
        'ready': bool(models) and ort_ok and ffmpeg_ok,
    }


def decode_audio(path: Path) -> np.ndarray:
    """Decode any audio/video file to mono 16 kHz float32 via ffmpeg."""
    cmd = ['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1', '-ar',
           '16000', '-f', 'f32le', '-']
    out = subprocess.run(cmd, capture_output=True, check=True)
    samples = np.frombuffer(out.stdout, dtype=np.float32).copy()
    if samples.size == 0:
        raise ValueError('no audio decoded from the file')
    return samples


class Audio2FaceEngine:
    """One loaded Audio2Face-3D model, ready for offline track extraction."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        card = json.loads((self.model_dir / 'model.json').read_text())
        info = json.loads((self.model_dir / 'network_info.json').read_text())
        params = info['params']
        audio = info['audio_params']

        self.name = f"{info['id']['actor']}_v{info['id']['version']}"
        self.buffer_len = int(audio['buffer_len'])
        self.buffer_ofs = int(audio['buffer_ofs'])
        self.samplerate = int(audio.get('bufferSamplerate',
                                        audio['samplerate']))

        # result layout: [skin coeffs | tongue coeffs | jaw | eyes]
        self.skin_dims = int(params['num_shapes_skin'])
        self.tongue_dims = int(params['num_shapes_tongue'])
        self.jaw_dims = int(params['result_jaw_size'])
        self.eyes_dims = int(params['result_eyes_size'])
        self.emotion_dims = (int(params['implicit_emotion_len']) +
                             len(params['explicit_emotions']))

        # ---- geometry data -------------------------------------------------
        model_data = np.load(self.model_dir / card['modelDataPath'])
        self.skin_mean = np.asarray(model_data['shapes_mean_skin'],
                                    dtype=np.float32)
        self.skin_shapes = np.asarray(model_data['shapes_matrix_skin'],
                                      dtype=np.float32)  # (S, V, 3)
        self.lip_open_delta = np.asarray(model_data['lip_open_pose_delta'],
                                         dtype=np.float32).reshape(-1)
        self.eye_close_delta = np.asarray(model_data['eye_close_pose_delta'],
                                          dtype=np.float32).reshape(-1)

        # ---- blendshape rig ------------------------------------------------
        bs = np.load(self.model_dir /
                     card['blendshapePaths']['skin']['data'])
        self.bs_names = [n.decode() for n in bs['poseNames']]
        self.bs_neutral = np.asarray(bs['neutral'], dtype=np.float32)
        # 'neutral' labels the base pose; the solve variables are the delta
        # shapes that follow it (52 ARKit shapes, 'tongueOut' last).
        self.solve_names = self.bs_names[1:]
        deltas = [np.asarray(bs[n], dtype=np.float32).reshape(-1)
                  for n in self.solve_names]
        bs_matrix = np.stack(deltas, axis=1)  # (V*3, 52)
        self.frontal = np.asarray(bs['frontalMask'], dtype=np.int64)

        cfg = json.loads((self.model_dir /
                          card['blendshapePaths']['skin']['config']
                          ).read_text())['blendshape_params']
        self.l1_reg = float(cfg.get('strengthL1regularization', 1.0))
        self.l2_reg = float(cfg.get('strengthL2regularization', 3.5))
        self.temporal_reg = float(cfg.get('strengthTemporalSmoothing', 0.3))
        self.template_bb = float(cfg.get('templateBBSize', 54.7))
        active = cfg.get('bsSolveActivePoses', [])
        self.active = (np.array([bool(v) for v in active], dtype=bool)
                       if len(active) == len(self.solve_names)
                       else np.ones(len(self.solve_names), dtype=bool))

        # ---- animator config (model_config.json mirrors the SDK params) ----
        anim_path = self.model_dir / card.get('modelConfigPath',
                                              'model_config.json')
        anim = json.loads(anim_path.read_text())['config']
        self.input_strength = float(anim.get('input_strength', 1.0))
        self.upper_smoothing = float(anim.get('upper_face_smoothing', 0.001))
        self.lower_smoothing = float(anim.get('lower_face_smoothing', 0.006))
        self.upper_strength = float(anim.get('upper_face_strength', 1.0))
        self.lower_strength = float(anim.get('lower_face_strength', 1.25))
        self.face_mask_level = float(anim.get('face_mask_level', 0.6))
        self.face_mask_softness = float(anim.get('face_mask_softness',
                                                 0.0085))
        self.skin_strength = float(anim.get('skin_strength', 1.0))
        self.blink_strength = float(anim.get('blink_strength', 1.0))
        self.lip_open_offset = float(anim.get('lip_open_offset', 0.0))
        self.eyelid_open_offset = float(anim.get('eyelid_open_offset', 0.0))
        self.blink_offset = float(anim.get('blink_offset', 0.0))
        self.eyeballs_strength = float(anim.get('eyeballs_strength', 1.0))
        self.saccade_strength = float(anim.get('saccade_strength', 0.6))

        saccade = np.asarray(model_data.get('saccade_rot_matrix',
                                            np.zeros((1, 2), np.float32)),
                             dtype=np.float32)
        self.saccade_rot = saccade

        # ---- restricted solve geometry --------------------------------------
        # Only frontal-mask vertices ever reach the blendshape solve, so the
        # PCA reconstruction and animator run on those columns only.
        idx3 = (self.frontal[:, None] * 3 + np.arange(3)[None, :]
                ).reshape(-1)  # [v0x, v0y, v0z, v1x, ...]
        self.solve_shapes = self.skin_shapes[:, self.frontal, :].reshape(
            self.skin_dims, -1)                                  # (S, M*3)
        self.solve_neutral = self.skin_mean.reshape(-1)[idx3]    # (M*3,)
        self.solve_bs = bs_matrix[idx3, :]                       # (M*3, 52)
        self.solve_lip = self.lip_open_delta[idx3]
        self.solve_eye_close = self.eye_close_delta[idx3]

        # Face mask (lower-face blend) from the neutral pose's Y coordinate,
        # normalised over the whole head exactly like the SDK kernel.
        ys = self.skin_mean[:, 1].astype(np.float32)
        y_min, y_max = float(ys.min()), float(ys.max())
        span = max(y_max - y_min, 1e-6)
        self.mask_lower = 1.0 / (
            1.0 + np.exp(-(self.face_mask_level - (ys - y_min) / span) /
                         max(self.face_mask_softness, 1e-6)))
        # expand per vertex to per-coordinate (x, y, z share one mask value)
        self.mask_lower = np.repeat(self.mask_lower[self.frontal],
                                    3)                # (M*3,)

        # Bounding-box scale for the regularisation weights.
        bb = self.skin_mean.max(axis=0) - self.skin_mean.min(axis=0)
        scale = (float(np.linalg.norm(bb)) / self.template_bb) ** 2
        l1_scale = 0.25 * scale
        l2_scale = 10.0 * scale
        temporal_scale = 100.0 * scale

        # Quadratic programme for one frame:
        #   min 0.5 w'Hw - w'c   s.t. 0 <= w <= 1
        # H = B'B + L1^2*L1scale*ones + L2*L2scale*I + Temporal*Tempscale*I
        # (the SDK's symmetry term is a no-op for these models: the shipped
        # config carries no symmetry pairs)
        cols = self.solve_bs[:, self.active]                  # (M*3, A)
        gram = (cols.T @ cols).astype(np.float64)
        reg = np.full(gram.shape, self.l1_reg ** 2 * l1_scale)
        reg[np.diag_indices_from(reg)] += (self.l2_reg * l2_scale +
                                           self.temporal_reg * temporal_scale)
        self.hessian = gram + reg
        self.bt_cols = cols.T.astype(np.float64)              # (A, M*3)
        # The temporal term enters the linear side as TemporalReg * scale
        # (not the Hessian's x100 scale - see BlendshapeSolver::SolveAsync's
        # `ATb + TemporalReg * scaleFactor * prevWeights`), which keeps the
        # frame-to-frame recursion stable.
        self.temporal_weight = self.temporal_reg * scale

        # Pre-factor the ADMM system (H + rho*I): the z-step is a clamp, so
        # only the w-step needs a linear solve and it is constant per track.
        rho = max(1.0, float(np.trace(self.hessian)) / gram.shape[0])
        self.admm_rho = rho
        self.admm_inv = np.linalg.inv(
            self.hessian + rho * np.eye(gram.shape[0]))

        # ---- onnx -----------------------------------------------------------
        import onnxruntime as ort
        # expose the pip CUDA/cuDNN DLLs before the provider loads (same
        # workaround the pose engine needs on Windows)
        from .pose_engine import _add_nvidia_dll_dirs, _preload_cuda_dlls
        _add_nvidia_dll_dirs()
        _preload_cuda_dlls()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        network = self.model_dir / card.get('networkPath', 'network.onnx')
        if not network.exists():
            network = self.model_dir / 'network.onnx'
        # ORT silently drops a GPU provider whose runtime DLLs are missing,
        # so verify what the session actually got and report that
        available = ort.get_available_providers()
        self.device = 'cpu'
        session = None
        for pref in ('CUDAExecutionProvider', 'TensorrtExecutionProvider'):
            if pref not in available:
                continue
            try:
                session = ort.InferenceSession(
                    str(network), opts, providers=[pref,
                                                   'CPUExecutionProvider'])
            except Exception:  # noqa: BLE001
                session = None
            if session is not None and \
                    session.get_providers()[0] == pref:
                self.device = pref.replace('ExecutionProvider',
                                           '').lower()
                break
            session = None
        if session is None:
            session = ort.InferenceSession(
                str(network), opts, providers=['CPUExecutionProvider'])
        self.session = session
        self.in_name = self.session.get_inputs()[0].name
        self.emo_name = self.session.get_inputs()[1].name
        self.out_name = self.session.get_outputs()[0].name

    # ------------------------------------------------------------------ parts
    def _infer_batch(self, windows: np.ndarray) -> np.ndarray:
        """windows (B, buffer_len) float32 -> results (B, D)."""
        batch = windows.shape[0]
        emo = np.zeros((batch, 1, self.emotion_dims), dtype=np.float32)
        out = self.session.run(
            [self.out_name],
            {self.in_name: windows[:, None, :].astype(np.float32),
             self.emo_name: emo})[0]
        return out.reshape(batch, -1)

    def _solve_qp(self, lin: np.ndarray, iterations: int = 64) -> np.ndarray:
        """min 0.5 w'Hw - w'lin s.t. 0 <= w <= 1 via ADMM."""
        rho = self.admm_rho
        z = np.clip(np.linalg.solve(self.hessian, lin), 0.0, 1.0)
        u = np.zeros_like(z)
        for _ in range(iterations):
            w = self.admm_inv @ (lin + rho * (z - u))
            z_new = np.clip(w + u, 0.0, 1.0)
            u = u + w - z_new
            if np.max(np.abs(w - z_new)) < 1e-6:
                z = z_new
                break
            z = z_new
        return z

    # ------------------------------------------------------------------- main
    def process(self, samples: np.ndarray, fps: int = FPS,
                stop=None, progress=None) -> dict:
        """Full offline pass: mono 16 kHz float32 -> animation track dict."""
        n = samples.size
        total = max(1, int(math.ceil((n / self.samplerate) * fps)))

        # window start of frame i: (i * samplerate) / fps - buffer_ofs, zero
        # padded before the start and after the end of the track
        starts = (np.arange(total, dtype=np.int64) * self.samplerate) // fps \
            - self.buffer_ofs
        idx = starts[:, None] + np.arange(self.buffer_len)[None, :]
        valid = (idx >= 0) & (idx < n)
        idx_clipped = np.clip(idx, 0, n - 1)
        windows = np.where(valid, samples[idx_clipped], 0.0)
        windows = (windows * self.input_strength).astype(np.float32)

        # EMA state for the skin animator (cascaded pair per face half)
        alpha_l = (1.0 - 0.5 ** ((1.0 / fps) / self.lower_smoothing)
                   if self.lower_smoothing > 0 else 1.0)
        alpha_u = (1.0 - 0.5 ** ((1.0 / fps) / self.upper_smoothing)
                   if self.upper_smoothing > 0 else 1.0)
        lower1 = lower2 = upper1 = upper2 = None

        # static offset added to every vertex before smoothing
        offset = (self.solve_eye_close *
                  (-self.eyelid_open_offset +
                   self.blink_offset * self.blink_strength) +
                  self.solve_lip * self.lip_open_offset)

        active_names = [n for n, a in zip(self.solve_names, self.active) if a]
        weights_all = np.zeros((total, len(active_names)), dtype=np.float32)
        eyes_out = np.zeros((total, 4), dtype=np.float32)
        prev_w = np.zeros(len(active_names), dtype=np.float64)

        for begin in range(0, total, _FRAMES_PER_BATCH):
            if stop is not None and stop.is_set():
                raise RuntimeError('cancelled')
            end = min(begin + _FRAMES_PER_BATCH, total)
            results = self._infer_batch(windows[begin:end])
            for row, coeff in enumerate(results):
                delta = (coeff[:self.skin_dims].astype(np.float32)
                         @ self.solve_shapes)

                # --- skin animator (ComputeSkinPostProcessing) ------------
                d = self.skin_strength * delta + offset
                if lower1 is None:
                    lower1 = lower2 = upper1 = upper2 = d
                else:
                    lower1 += (d - lower1) * alpha_l
                    lower2 += (lower1 - lower2) * alpha_l
                    upper1 += (d - upper1) * alpha_u
                    upper2 += (upper1 - upper2) * alpha_u
                geom = (self.solve_neutral +
                        upper2 * self.upper_strength * (1.0 - self.mask_lower)
                        + lower2 * self.lower_strength * self.mask_lower)

                # --- blendshape solve -------------------------------------
                target = geom.reshape(-1).astype(np.float64) \
                    - self.solve_neutral
                lin = self.bt_cols @ target + self.temporal_weight * prev_w
                weights = self._solve_qp(lin)
                weights_all[begin + row] = weights
                prev_w = weights

                # --- eyes (degrees; SDK AnimatorEyes output, see its unit
                # tests where offsets like +-2.0/+-100.0 pass validation) ---
                # network rotation + saccade table sampled at 30 fps
                eye = coeff[self.skin_dims + self.tongue_dims +
                            self.jaw_dims:]
                sacc = self.saccade_rot[(begin + row) // 2 %
                                        len(self.saccade_rot)]
                eyes_out[begin + row] = (
                    self.eyeballs_strength * eye
                    + self.saccade_strength *
                    np.array([sacc[0], sacc[1], sacc[0], sacc[1]],
                             dtype=np.float32))

            if progress:
                progress(min(1.0, end / total))

        # columnar storage: one rounded array per shape that ever moved
        ever = (weights_all > STORE_THRESHOLD).any(axis=0)
        shapes_out = {
            name: [round(float(v), 3) for v in weights_all[:, k]]
            for k, name in enumerate(active_names) if ever[k]
        }
        duration = round(total / fps, 3)
        return {
            'version': 1,
            'fps': fps,
            'frames': total,
            'duration': duration,
            'shapes': shapes_out,
            'eyes': [[round(float(a), 4) for a in row] for row in eyes_out],
        }


class Audio2FaceJob:
    """One offline face-animation extraction, tracked so the UI can poll it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._status = {'state': 'idle', 'error': '', 'started': 0.0,
                        'elapsed': 0.0, 'progress': 0.0, 'result': None}

    def status(self) -> dict:
        with self._lock:
            data = dict(self._status)
        if data['state'] == 'running' and data['started']:
            data['elapsed'] = round(time.time() - data['started'], 1)
        data.pop('result', None)          # keep the poll response small
        return data

    def result(self) -> Optional[dict]:
        with self._lock:
            return self._status.get('result')

    def busy(self) -> bool:
        with self._lock:
            return self._status['state'] == 'running'

    def start(self, audio: Path, model: str | None = None) -> dict:
        with self._lock:
            if self._status['state'] == 'running':
                raise RuntimeError('an Audio2Face job is already running')
            self._stop.clear()
            self._status.update(state='running', error='', started=time.time(),
                                elapsed=0.0, progress=0.0, result=None)
        self._thread = threading.Thread(
            target=self._run, args=(audio, model), daemon=True)
        self._thread.start()
        return self.status()

    def _run(self, audio: Path, model: str | None):
        try:
            track = self._extract(audio, model)
            with self._lock:
                self._status.update(state='done', result=track,
                                    progress=1.0)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._status.update(state='error', error=str(exc))

    def _extract(self, audio: Path, model: str | None) -> dict:
        available = availability()
        if not available['ready']:
            missing = [k for k in ('models', 'onnxruntime', 'ffmpeg')
                       if not available[k]]
            raise RuntimeError(
                'Audio2Face is not set up (missing: ' + ', '.join(missing) +
                '); see README "Audio2Face-3D" for the download step')
        model_dir = (MODELS_DIR / model) if model else \
            MODELS_DIR / available['models'][0]
        if not (model_dir / 'model.json').exists():
            raise RuntimeError(f'unknown Audio2Face model: {model_dir.name}')

        engine = _engine_for(model_dir)
        samples = decode_audio(audio)
        track = engine.process(samples, stop=self._stop,
                               progress=self._progress)
        track['audio_url'] = f'/audio/{audio.name}'
        track['model'] = engine.name
        track['source_audio'] = audio.name
        return track

    def _progress(self, frac: float):
        with self._lock:
            self._status['progress'] = round(min(1.0, frac), 3)

    def cancel(self) -> dict:
        self._stop.set()
        return self.status()


_engine_lock = threading.Lock()
_engine: Optional[Audio2FaceEngine] = None
_engine_name: Optional[str] = None


def _engine_for(model_dir: Path) -> Audio2FaceEngine:
    """Load (and cache) one engine; models are big, so reuse across jobs."""
    global _engine, _engine_name
    with _engine_lock:
        if _engine is None or _engine_name != model_dir.name:
            _engine = Audio2FaceEngine(model_dir)
            _engine_name = model_dir.name
            print(f'[audio2face] loaded {model_dir.name} '
                  f'({_engine.device})', flush=True)
        return _engine
