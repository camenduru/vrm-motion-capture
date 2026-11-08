"""ExMR jobs: video -> SMPL-X motion -> per-frame 3D joints.

ExMR cannot run inside this app's interpreter. It pins Python 3.12 with its
own torch/SAM stack while this app runs on 3.13 with onnxruntime, so a job is a
subprocess in ExMR's own Python that runs ``tools/demo.py`` and leaves
``outputs/<clip>/results.npz``. This module reads that npz and turns it into
the frame payload the browser retargets.

Why 3D joints rather than bone rotations: the avatar's rest pose is only known
in the browser, so the retarget has to happen there (``MocapRetargeter``). What
ExMR adds over the 2D path is exact 3D joint positions - and, unlike FlowHMR,
refined hand and foot extremities - so these frames carry ``joints3d`` instead
of ``keypoints``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXMR = Path(os.environ.get('EXMR_DIR') or (ROOT / 'ExMR'))
OUTPUT = EXMR / 'outputs'

# App-owned results, kept apart from the folders ExMR writes for its own runs.
JOB_DIR = OUTPUT / '_rtmlib'

# SMPL-X joints 0-21 are the body; 22-24 are face joints and 25-51 are fingers.
# The VRM humanoid rig this app drives has no finger bones, so they would only
# bloat the payload.
BODY_JOINTS = 22

# cap so a stray long clip cannot fill the browser with a 40 MB frame list
MAX_FRAMES = 6000

# ExMR runs four models per clip (detector, keypoints, body features, hands);
# several minutes for a short clip on a mid-range GPU is normal.
JOB_TIMEOUT = 2 * 60 * 60

# The files ``Assets.check()`` insists on for the default ``balanced`` mode,
# minus the optional DPVO install. Kept in roughly the same order so the
# message matches ExMR's own preflight.
REQUIRED = [
    'exmr/exmr.safetensors',
    'body_models/smplx/SMPLX_NEUTRAL.npz',
    'yolo/yolo11m.pt',
    'sapiens2/sapiens2_1b_pose.safetensors',
    'sam-3d-body-dinov3/model.ckpt',
    'sam-3d-body-dinov3/model_config.yaml',
    'sam-3d-body-dinov3/assets/mhr_model.pt',
    'hamer/hamer_ckpts/checkpoints/hamer.ckpt',
    'hamer/hamer_ckpts/model_config.yaml',
    'hamer/data/mano_mean_params.npz',
    'hamer/data/mano/MANO_RIGHT.pkl',
]


def interpreter() -> Optional[Path]:
    """The Python that has ExMR's torch/SAM dependencies."""
    env = os.environ.get('EXMR_PYTHON')
    if env and Path(env).exists():
        return Path(env)
    marker = EXMR / '.exmr_python'
    if marker.exists():
        found = Path(marker.read_text(encoding='utf-8').strip())
        if found.exists():
            return found
    for candidate in (EXMR / '.venv' / 'Scripts' / 'python.exe',
                      EXMR / '.venv' / 'bin' / 'python'):
        if candidate.exists():
            return candidate
    return None


def checkpoint_dir() -> Path:
    """ExMR's checkpoint root; its default relative to the checkout."""
    env = os.environ.get('EXMR_CHECKPOINT_DIR')
    if not env:
        return EXMR / 'checkpoints'
    path = Path(env)
    return path if path.is_absolute() else EXMR / path


def missing_files() -> list:
    root = checkpoint_dir()
    return [name for name in REQUIRED if not (root / name).exists()]


def availability() -> dict:
    """What the UI needs to know before offering the option."""
    if not (EXMR / 'tools' / 'demo.py').exists():
        return {'ok': False,
                'reason': f'ExMR checkout not found in {EXMR}',
                'missing': []}
    python = interpreter()
    if python is None:
        return {'ok': False,
                'reason': 'ExMR Python not configured - create its environment '
                          '(ExMR/docs/INSTALL.md) and set EXMR_PYTHON to that '
                          'interpreter',
                'missing': []}
    missing = missing_files()
    if missing:
        return {'ok': False,
                'reason': f'{len(missing)} ExMR file(s) missing - SAM 3D Body '
                          f'and HaMeR are gated downloads, see '
                          f'ExMR/docs/INSTALL.md',
                'missing': missing}
    return {'ok': True, 'reason': 'ready', 'python': str(python),
            'missing': []}


def frames_from_npz(path: Path, max_frames: int = MAX_FRAMES) -> dict:
    """Turn ExMR's results npz into the per-frame payload the browser wants.

    ``joints_global`` is the gravity-aligned, smoothed and extremity-refined
    motion. The retargeter projects the joints into the person's own body frame
    per frame, so the world orientation - and the choice between this and the
    camera-frame ``joints_incam`` - does not change the resulting bone
    directions; the global output simply carries ExMR's post-processing.
    """
    started = time.perf_counter()
    with np.load(path) as data:
        keys = set(data.files)
        if 'joints_global' not in keys:
            raise RuntimeError(
                f'{path.name} has no joints_global (found: '
                f'{", ".join(sorted(keys))}) - this npz did not come from '
                f'ExMR 0.1.4 or newer')
        joints = np.asarray(data['joints_global'], dtype=np.float32)
        fps = float(data['fps']) if 'fps' in keys else 30.0

    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise RuntimeError(f'{path.name}: unexpected joint shape '
                           f'{joints.shape}, expected (T, J, 3)')
    joints = joints[:, :BODY_JOINTS]

    if not 1.0 <= fps <= 240.0:
        fps = 30.0

    # JSON cannot carry NaN/Inf. A bad frame is filled from the nearest good
    # one so the bake keeps its timing instead of silently speeding up.
    finite = np.isfinite(joints).all(axis=(1, 2))
    if not finite.all():
        if not finite.any():
            raise RuntimeError(f'{path.name}: every frame has non-finite joints')
        good = np.flatnonzero(finite)
        for index in np.flatnonzero(~finite):
            nearest = good[np.abs(good - index).argmin()]
            joints[index] = joints[nearest]

    total = len(joints)
    truncated = total > max_frames
    if truncated:
        joints = joints[:max_frames]
        total = max_frames

    root = joints[:, 0] - joints[0, 0]
    frames = [{
        't': round(i / fps, 4),
        'fps': fps,
        'detected': bool(finite[i]),
        'joints3d': np.round(joints[i], 4).tolist(),
        'root': np.round(root[i], 4).tolist(),
    } for i in range(total)]

    return {
        'frames': frames,
        'fps': fps,
        'duration': total / fps,
        'joints': BODY_JOINTS,
        'truncated': truncated,
        'seconds': round(time.perf_counter() - started, 2),
    }


def output_dir(video: Path) -> Path:
    """Where the app keeps its own ExMR run for this clip."""
    return JOB_DIR / video.stem


def clear_stale_cache(video: Path, folder: Path) -> bool:
    """Drop app-owned results when the clip is newer than what was computed.

    ExMR keys its caches on the video path, which is right for a re-run of the
    same file and wrong for a replaced file that happens to keep its name. Only
    the app's private ``outputs/_rtmlib`` folder is touched.
    """
    try:
        video_time = video.stat().st_mtime
    except OSError:
        return False
    candidates = [folder / 'results.npz', folder / 'results.pt',
                  folder / 'run.json']
    cache = folder / 'cache'
    if cache.is_dir():
        candidates.extend(p for p in cache.rglob('*') if p.is_file())
    stale = any(p.exists() and p.stat().st_mtime < video_time
                for p in candidates)
    if not stale:
        return False
    shutil.rmtree(cache, ignore_errors=True)
    for name in ('results.npz', 'results.pt', 'run.json'):
        (folder / name).unlink(missing_ok=True)
    return True


class ExMRJob:
    """One ExMR run, tracked so the UI can poll it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._proc = None
        self._cancel = threading.Event()
        # index into _PHASES, so a late line from an earlier stage is ignored
        self._stage = -1
        self._status = {
            'state': 'idle',
            'phase': '',
            'message': '',
            'error': '',
            'started': 0.0,
            'elapsed': 0.0,
            'frames': 0,
            'result': None,
        }

    # ------------------------------------------------------------------ state
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

    def _set(self, **fields):
        with self._lock:
            self._status.update(fields)

    def busy(self) -> bool:
        with self._lock:
            return self._status['state'] == 'running'

    # ------------------------------------------------------------------ start
    def start(self, video: Path, static_cam: bool = True) -> dict:
        check = availability()
        if not check['ok']:
            raise RuntimeError(check['reason'])
        if not video.exists():
            raise FileNotFoundError(f'video not found: {video}')
        with self._lock:
            if self._status['state'] == 'running':
                raise RuntimeError('an ExMR job is already running')
            self._cancel.clear()
            self._stage = -1
            self._status.update(state='running', phase='starting',
                                message='loading models', error='',
                                started=time.time(), elapsed=0.0, frames=0,
                                result=None)
        self._thread = threading.Thread(
            target=self._run, args=(video, bool(static_cam)), daemon=True)
        self._thread.start()
        return self.status()

    def cancel(self) -> dict:
        with self._lock:
            running = self._status['state'] == 'running'
            if running:
                self._cancel.set()
                self._status.update(phase='cancelling',
                                    message='stopping ExMR')
            proc = self._proc
        if running and proc is not None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        return self.status()

    # ------------------------------------------------------------------- work
    # what demo.py prints -> the phase the UI shows, in pipeline order.
    # Cached stages log "Using cached <file>" and are deliberately not matched,
    # so a re-run jumps straight to whichever stage actually runs.
    _PHASES = (
        (('extracting frames',), ('reading', 'extracting frames')),
        (('computing tracks',), ('tracking', 'finding the person')),
        (('computing keypoints',), ('keypoints', 'estimating 2D keypoints')),
        (('computing body_features',), ('body', 'extracting body features')),
        (('computing hand_features',), ('hands', 'extracting hand features')),
        (('computing camera_rotations',),
         ('camera', 'estimating camera motion')),
        (('running exmr',), ('generating', 'refining hands and feet')),
    )

    def _phase_of(self, line: str):
        """Update the phase from one line of the demo's output.

        Stages only ever move forward, so a cached-step line that arrives after
        generation started must not send the UI back.
        """
        low = line.lower()
        for index, (needles, (phase, message)) in enumerate(self._PHASES):
            if any(needle in low for needle in needles):
                if index > self._stage:
                    self._stage = index
                    self._set(phase=phase, message=message)
                return

    def _run(self, video: Path, static_cam: bool):
        try:
            npz = self._infer(video, static_cam)
            self._set(phase='retargeting', message='reading the motion')
            result = frames_from_npz(npz)
            self._set(state='done', phase='done', result=result,
                      frames=len(result['frames']),
                      message=f'{len(result["frames"])} frames '
                              f'({result["duration"]:.1f}s)')
        except Exception as exc:  # noqa: BLE001
            if self._cancel.is_set():
                self._set(state='cancelled', phase='cancelled', error='')
            else:
                self._set(state='error', phase='failed', error=str(exc))

    def _infer(self, video: Path, static_cam: bool) -> Path:
        python = interpreter()
        if python is None:
            raise RuntimeError('ExMR Python not configured')
        folder = output_dir(video)
        folder.mkdir(parents=True, exist_ok=True)
        clear_stale_cache(video, folder)

        cmd = [str(python), 'tools/demo.py', '--video', str(video),
               '--output_dir', str(folder), '--no_render',
               '--mode', 'balanced']
        if static_cam:
            cmd.append('--static_cam')

        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        # the pipeline prints arrows (U+2192); a cp1252 console would abort the
        # run mid-stage with UnicodeEncodeError, so force UTF-8 on the child
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        env['PYTHONPATH'] = str(EXMR) + os.pathsep + env.get('PYTHONPATH', '')
        # HaMeR's renderer module forces PyOpenGL's EGL backend, whose DLL is
        # not on a stock Windows install; WGL works even without rendering.
        if os.name == 'nt':
            env.setdefault('PYOPENGL_PLATFORM', 'win32')

        self._set(phase='starting', message='loading models')
        log_tail: list = []
        if self._cancel.is_set():
            raise RuntimeError('cancelled')
        proc = subprocess.Popen(cmd, cwd=str(EXMR), env=env,
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding='utf-8', errors='replace')
        self._proc = proc
        if self._cancel.is_set():
            proc.kill()
        # a silent hang never reaches the loop below, so it needs its own timer
        watchdog = threading.Timer(JOB_TIMEOUT, proc.kill)
        watchdog.start()
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.rstrip()
                if not line:
                    continue
                log_tail.append(line)
                if len(log_tail) > 40:
                    log_tail.pop(0)
                self._phase_of(line)
        finally:
            watchdog.cancel()
            proc.wait()
            self._proc = None

        if proc.returncode != 0:
            raise RuntimeError('ExMR failed:\n' + '\n'.join(log_tail[-8:]))

        found = folder / 'results.npz'
        if not found.exists():
            raise RuntimeError(
                f'ExMR produced no results for {video.name} - see the log '
                f'above')
        return found
