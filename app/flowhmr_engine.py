"""FlowHMR jobs: video -> SMPL-H motion -> per-frame 3D joints.

FlowHMR cannot run inside this app's interpreter. It pins Python 3.10 with its
own torch/perception stack (YOLOX, VGGT-Omega, SAM 3D Body, SMPL-H) while this
app runs on 3.13 with onnxruntime, so a job is a subprocess in FlowHMR's own
Python that runs ``run_v2m_demo_generation.py`` and leaves a ``.npz`` of SMPL-H
motion. This module reads that npz and turns it into the frame payload the
browser retargets.

Why 3D joints rather than bone rotations: the avatar's rest pose is only known
in the browser, so the retarget has to happen there (``MocapRetargeter``). What
FlowHMR adds is exact 3D joint positions - the thing the 2D path has to guess
from limb foreshortening - so these frames carry ``joints3d`` instead of
``keypoints``.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FLOWHMR = Path(os.environ.get('FLOWHMR_DIR') or (ROOT / 'flowhmr'))
OUTPUT = FLOWHMR / 'output'

# SMPL-H joints 0-21 are the body; 22-51 are fingers, and the VRM humanoid rig
# this app drives has no finger bones, so they would only bloat the payload.
BODY_JOINTS = 22

# cap so a stray long clip cannot fill the browser with a 40 MB frame list
MAX_FRAMES = 6000

# FlowHMR runs four models per clip (detector, camera, body features, flow
# matching); a 2 minute clip on a mid-range GPU is already ~20 minutes.
JOB_TIMEOUT = 2 * 60 * 60

# The files run_v2m_demo_generation.check_files() insists on, plus the
# checkpoint itself. Kept in the same order so the message matches theirs.
REQUIRED = [
    'ckpts/yolox/yolox_l.pth',
    'ckpts/vggt-omega/vggt_omega_1b_512.pt',
    'ckpts/sam-3d-body-dinov3/model.ckpt',
    'ckpts/sam-3d-body-dinov3/assets/mhr_model.pt',
    'assets/body_models/smplh/neutral/model.npz',
    'assets/body_models/smpl_neutral_J_regressor.pt',
]
DEFAULT_CKPT = 'checkpoints/flowhmr_latest/flowhmr_latest.ckpt'


def interpreter() -> Optional[Path]:
    """The Python that has FlowHMR's V2M dependencies."""
    env = os.environ.get('FLOWHMR_PYTHON')
    if env and Path(env).exists():
        return Path(env)
    marker = FLOWHMR / '.flowhmr_python'
    if marker.exists():
        found = Path(marker.read_text(encoding='utf-8').strip())
        if found.exists():
            return found
    for candidate in (FLOWHMR / '.venv' / 'Scripts' / 'python.exe',
                      FLOWHMR / '.venv' / 'bin' / 'python'):
        if candidate.exists():
            return candidate
    return None


def checkpoint() -> Path:
    env = os.environ.get('FLOWHMR_CKPT')
    return Path(env) if env else (FLOWHMR / DEFAULT_CKPT)


def missing_files() -> list:
    missing = [name for name in REQUIRED if not (FLOWHMR / name).exists()]
    if not checkpoint().exists():
        missing.insert(0, str(checkpoint()))
    return missing


def cached_artifacts(video: Path) -> list:
    """The per-video caches ``run_v2m_demo_generation.py --overwrite`` drops."""
    stem = video.stem
    folder = OUTPUT / stem
    return [folder / name for name in (
        f'{stem}_30fps.mp4', f'{stem}_bbox.npz',
        f'{stem}_vggt_camera.npz', f'{stem}_sam3d_feat.pt')]


def needs_overwrite(video: Path) -> bool:
    """True when a cached artefact predates the clip it was built from.

    The pipeline keys its caches on the video *stem*, which is right for a
    re-run of the same clip and wrong for a different clip that happens to
    share a name. It only reports the mismatch for features, so detection and
    camera caches would otherwise be reused silently.
    """
    try:
        video_time = video.stat().st_mtime
    except OSError:
        return False
    return any(p.exists() and p.stat().st_mtime < video_time
               for p in cached_artifacts(video))


def availability() -> dict:
    """What the UI needs to know before offering the option."""
    if not (FLOWHMR / 'run_v2m_demo_generation.py').exists():
        return {'ok': False,
                'reason': f'FlowHMR checkout not found in {FLOWHMR}',
                'missing': []}
    python = interpreter()
    if python is None:
        return {'ok': False,
                'reason': 'FlowHMR Python not configured - create its V2M '
                          'environment (flowhmr/docs/install.md) and set '
                          'FLOWHMR_PYTHON to that interpreter',
                'missing': []}
    missing = missing_files()
    if missing:
        return {'ok': False,
                'reason': f'{len(missing)} FlowHMR file(s) missing - the '
                          f'perception checkpoints are gated downloads, see '
                          f'flowhmr/docs/install.md',
                'missing': missing}
    return {'ok': True, 'reason': 'ready', 'python': str(python),
            'ckpt': str(checkpoint()), 'missing': []}


def frames_from_npz(path: Path, max_frames: int = MAX_FRAMES) -> dict:
    """Turn a FlowHMR motion npz into the per-frame payload the browser wants.

    ``local_joints_positions`` is (T, 52, 3) with the root's XZ already removed,
    so it carries exactly the joint *directions* the retarget needs. ``trans``
    is the pelvis offset from its rest position, which is what moves the avatar
    around the floor.
    """
    started = time.perf_counter()
    with np.load(path) as data:
        keys = set(data.files)
        if 'local_joints_positions' not in keys:
            raise RuntimeError(
                f'{path.name} has no local_joints_positions (found: '
                f'{", ".join(sorted(keys))}) - this npz did not come from '
                f'flowhmr/pipeline/pipeline_v2m.py')
        joints = np.asarray(data['local_joints_positions'], dtype=np.float32)
        fps = float(data['mocap_framerate']) if 'mocap_framerate' in keys \
            else 30.0
        trans = np.asarray(data['trans'], dtype=np.float32) \
            if 'trans' in keys else None

    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise RuntimeError(f'{path.name}: unexpected joint shape '
                           f'{joints.shape}, expected (T, J, 3)')
    joints = joints[:, :BODY_JOINTS]

    if trans is not None and len(trans) == len(joints):
        # save_fbx rotates `poses` and `trans` 180 deg about Y before writing
        # the npz but stores local_joints_positions unrotated. Undo it so the
        # root offset is expressed in the same frame as the joints; only the
        # delta matters, so the pelvis rest offset drops out.
        trans = trans * np.array([-1.0, 1.0, -1.0], np.float32)
        root = trans - trans[0]
    else:
        root = np.zeros((len(joints), 3), np.float32)

    if not 1.0 <= fps <= 240.0:
        fps = 30.0

    total = len(joints)
    truncated = total > max_frames
    if truncated:
        joints = joints[:max_frames]
        root = root[:max_frames]
        total = max_frames

    frames = [{
        't': round(i / fps, 4),
        'fps': fps,
        'detected': True,
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


class FlowHMRJob:
    """One FlowHMR run, tracked so the UI can poll it."""

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
    def start(self, video: Path, postprocess: bool = True,
              vggt_interval: int = 1) -> dict:
        check = availability()
        if not check['ok']:
            raise RuntimeError(check['reason'])
        if not video.exists():
            raise FileNotFoundError(f'video not found: {video}')
        with self._lock:
            if self._status['state'] == 'running':
                raise RuntimeError('a FlowHMR job is already running')
            self._cancel.clear()
            self._stage = -1
            self._status.update(state='running', phase='starting',
                                message='loading models', error='',
                                started=time.time(), elapsed=0.0, frames=0,
                                result=None)
        self._thread = threading.Thread(
            target=self._run, args=(video, postprocess, int(vggt_interval)),
            daemon=True)
        self._thread.start()
        return self.status()

    def cancel(self) -> dict:
        with self._lock:
            running = self._status['state'] == 'running'
            if running:
                self._cancel.set()
                self._status.update(phase='cancelling',
                                    message='stopping FlowHMR')
            proc = self._proc
        if running and proc is not None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        return self.status()

    # ------------------------------------------------------------------- work
    # what the pipeline prints -> the phase the UI shows, in pipeline order.
    # The runner announces its stages as "=== Step N" and tags the detail lines
    # ([preprocess], [detection], [vggt], [sam], [generation]); both are matched
    # so the phase also advances while a cached step is loaded instead of
    # computed. Order matters: phases only ever move forward.
    _PHASES = (
        (('[preprocess]',), ('reading', 'transcoding the clip')),
        (('=== step 1', '[detection]'), ('detecting', 'finding the person')),
        (('=== step 2', '[vggt]'), ('camera', 'estimating camera motion')),
        (('=== step 3', '[sam]', '[feature_extract]'),
         ('features', 'extracting body features')),
        (('=== step 4', '[generation]', '[gen]'),
         ('generating', 'generating motion')),
    )

    def _phase_of(self, line: str):
        """Update the phase from one line of the runner's output.

        Stages only ever move forward, so a cached-step line that arrives
        after generation started (``[vggt] skip cached camera``) must not send
        the UI back to camera estimation.
        """
        low = line.lower()
        for index, (needles, (phase, message)) in enumerate(self._PHASES):
            if any(needle in low for needle in needles):
                if index > self._stage:
                    self._stage = index
                    self._set(phase=phase, message=message)
                return

    def _run(self, video: Path, postprocess: bool, vggt_interval: int):
        try:
            npz = self._infer(video, postprocess, vggt_interval)
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

    def _infer(self, video: Path, postprocess: bool,
               vggt_interval: int) -> Path:
        python = interpreter()
        if python is None:
            raise RuntimeError('FlowHMR Python not configured')
        ckpt = checkpoint()

        cmd = [str(python), 'run_v2m_demo_generation.py', str(video),
               '--ckpt', str(ckpt), '--vggt_interval', str(max(1, vggt_interval))]
        if postprocess:
            cmd.append('--postprocess')

        if needs_overwrite(video):
            cmd.append('--overwrite')

        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        # the pipeline prints arrows (U+2192); a cp1252 console would abort the
        # run mid-stage with UnicodeEncodeError, so force UTF-8 on the child
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        env['PYTHONPATH'] = str(FLOWHMR) + os.pathsep + env.get('PYTHONPATH', '')

        self._set(phase='starting', message='loading models')
        log_tail: list = []
        if self._cancel.is_set():
            raise RuntimeError('cancelled')
        proc = subprocess.Popen(cmd, cwd=str(FLOWHMR), env=env,
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
            raise RuntimeError('FlowHMR failed:\n' + '\n'.join(log_tail[-8:]))

        found = self._output_npz(video, ckpt)
        if found is None:
            raise RuntimeError(
                f'FlowHMR produced no motion file for {video.name} - see the '
                f'log above')
        return found

    def _output_npz(self, video: Path, ckpt: Path) -> Optional[Path]:
        folder = OUTPUT / video.stem
        preferred = folder / f'{ckpt.stem}.npz'
        if preferred.exists():
            return preferred
        if not folder.is_dir():
            return None
        # the tag follows the checkpoint, so any npz that is not one of the
        # cached feature dumps is the motion
        candidates = [
            p for p in folder.glob('*.npz')
            if not p.name.endswith(('_bbox.npz', '_vggt_camera.npz'))
        ]
        return max(candidates, key=lambda p: p.stat().st_mtime) \
            if candidates else None
