"""MuseTalk v1.5 lip-sync jobs.

MuseTalk cannot run inside this app's interpreter: it pins numpy 1.x and a
torch/diffusers stack, while the app runs on Python 3.13. So a job is a
subprocess in MuseTalk's own Python, driven by a generated realtime config.

The generated clip lands in ``videos/`` where the normal pose source picker
can play it; the existing MediaPipe face layer then drives the VRM mouth from
it, which is what gives the avatar lip-synced speech.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

ROOT = Path(__file__).resolve().parent.parent
MUSETALK = ROOT / 'MuseTalk'
MODELS = MUSETALK / 'models'
VIDEOS = ROOT / 'videos'
IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

# Files the v1.5 realtime path loads (see scripts/setup_musetalk.py).
REQUIRED = [
    'musetalkV15/unet.pth',
    'musetalkV15/musetalk.json',
    'sd-vae/config.json',
    'sd-vae/diffusion_pytorch_model.bin',
    'whisper/config.json',
    'whisper/pytorch_model.bin',
    'whisper/preprocessor_config.json',
    'face-parse-bisent/79999_iter.pth',
    'face-parse-bisent/resnet18-5c106cde.pth',
]

JOB_TIMEOUT = 30 * 60
# MuseTalk preprocesses (face detect + VAE encode) every identity frame, which
# is ~0.5 s each. A photo avatar is the intended mode; longer clips are allowed
# up to this many frames rather than silently taking an hour.
MAX_IDENTITY_FRAMES = 900

# Scratch space for the transcode / trim / captured frame. MuseTalk needs real
# files on disk, but they are throwaway - keep them out of the project and
# wipe them between jobs.
WORK = Path(tempfile.gettempdir()) / 'rtmlib_lipsync'


def reset_work() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)


def work_path(name: str) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    return WORK / name


def clean_work() -> None:
    shutil.rmtree(WORK, ignore_errors=True)


def interpreter() -> Optional[Path]:
    """The Python that has MuseTalk's dependencies (written by setup)."""
    marker = MUSETALK / '.musetalk_python'
    if marker.exists():
        found = Path(marker.read_text(encoding='utf-8').strip())
        if found.exists():
            return found
    env = os.environ.get('MUSETALK_PYTHON')
    if env and Path(env).exists():
        return Path(env)
    for candidate in (MUSETALK / '.venv' / 'Scripts' / 'python.exe',
                      MUSETALK / '.venv' / 'bin' / 'python'):
        if candidate.exists():
            return candidate
    return None


def identity_info(path: Path):
    """Frame count and fps of a video (0, 0 for anything else)."""
    if path.suffix.lower() in IMAGE_EXT:
        return 0, 0.0
    capture = cv2.VideoCapture(str(path))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    return frames, fps


def trim_clip(source: Path, seconds: float) -> Path:
    """Cut the first `seconds` of a clip (video + audio) for a shorter bake."""
    out = work_path(f'identity_{time.strftime("%H%M%S")}.mp4')
    result = subprocess.run(
        ['ffmpeg', '-y', '-v', 'error', '-i', str(source),
         '-t', f'{seconds:.2f}', '-c', 'copy', str(out)],
        capture_output=True, text=True)
    if result.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        return source
    return out


def fit_identity(face: Path, audio: Path):
    """Keep the identity under the cap; long clips are trimmed to fit.

    MuseTalk preprocesses every identity frame (~0.5 s each), so a long video
    is cut to the first MAX_IDENTITY_FRAMES worth and the audio follows it -
    the baked animation then covers that span.
    """
    frames, fps = identity_info(face)
    if frames <= MAX_IDENTITY_FRAMES:
        return face, audio
    seconds = max(1.0, (MAX_IDENTITY_FRAMES - 10) / max(1.0, fps))
    trimmed = trim_clip(face, seconds)
    if trimmed is face:
        raise RuntimeError(
            f'identity video has {frames} frames and could not be trimmed')
    print(f'[lipsync] trimmed identity to {seconds:.1f}s '
          f'({MAX_IDENTITY_FRAMES} frames)', flush=True)
    return trimmed, trimmed


def missing_weights() -> list:
    return [name for name in REQUIRED if not (MODELS / name).exists()]


def to_wav(source: Path, prefix: str = 'clip') -> Path:
    """Transcode anything MuseTalk is handed into 16 kHz mono WAV.

    MuseTalk's audio path expects a plain PCM wav; browser recordings arrive
    as webm/opus, so normalising here keeps every entry point equal. The file
    lives in the scratch dir and is deleted when the job ends.
    """
    out = work_path(f'{prefix}_{time.strftime("%H%M%S")}.wav')
    result = subprocess.run(
        ['ffmpeg', '-y', '-v', 'error', '-i', str(source),
         '-vn', '-ac', '1', '-ar', '16000', str(out)],
        capture_output=True, text=True)
    if result.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        raise RuntimeError('no usable audio track in that clip')
    return out


def extract_source_audio(source: Path) -> Path:
    """Pull the audio track out of a video as 16 kHz mono WAV."""
    return to_wav(source, 'source')


def availability() -> dict:
    """What the UI needs to know before offering the feature."""
    if not (MUSETALK / 'scripts' / 'realtime_inference.py').exists():
        return {'ok': False, 'reason': f'MuseTalk checkout not found in '
                                      f'{MUSETALK}'}
    python = interpreter()
    if python is None:
        return {'ok': False,
                'reason': 'MuseTalk Python not configured - run '
                          'scripts/setup_musetalk.bat'}
    missing = missing_weights()
    if missing:
        return {'ok': False,
                'reason': f'{len(missing)} weight files missing - run '
                          'scripts/setup_musetalk.bat'}
    return {'ok': True, 'reason': 'ready', 'python': str(python)}


class LipSyncJob:
    """One MuseTalk run, tracked so the UI can poll it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._proc = None
        self._cancel = threading.Event()
        self._status = {
            'state': 'idle',
            'phase': '',
            'message': '',
            'output': '',
            'error': '',
            'started': 0.0,
            'elapsed': 0.0,
        }

    # ------------------------------------------------------------------ state
    def status(self) -> dict:
        with self._lock:
            data = dict(self._status)
        if data['state'] == 'running' and data['started']:
            data['elapsed'] = round(time.time() - data['started'], 1)
        return data

    def _set(self, **fields):
        with self._lock:
            self._status.update(fields)

    # ------------------------------------------------------------------ start
    def start(self, audio: Path, face: Path) -> dict:
        check = availability()
        if not check['ok']:
            raise RuntimeError(check['reason'])
        if not audio.exists():
            raise FileNotFoundError(f'audio not found: {audio}')
        if not face.exists():
            raise FileNotFoundError(f'face not found: {face}')
        if audio.suffix.lower() != '.wav':
            # browser mic recordings are webm/opus
            audio = to_wav(audio, 'voice')
        with self._lock:
            if self._status['state'] == 'running':
                raise RuntimeError('a lip-sync job is already running')
            self._cancel.clear()
            self._status.update(state='running', phase='starting',
                                message='loading models', output='', error='',
                                started=time.time(), elapsed=0.0)
        self._thread = threading.Thread(target=self._run, args=(audio, face),
                                        daemon=True)
        self._thread.start()
        return self.status()

    # ------------------------------------------------------------------- work
    def _run(self, audio: Path, face: Path):
        try:
            output = self._infer(audio, face)
            if output is None:
                raise RuntimeError('MuseTalk produced no video - see the log')
            stamp = time.strftime('%Y%m%d-%H%M%S')
            VIDEOS.mkdir(parents=True, exist_ok=True)
            final = VIDEOS / f'lipsync_{stamp}.mp4'
            shutil.copyfile(output, final)
            self._set(state='done', phase='done',
                      message=f'{final.name} ({final.stat().st_size / 1e6:.1f} MB)',
                      output=f'videos/{final.name}')
        except Exception as exc:  # noqa: BLE001
            if self._cancel.is_set():
                self._set(state='cancelled', phase='cancelled', error='')
            else:
                self._set(state='error', phase='failed', error=str(exc))
        finally:
            clean_work()

    def busy(self) -> bool:
        with self._lock:
            return self._status['state'] == 'running'

    def cancel(self) -> dict:
        """Stop a running job (MuseTalk saturates the GPU)."""
        with self._lock:
            running = self._status['state'] == 'running'
            if running:
                self._cancel.set()
                self._status.update(phase='cancelling',
                                    message='stopping MuseTalk')
            proc = self._proc
        if running and proc is not None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        return self.status()

    def _infer(self, audio: Path, face: Path) -> Optional[Path]:
        python = interpreter()
        # one cached avatar directory per distinct identity, so preprocessing
        # runs once. The signature avoids hashing a whole video.
        stat = face.stat()
        signature = f'{face.name}:{stat.st_size}:{int(stat.st_mtime)}'
        digest = hashlib.md5(signature.encode()).hexdigest()[:10]
        avatar_id = f'app_{digest}'

        # MuseTalk reads a video with cv2.VideoCapture, or a directory of PNGs.
        # OpenCV 5 cannot open a still through VideoCapture, so an image is
        # handed over as a one-file directory instead.
        if face.suffix.lower() in IMAGE_EXT:
            source_dir = MUSETALK / 'results' / 'v15' / 'faces' / avatar_id
            source_dir.mkdir(parents=True, exist_ok=True)
            target = source_dir / '00000000.png'
            if not target.exists():
                shutil.copyfile(face, target)
            identity = source_dir
        else:
            identity = face
            capture = cv2.VideoCapture(str(face))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            capture.release()
            if frames > MAX_IDENTITY_FRAMES:
                raise RuntimeError(
                    f'identity video has {frames} frames and MuseTalk '
                    f'preprocesses each one; use a still frame (or a clip '
                    f'under {MAX_IDENTITY_FRAMES} frames)')

        avatar_dir = MUSETALK / 'results' / 'v15' / 'avatars' / avatar_id
        # latents.pt is the last artefact prepare_material writes, so its
        # presence is what "this face is prepared" really means
        preparation = not (avatar_dir / 'latents.pt').exists()
        if preparation and avatar_dir.exists():
            # a half-finished previous run: MuseTalk would stop on an input()
            # prompt ("re-create?") which nobody can answer in a subprocess
            shutil.rmtree(avatar_dir, ignore_errors=True)

        config = MUSETALK / 'configs' / 'inference' / '_app_job.yaml'
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            f'{avatar_id}:\n'
            f'  preparation: {str(preparation).lower()}\n'
            f'  bbox_shift: 0\n'
            f'  video_path: "{identity.as_posix()}"\n'
            f'  audio_clips:\n'
            f'    out: "{audio.as_posix()}"\n',
            encoding='utf-8')

        cmd = [
            str(python), 'scripts/realtime_inference.py',
            '--version', 'v15',
            '--vae_type', 'sd-vae',
            '--unet_config', './models/musetalkV15/musetalk.json',
            '--unet_model_path', './models/musetalkV15/unet.pth',
            '--whisper_dir', './models/whisper',
            '--inference_config', str(config),
            '--result_dir', './results',
            '--fps', '25',
            '--batch_size', '20',
            '--output_vid_name', 'out',
        ]
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        # running scripts/realtime_inference.py puts scripts/ on sys.path, so
        # the repo root has to be importable for `import musetalk` to work
        env['PYTHONPATH'] = str(MUSETALK) + os.pathsep + env.get('PYTHONPATH', '')

        self._set(phase='preparing face' if preparation else 'loading models',
                  message='first run per face is slower')
        log_tail = []
        if self._cancel.is_set():
            raise RuntimeError('cancelled')
        proc = subprocess.Popen(cmd, cwd=str(MUSETALK), env=env,
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding='utf-8', errors='replace')
        self._proc = proc
        if self._cancel.is_set():
            proc.kill()
        # the loop only advances when a line arrives, so a silent hang needs
        # its own timer rather than a deadline check inside the loop
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
                low = line.lower()
                if 'creating avator' in low or 'face detect' in low:
                    self._set(phase='preparing face',
                              message='detecting face landmarks')
                elif 'start inference' in low:
                    self._set(phase='generating', message='synthesising mouth')
                elif 'result is save to' in low:
                    self._set(phase='rendering video', message='muxing audio')
        finally:
            watchdog.cancel()
            proc.wait()
            self._proc = None

        if proc.returncode != 0:
            raise RuntimeError('MuseTalk failed:\n' + '\n'.join(log_tail[-8:]))
        output = avatar_dir / 'vid_output' / 'out.mp4'
        return output if output.exists() else None
