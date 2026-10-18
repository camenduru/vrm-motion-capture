"""Offline pose extraction for VRMA generation.

This is deliberately separate from the live engine: it opens the video itself,
runs its own tracker/FaceLandmarker over every frame, and returns the per-frame
payloads. The live source, preview, audio and realtime state are never touched.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .face_engine import FaceEngine
from .pose_engine import PoseEngine, _unmirror_labels

# cap so a stray 3-hour video cannot run for an hour
MAX_FRAMES = 6000


def extract(source: Path, mode: str = 'auto', device: str = 'auto',
            face: bool = True, max_frames: int = MAX_FRAMES,
            stop: Optional[threading.Event] = None) -> dict:
    """Run one pass over `source` and return the per-frame pose payloads."""
    engine = PoseEngine(source=str(source), mode=mode, device=device,
                        face=face)
    tracker = engine._build_tracker()

    face_engine = None
    if face:
        candidate = FaceEngine()
        if candidate.open():
            face_engine = candidate

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {source}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 120 else 30.0

    frames = []
    index = 0
    started = time.perf_counter()
    try:
        while index < max_frames:
            if stop is not None and stop.is_set():
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            # the app works on the mirrored frame (selfie view); keep it
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            keypoints = scores = None
            try:
                all_kp, all_scores = tracker(frame)
                keypoints, scores = engine._pick_person(all_kp, all_scores)
                if keypoints is not None:
                    keypoints, scores = _unmirror_labels(keypoints, scores)
            except Exception:  # noqa: BLE001
                keypoints = scores = None

            face_payload = None
            if face_engine is not None:
                try:
                    face_payload = face_engine.process(frame)
                except Exception:  # noqa: BLE001
                    face_payload = None

            frames.append({
                'seq': index,
                't': round(index / fps, 4),
                'fps': fps,
                'detected': keypoints is not None,
                'keypoints': (keypoints /
                              np.array([width, height], np.float32)
                              ).round(5).tolist()
                if keypoints is not None else [],
                'scores': np.round(scores, 4).tolist()
                if scores is not None else [],
                'face': face_payload,
                'width': width,
                'height': height,
            })
            index += 1
    finally:
        cap.release()
        if face_engine is not None:
            face_engine.close()

    return {
        'frames': frames,
        'fps': fps,
        'duration': index / fps,
        'truncated': index >= max_frames,
        'seconds': round(time.perf_counter() - started, 2),
    }


class BakeJob:
    """One offline extraction, tracked so the UI can poll it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._status = {'state': 'idle', 'error': '', 'started': 0.0,
                        'elapsed': 0.0, 'frames': 0, 'result': None}

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

    def start(self, source: Path, mode: str, device: str, face: bool) -> dict:
        with self._lock:
            if self._status['state'] == 'running':
                raise RuntimeError('a VRMA bake is already running')
            self._stop.clear()
            self._status.update(state='running', error='', started=time.time(),
                                elapsed=0.0, frames=0, result=None)
        self._thread = threading.Thread(
            target=self._run, args=(source, mode, device, face), daemon=True)
        self._thread.start()
        return self.status()

    def _run(self, source: Path, mode: str, device: str, face: bool):
        try:
            result = extract(source, mode=mode, device=device, face=face,
                             stop=self._stop)
            with self._lock:
                self._status.update(state='done', result=result,
                                    frames=len(result['frames']))
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._status.update(state='error', error=str(exc))
        finally:
            # the MuseTalk dub is a throwaway input: never leave it behind
            try:
                cache = (Path(__file__).resolve().parent.parent / '.cache')
                if cache in source.parents:
                    source.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    def cancel(self) -> dict:
        self._stop.set()
        return self.status()
