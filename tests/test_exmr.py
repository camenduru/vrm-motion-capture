"""Checks for the ExMR frame conversion and its preflight.

Plain asserts, no pytest - run it with the app's interpreter:

    .venv\\Scripts\\python.exe tests\\test_exmr.py

ExMR itself is a separate environment with gated checkpoints, so this covers
the part that must be right regardless of whether the model runs: reading the
npz the pipeline writes, never letting NaN into the JSON payload, and refusing
to start when its files are absent.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import exmr_engine as ee  # noqa: E402

results = []


def check(name):
    def wrap(fn):
        try:
            fn()
            results.append(f'  PASS  {name}')
        except AssertionError as exc:
            results.append(f'  FAIL  {name}\n        {exc}')
            globals()['FAILED'] = True
        return fn
    return wrap


FAILED = False


def write_npz(folder: Path, frames: int = 4, joints: int = 76,
              fps: float = 30.0, keys: list | None = None,
              bad: tuple = ()) -> Path:
    """A results npz shaped like ExMR's save_results() output."""
    rng = np.random.default_rng(0)
    world = rng.normal(size=(frames, joints, 3)).astype(np.float32) * 0.1
    world[:, 0, :] = 0.0                    # pelvis sits at the origin
    for index in bad:
        world[index, 3] = np.nan
    payload = {
        'joints_global': world,
        'joints_incam': world.copy(),
        'fps': np.float64(fps),
        'joint_names': np.array([f'j{i}' for i in range(joints)]),
    }
    if keys is not None:
        payload = {k: v for k, v in payload.items() if k in keys}
    path = folder / 'results.npz'
    np.savez(path, **payload)
    return path


@check('joints, fps and times come through')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_npz(Path(tmp), frames=5, fps=25.0)
        out = ee.frames_from_npz(path)
    assert len(out['frames']) == 5, len(out['frames'])
    assert out['fps'] == 25.0, out['fps']
    assert [f['t'] for f in out['frames']] == [0.0, 0.04, 0.08, 0.12, 0.16], \
        [f['t'] for f in out['frames']]
    assert abs(out['duration'] - 0.2) < 1e-6, out['duration']
    assert out['truncated'] is False
    assert all(f['detected'] for f in out['frames'])


@check('only the body joints are shipped, not the face or fingers')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_npz(Path(tmp), joints=76)
        out = ee.frames_from_npz(path)
    assert out['joints'] == 22, out['joints']
    assert len(out['frames'][0]['joints3d']) == 22, \
        len(out['frames'][0]['joints3d'])
    assert len(out['frames'][0]['joints3d'][0]) == 3


@check('NaN frames are filled and the payload stays valid JSON')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_npz(Path(tmp), frames=4, bad=(1,))
        out = ee.frames_from_npz(path)
    text = json.dumps(out, allow_nan=False)
    assert 'NaN' not in text
    assert out['frames'][1]['detected'] is False
    assert out['frames'][1]['joints3d'] == out['frames'][0]['joints3d']


@check('a missing joints_global is named, not silently empty')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_npz(Path(tmp), keys=['joints_incam', 'fps'])
        try:
            ee.frames_from_npz(path)
        except RuntimeError as exc:
            message = str(exc)
            assert 'joints_global' in message, message
            assert 'joints_incam' in message, f'keys should be listed: {message}'
        else:
            raise AssertionError('expected a RuntimeError')


@check('long clips are truncated instead of shipped whole')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_npz(Path(tmp), frames=12)
        out = ee.frames_from_npz(path, max_frames=5)
    assert len(out['frames']) == 5, len(out['frames'])
    assert out['truncated'] is True


@check('a nonsense fps falls back to 30')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_npz(Path(tmp), fps=0.0)
        out = ee.frames_from_npz(path)
    assert out['fps'] == 30.0, out['fps']


@check('preflight walks checkout -> python -> files, in that order')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        saved = ee.EXMR
        ee.EXMR = root / 'ExMR'
        try:
            missing_repo = ee.availability()
            assert not missing_repo['ok']
            assert 'checkout not found' in missing_repo['reason'], \
                missing_repo['reason']

            (ee.EXMR / 'tools').mkdir(parents=True)
            (ee.EXMR / 'tools' / 'demo.py').write_text('')
            no_python = ee.availability()
            assert not no_python['ok']
            assert 'Python not configured' in no_python['reason'], \
                no_python['reason']

            interpreter = ee.EXMR / '.venv' / 'Scripts' / 'python.exe'
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text('')
            missing = ee.availability()
            assert not missing['ok']
            assert len(missing['missing']) == len(ee.REQUIRED), \
                missing['missing']
            assert any('exmr.safetensors' in name
                       for name in missing['missing']), missing['missing']
        finally:
            ee.EXMR = saved


@check('app results live under outputs/_rtmlib/<stem>')
def _():
    saved = ee.JOB_DIR
    ee.JOB_DIR = Path('out')
    try:
        assert ee.output_dir(Path('clip.mp4')) == Path('out') / 'clip'
    finally:
        ee.JOB_DIR = saved


@check('a newer clip clears only the app-owned cache and results')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / 'clip'
        cache = folder / 'cache'
        cache.mkdir(parents=True)
        video = Path(tmp) / 'clip.mp4'
        video.write_bytes(b'')
        for name in ('results.npz', 'results.pt', 'run.json'):
            (folder / name).write_bytes(b'')
        (cache / 'tracks.pt').write_bytes(b'')
        for path in folder.rglob('*'):
            if path.is_file():
                os.utime(path, (1e9, 1e9))
        os.utime(video, (2e9, 2e9))

        assert ee.clear_stale_cache(video, folder) is True
        assert not cache.exists()
        assert not (folder / 'results.npz').exists()


@check('an already fresh run keeps its cache')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / 'clip'
        cache = folder / 'cache'
        cache.mkdir(parents=True)
        video = Path(tmp) / 'clip.mp4'
        video.write_bytes(b'')
        (folder / 'results.npz').write_bytes(b'')
        os.utime(video, (1e9, 1e9))
        for path in folder.rglob('*'):
            if path.is_file():
                os.utime(path, (2e9, 2e9))
        assert ee.clear_stale_cache(video, folder) is False
        assert cache.exists() and (folder / 'results.npz').exists()


@check('the runner reports its stages as the pipeline prints them')
def _():
    job = ee.ExMRJob()
    for line, phase in (
            ('[12:00:00] Extracting frames to out/frames', 'reading'),
            ('[12:00:01] Computing tracks', 'tracking'),
            ('[12:00:02] Computing keypoints', 'keypoints'),
            ('[12:00:03] Computing body_features', 'body'),
            ('[12:00:04] Computing hand_features', 'hands'),
            ('[12:00:05] Computing camera_rotations', 'camera'),
            ('[12:00:06] Running ExMR', 'generating'),
    ):
        job._set(phase='')
        job._stage = -1
        job._phase_of(line)
        assert job.status()['phase'] == phase, \
            f'{line!r} -> {job.status()["phase"]!r}, expected {phase!r}'

    # a cached-stage line and an early step must not walk the UI back
    job._phase_of('Using cached keypoints.pt')
    assert job.status()['phase'] == 'generating', job.status()['phase']
    job._phase_of('[12:00:06] Computing tracks')
    assert job.status()['phase'] == 'generating', job.status()['phase']


@check('a finished job carries its frames; a failed one carries the error')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_npz(Path(tmp), frames=3)
        job = ee.ExMRJob()
        job._infer = lambda *a, **k: path      # no model, no subprocess
        job._run(Path('clip.mp4'), True)
        status = job.status()
        assert status['state'] == 'done', status
        assert status['frames'] == 3, status
        assert len(job.result()['frames']) == 3

        broken = ee.ExMRJob()

        def boom(*_a, **_k):
            raise RuntimeError('missing files')

        broken._infer = boom
        broken._run(Path('clip.mp4'), True)
        assert broken.status()['state'] == 'error', broken.status()
        assert 'missing files' in broken.status()['error']

        stopped = ee.ExMRJob()

        def cancelled(*_a, **_k):
            stopped._cancel.set()
            raise RuntimeError('killed')

        stopped._infer = cancelled
        stopped._run(Path('clip.mp4'), True)
        assert stopped.status()['state'] == 'cancelled', stopped.status()


print('\n'.join(results))
print('\nExMR frames: ' + ('FAILED' if FAILED else 'ok'))
sys.exit(1 if FAILED else 0)
