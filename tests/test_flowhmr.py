"""Checks for the FlowHMR frame conversion and its preflight.

Plain asserts, no pytest - run it with the app's interpreter:

    .venv\\Scripts\\python.exe tests\\test_flowhmr.py

FlowHMR itself is a separate environment with gated checkpoints, so this covers
the part that must be right regardless of whether the model runs: reading the
npz the pipeline writes, and refusing to start when its files are absent.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import flowhmr_engine as fe  # noqa: E402

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


def write_npz(folder: Path, frames: int = 4, joints: int = 52,
              trans: np.ndarray | None = None, fps: float = 30.0,
              keys: list | None = None) -> Path:
    """A motion npz shaped like the one save_fbx() writes."""
    rng = np.random.default_rng(0)
    ljp = rng.normal(size=(frames, joints, 3)).astype(np.float32) * 0.1
    ljp[:, 0, :] = 0.0                      # pelvis sits at the local origin
    payload = {
        'local_joints_positions': ljp,
        'betas': np.zeros((1, 10), np.float32),
        'poses': np.zeros((frames, 52, 3), np.float32),
        'trans': trans if trans is not None else np.zeros((frames, 3),
                                                          np.float32),
        'mocap_framerate': np.float32(fps),
        'num_frames': np.int32(frames),
    }
    if keys is not None:
        payload = {k: v for k, v in payload.items() if k in keys}
    path = folder / 'clip.npz'
    np.savez(path, **payload)
    return path


@check('joints, fps and times come through')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_npz(Path(tmp), frames=5, fps=25.0)
        out = fe.frames_from_npz(path)
    assert len(out['frames']) == 5, out['frames'].__len__()
    assert out['fps'] == 25.0, out['fps']
    assert [f['t'] for f in out['frames']] == [0.0, 0.04, 0.08, 0.12, 0.16], \
        [f['t'] for f in out['frames']]
    assert abs(out['duration'] - 0.2) < 1e-6, out['duration']
    assert out['truncated'] is False
    assert all(f['detected'] for f in out['frames'])


@check('only the body joints are shipped, not the fingers')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_npz(Path(tmp), joints=52)
        out = fe.frames_from_npz(path)
    assert out['joints'] == 22, out['joints']
    assert len(out['frames'][0]['joints3d']) == 22, \
        len(out['frames'][0]['joints3d'])
    assert len(out['frames'][0]['joints3d'][0]) == 3


@check('the root offset undoes the 180 degree Y rotation save_fbx applies')
def _():
    # save_fbx rotates `trans` by 180 deg about Y but leaves
    # local_joints_positions alone, so x and z must be flipped back
    trans = np.array([[1.0, 0.0, 2.0], [1.1, 0.0, 2.0]], np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_npz(Path(tmp), frames=2, trans=trans)
        out = fe.frames_from_npz(path)
    first, second = out['frames'][0]['root'], out['frames'][1]['root']
    assert abs(first[0]) < 1e-6 and abs(first[2]) < 1e-6, \
        f'the first frame is the reference, got {first}'
    assert abs(second[0] + 0.1) < 1e-5, f'x delta should be -0.1, got {second}'
    assert abs(second[2]) < 1e-6, second


@check('a missing local_joints_positions is named, not silently empty')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_npz(Path(tmp), keys=['poses', 'trans', 'betas'])
        try:
            fe.frames_from_npz(path)
        except RuntimeError as exc:
            message = str(exc)
            assert 'local_joints_positions' in message, message
            assert 'poses' in message, f'keys should be listed: {message}'
        else:
            raise AssertionError('expected a RuntimeError')


@check('long clips are truncated instead of shipped whole')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_npz(Path(tmp), frames=12)
        out = fe.frames_from_npz(path, max_frames=5)
    assert len(out['frames']) == 5, len(out['frames'])
    assert out['truncated'] is True


@check('a nonsense fps falls back to 30')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_npz(Path(tmp), fps=0.0)
        out = fe.frames_from_npz(path)
    assert out['fps'] == 30.0, out['fps']


@check('preflight walks checkout -> python -> files, in that order')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        saved = (fe.FLOWHMR, fe.OUTPUT)
        fe.FLOWHMR = root / 'flowhmr'
        fe.OUTPUT = fe.FLOWHMR / 'output'
        try:
            missing_repo = fe.availability()
            assert not missing_repo['ok']
            assert 'checkout not found' in missing_repo['reason'], \
                missing_repo['reason']

            fe.FLOWHMR.mkdir(parents=True)
            (fe.FLOWHMR / 'run_v2m_demo_generation.py').write_text('')
            no_python = fe.availability()
            assert not no_python['ok']
            assert 'Python not configured' in no_python['reason'], \
                no_python['reason']

            interpreter = fe.FLOWHMR / '.venv' / 'Scripts' / 'python.exe'
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text('')
            missing = fe.availability()
            assert not missing['ok']
            assert len(missing['missing']) == len(fe.REQUIRED) + 1, \
                missing['missing']
            assert any('flowhmr_latest.ckpt' in name
                       for name in missing['missing']), missing['missing']
        finally:
            fe.FLOWHMR, fe.OUTPUT = saved


@check('the motion npz is found next to the cached feature dumps')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        saved = (fe.FLOWHMR, fe.OUTPUT)
        fe.FLOWHMR = root
        fe.OUTPUT = root / 'output'
        try:
            folder = fe.OUTPUT / 'clip'
            folder.mkdir(parents=True)
            # the cache dumps are written last and are therefore the newest
            # files in the folder - they must still not be mistaken for motion
            (folder / 'clip_0_seed0.npz').write_bytes(b'')
            for name in ('clip_bbox.npz', 'clip_vggt_camera.npz'):
                (folder / name).write_bytes(b'')
                os.utime(folder / name, (2e9, 2e9))
            job = fe.FlowHMRJob()
            found = job._output_npz(Path('clip.mp4'), Path('flowhmr_latest.ckpt'))
            assert found is not None and found.name == 'clip_0_seed0.npz', found

            (folder / 'flowhmr_latest.npz').write_bytes(b'')
            found = job._output_npz(Path('clip.mp4'), Path('flowhmr_latest.ckpt'))
            assert found.name == 'flowhmr_latest.npz', found
        finally:
            fe.FLOWHMR, fe.OUTPUT = saved


@check('a newer clip invalidates every cached artefact, not just features')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        saved = (fe.FLOWHMR, fe.OUTPUT)
        fe.FLOWHMR = root
        fe.OUTPUT = root / 'output'
        try:
            video = root / 'videos' / 'clip.mp4'
            video.parent.mkdir(parents=True)
            video.write_bytes(b'')
            assert fe.needs_overwrite(video) is False, \
                'no cache yet, so nothing to overwrite'

            folder = fe.OUTPUT / 'clip'
            folder.mkdir(parents=True)
            # only the camera cache is stale and only features used to be
            # checked - the camera is the one that silently ruins a re-bake
            for name, when in (('clip_sam3d_feat.pt', 2e9),
                               ('clip_vggt_camera.npz', 1e9)):
                (folder / name).write_bytes(b'')
                os.utime(folder / name, (when, when))
            os.utime(video, (1.5e9, 1.5e9))
            assert fe.needs_overwrite(video) is True
        finally:
            fe.FLOWHMR, fe.OUTPUT = saved


@check('the runner reports its stages as the pipeline prints them')
def _():
    job = fe.FlowHMRJob()
    for line, phase in (
            ('  [preprocess] Transcoded: x_30fps.mp4', 'reading'),
            ('  === Step 1: Bbox detection', 'detecting'),
            ('  [detection] Auto-selected track 3', 'detecting'),
            ('  === Step 2: VGGT Camera', 'camera'),
            ('  [vggt] sampled 12/40 frames (interval=4)', 'camera'),
            ('  === Step 3: SAM features', 'features'),
            ('  [sam] saved feat.pt', 'features'),
            ('  === Step 4: Generation', 'generating'),
            ('  [generation] T=40, range=[0,40]', 'generating'),
    ):
        job._set(phase='')
        job._stage = -1
        job._phase_of(line)
        assert job.status()['phase'] == phase, \
            f'{line!r} -> {job.status()["phase"]!r}, expected {phase!r}'

    # generation re-prints the cached camera line; it must not walk the UI back
    job._phase_of('  [vggt] skip cached camera: x.npz')
    assert job.status()['phase'] == 'generating', job.status()['phase']
    job._phase_of('  === Step 2: VGGT Camera')
    assert job.status()['phase'] == 'generating', job.status()['phase']


@check('a finished job carries its frames; a failed one carries the error')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_npz(Path(tmp), frames=3)
        job = fe.FlowHMRJob()
        job._infer = lambda *a, **k: path      # no model, no subprocess
        job._run(Path('clip.mp4'), True, 1)
        status = job.status()
        assert status['state'] == 'done', status
        assert status['frames'] == 3, status
        assert len(job.result()['frames']) == 3

        broken = fe.FlowHMRJob()

        def boom(*_a, **_k):
            raise RuntimeError('missing files')

        broken._infer = boom
        broken._run(Path('clip.mp4'), True, 1)
        assert broken.status()['state'] == 'error', broken.status()
        assert 'missing files' in broken.status()['error']

        stopped = fe.FlowHMRJob()

        def cancelled(*_a, **_k):
            stopped._cancel.set()
            raise RuntimeError('killed')

        stopped._infer = cancelled
        stopped._run(Path('clip.mp4'), True, 1)
        assert stopped.status()['state'] == 'cancelled', stopped.status()


print('\n'.join(results))
print('\nFlowHMR frames: ' + ('FAILED' if FAILED else 'ok'))
sys.exit(1 if FAILED else 0)
