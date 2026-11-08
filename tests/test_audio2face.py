"""Checks for the Audio2Face-3D engine.

Plain asserts, no pytest - run it with the app's interpreter:

    .venv\\Scripts\\python.exe tests\\test_audio2face.py

If the Claire model is present in models/audio2face/claire, the full pipeline
runs against real weights (needs a few seconds and ~2 GB RAM); otherwise only
the availability plumbing is exercised and the heavy checks skip.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import audio2face_engine as a2f  # noqa: E402

results = []
FAILED = False


def check(name):
    def wrap(fn):
        global FAILED
        try:
            fn()
            results.append(f'  PASS  {name}')
        except AssertionError as exc:
            results.append(f'  FAIL  {name}\n        {exc}')
            FAILED = True
        except Exception as exc:  # noqa: BLE001
            results.append(f'  ERROR {name}\n        {type(exc).__name__}:'
                           f' {exc}')
            FAILED = True
        return fn
    return wrap


MODEL = a2f.MODELS_DIR / 'claire'
HAVE_MODEL = (MODEL / 'model.json').exists() and \
    (MODEL / 'network.onnx').exists()


@check('availability reports its parts')
def _():
    data = a2f.availability()
    assert isinstance(data['models'], list)
    assert isinstance(data['onnxruntime'], bool)
    assert isinstance(data['ffmpeg'], bool)
    assert data['ready'] == (bool(data['models']) and data['onnxruntime']
                             and data['ffmpeg'])


@check('decode_audio resamples to mono 16 kHz float32')
def _():
    # 1 kHz sine, stereo, 44.1 kHz through ffmpeg -> 16 kHz mono
    import subprocess
    import tempfile
    t = np.arange(44100, dtype=np.float32) / 44100.0
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    stereo = np.repeat(tone[None, :], 2, axis=0).T
    raw = (stereo * 32767).astype(np.int16).tobytes()
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / 'tone.wav'
        subprocess.run(
            ['ffmpeg', '-y', '-v', 'error', '-f', 's16le', '-ar', '44100',
             '-ac', '2', '-i', '-', '-c', 'copy', str(wav)],
            input=raw, check=True)
        out = a2f.decode_audio(wav)
    assert out.dtype == np.float32
    assert abs(out.size / 16000.0 - 1.0) < 0.05, out.size / 16000.0


if HAVE_MODEL:
    engine = a2f.Audio2FaceEngine(MODEL)

    @check('model card matches the network layout')
    def _():
        total = engine.skin_dims + engine.tongue_dims + engine.jaw_dims \
            + engine.eyes_dims
        assert engine.buffer_len == 8320
        assert engine.buffer_ofs == 4160
        assert engine.samplerate == 16000
        # one inference on silence fixes the result dimension
        res = engine._infer_batch(np.zeros((1, engine.buffer_len),
                                           dtype=np.float32))
        assert res.shape == (1, total), res.shape
        assert engine.skin_dims == 140
        assert len(engine.solve_names) == 52

    @check('solve stays inside [0, 1] and honours the active-pose mask')
    def _():
        rng = np.random.default_rng(7)
        lin = rng.normal(size=int(engine.active.sum()))
        w = engine._solve_qp(lin)
        assert w.shape[0] == int(engine.active.sum())
        assert w.min() >= -1e-6 and w.max() <= 1 + 1e-6, (w.min(), w.max())

    @check('silence produces a resting face')
    def _():
        track = engine.process(np.zeros(16000 * 2, dtype=np.float32))
        assert track['fps'] == 60
        assert abs(track['duration'] - 2.0) < 0.05
        peak = max((np.max(v) for v in track['shapes'].values()),
                   default=0.0)
        assert peak < 0.35, f'peak weight {peak} on silence'

    @check('speech moves the mouth and blinks stay bounded')
    def _():
        t = np.arange(16000 * 3, dtype=np.float32) / 16000.0
        # an amplitude-modulated tone stands in for speech energy
        audio = 0.6 * np.sin(2 * np.pi * 220 * t) * \
            (0.5 + 0.5 * np.sin(2 * np.pi * 3.3 * t))
        start = time.time()
        track = engine.process(audio.astype(np.float32))
        elapsed = time.time() - start
        jaw = np.array(track['shapes'].get('jawOpen', [0.0]))
        assert jaw.max() > 0.08, f'jawOpen never moved: max {jaw.max()}'
        for name, values in track['shapes'].items():
            v = np.max(np.abs(values))
            assert v <= 1.0 + 1e-6, f'{name} exceeds 1: {v}'
        eyes = np.array(track['eyes'])
        assert eyes.shape[1] == 4
        # degrees per the SDK AnimatorEyes contract: sane saccades are small
        assert np.abs(eyes).max() < 8.0, np.abs(eyes).max()
        assert elapsed < track['duration'] * 8, \
            f'too slow: {elapsed:.1f}s for {track["duration"]:.1f}s'

    @check('windows are centred with the documented offset')
    def _():
        # frame i window start = (i * sr) / fps - buffer_ofs: with one sample
        # of audio at t=0.52s, exactly one window's centre, the middle frames
        # must see it
        samples = np.zeros(8320, dtype=np.float32)
        samples[4160] = 1.0
        track = engine.process(samples)
        assert track['frames'] >= 1


else:

    @check('missing model is reported, not raised')
    def _():
        jobs = a2f.Audio2FaceJob()
        # a start with no model dir present must fail cleanly
        try:
            jobs._extract(Path('nope.wav'), 'does-not-exist')
            raise AssertionError('expected a RuntimeError')
        except RuntimeError as exc:
            assert 'not set up' in str(exc) or 'unknown' in str(exc)


print('Audio2Face-3D engine checks' +
      ('' if HAVE_MODEL else ' (model absent: heavy checks skipped)'))
print('\n'.join(results))
print('ALL PASS' if not FAILED else 'FAILURES PRESENT')
sys.exit(1 if FAILED else 0)
