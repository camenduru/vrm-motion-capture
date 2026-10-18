"""Download the ONNX pose-estimation models used by the mocap app.

rtmlib normally downloads checkpoints into a cache directory. We instead
vendor the two model sets it needs into ``models/`` so the app runs fully
offline and the model files live with the project.

Usage:
    python scripts/download_models.py [--set performance] [--set balanced]
                                      [--set lightweight] [--set face]
"""

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / 'models'

# OpenMMLab's download host is often rate-limited; the project's Hugging Face
# mirror hosts identical checkpoints.
BASE = 'https://huggingface.co/Tau-J/RTMPose/resolve/main/rtmposev1/onnx_sdk/'

# MediaPipe face landmarker: 478 landmarks, 52 blendshapes and a head-pose
# matrix. Ships as a .task bundle rather than a zip of .onnx files.
FACE_MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/'
                  'face_landmarker/face_landmarker/float16/1/'
                  'face_landmarker.task')

MODEL_SETS = {
    'performance': [
        ('yolox_x_8xb8-300e_humanart-a39d44ed.zip',
         'yolox_x_humanart.onnx'),
        ('rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',
         'rtmpose_x_body7_384x288.onnx'),
    ],
    'balanced': [
        ('yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
         'yolox_m_humanart.onnx'),
        ('rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
         'rtmpose_m_body7_256x192.onnx'),
    ],
    'lightweight': [
        ('yolox_tiny_8xb8-300e_humanart-6f3252f9.zip',
         'yolox_tiny_humanart.onnx'),
        ('rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip',
         'rtmpose_s_body7_256x192.onnx'),
    ],
}


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + '.part')
    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get('Content-Length') or 0)
        with open(tmp, 'wb') as f, tqdm(
                total=total, unit='B', unit_scale=True, unit_divisor=1024,
                desc=dst.name) as bar:
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dst)


def extract_onnx(zip_path: Path, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmpdir)
        found = None
        for path in Path(tmpdir).rglob('*.onnx'):
            found = path
            break
        if found is None:
            raise RuntimeError(f'No .onnx file inside {zip_path.name}')
        shutil.move(str(found), dest)


def fetch_model(filename: str, output: str) -> None:
    dest = MODELS_DIR / output
    if dest.exists() and dest.stat().st_size > 0:
        print(f'[skip] {output} already present '
              f'({dest.stat().st_size / 1e6:.1f} MB)')
        return

    print(f'[get ] {filename}')
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / filename
        download(BASE + filename, archive)
        print(f'[unzip] -> {output}')
        extract_onnx(archive, dest)
    print(f'[ ok ] {output} ({dest.stat().st_size / 1e6:.1f} MB)')


def fetch_url(url: str, output: str) -> None:
    """Download a single file (the MediaPipe bundles are not zips)."""
    dest = MODELS_DIR / output
    if dest.exists() and dest.stat().st_size > 0:
        print(f'[skip] {output} already present '
              f'({dest.stat().st_size / 1e6:.1f} MB)')
        return
    print(f'[get ] {url.rsplit("/", 1)[-1]}')
    download(url, dest)
    print(f'[ ok ] {output} ({dest.stat().st_size / 1e6:.1f} MB)')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set',
                        action='append',
                        choices=sorted(list(MODEL_SETS) + ['face']),
                        help='Model set to download (default: all sets)')
    args = parser.parse_args()

    selected = args.set or list(MODEL_SETS) + ['face']
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name in selected:
        print(f'== {name} ==')
        if name == 'face':
            fetch_url(FACE_MODEL_URL, 'face_landmarker.task')
            continue
        for filename, output in MODEL_SETS[name]:
            fetch_model(filename, output)

    print(f'\nModels are in {MODELS_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
