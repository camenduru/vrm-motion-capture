"""Mutation-test tests/test_exmr.py.

Breaks app/exmr_engine.py on purpose, one edit at a time, and asserts the
suite notices. Bytes and mtime are restored exactly.

    .venv\\Scripts\\python.exe tests\\mutate_exmr.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / 'app' / 'exmr_engine.py'
TEST = HERE / 'test_exmr.py'
PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'

MUTATIONS = [
    ('face and finger joints are shipped too',
     'joints = joints[:, :BODY_JOINTS]',
     'joints = joints[:, :]'),
    ('a missing joints key is not reported',
     "if 'joints_global' not in keys:",
     'if False:'),
    ('NaN frames are left in the payload',
     'if not finite.all():',
     'if False:'),
    ('the checkout is not checked first',
     "if not (EXMR / 'tools' / 'demo.py').exists():",
     'if False:'),
    ('app results are not separated by clip',
     'return JOB_DIR / video.stem',
     'return JOB_DIR'),
    ('a stale clip is reused',
     'p.stat().st_mtime < video_time',
     'False'),
    ('the camera stage reports itself as keypoints',
     "(('computing camera_rotations',),",
     "(('computing camera_rotations',), ('keypoints', 'estimating 2D keypoints')),"),
    ('nonsense fps is not replaced',
     'if not 1.0 <= fps <= 240.0:\n        fps = 30.0',
     'if False:\n        fps = 30.0'),
]


def run() -> bool:
    out = subprocess.run([str(PYTHON), str(TEST)], cwd=str(ROOT),
                         capture_output=True, text=True)
    return out.returncode == 0


def main() -> int:
    raw = SRC.read_bytes()
    stat = SRC.stat()
    text = raw.decode('utf-8')
    if not run():
        print('baseline already failing - fix that first')
        return 1

    missed = []
    try:
        for label, find, replace in MUTATIONS:
            if text.count(find) != 1:
                print(f'  SKIP  {label}: pattern matched {text.count(find)}x')
                missed.append(label)
                continue
            SRC.write_bytes(text.replace(find, replace).encode('utf-8'))
            caught = not run()
            print(f'  {"CAUGHT" if caught else "MISSED"}  {label}')
            if not caught:
                missed.append(label)
    finally:
        SRC.write_bytes(raw)
        os.utime(SRC, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    print(f'\n{len(MUTATIONS) - len(missed)}/{len(MUTATIONS)} mutations caught')
    return 1 if missed else 0


if __name__ == '__main__':
    sys.exit(main())
