"""Mutation-test tests/test_flowhmr.py.

Breaks app/flowhmr_engine.py on purpose, one edit at a time, and asserts the
suite notices. Bytes and mtime are restored exactly.

    .venv\\Scripts\\python.exe tests\\mutate_flowhmr.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / 'app' / 'flowhmr_engine.py'
TEST = HERE / 'test_flowhmr.py'
PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'

MUTATIONS = [
    ('root offset keeps the 180 deg rotation',
     'trans = trans * np.array([-1.0, 1.0, -1.0], np.float32)',
     'trans = trans * np.array([1.0, 1.0, 1.0], np.float32)'),
    ('root is absolute instead of a delta',
     'root = trans - trans[0]',
     'root = trans'),
    ('fingers are shipped too',
     'joints = joints[:, :BODY_JOINTS]',
     'joints = joints[:, :]'),
    ('a missing joints key is not reported',
     "if 'local_joints_positions' not in keys:",
     'if False:'),
    ('the checkout is not checked first',
     "if not (FLOWHMR / 'run_v2m_demo_generation.py').exists():",
     'if False:'),
    ('cache dumps count as motion',
     "if not p.name.endswith(('_bbox.npz', '_vggt_camera.npz'))",
     'if True'),
    ('a stale camera cache is reused',
     'p.stat().st_mtime < video_time',
     'False'),
    ('the camera stage reports itself as detection',
     "(('=== step 2', '[vggt]'), ('camera', 'estimating camera motion'))",
     "(('=== step 2', '[vggt]'), ('detecting', 'finding the person'))"),
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
