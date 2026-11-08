"""Mutation-test the 3D retarget checks.

A guard that has never been seen to fail is not a guard: this breaks
web/src/mocap.js on purpose, one edit at a time, and asserts that
web/test/mocap3d.test.mjs notices. Restores the exact bytes and the original
mtime (a bumped mtime makes the next vite build think the source changed).

    python web/test/mutate3d.py
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB = HERE.parent
SRC = WEB / 'src' / 'mocap.js'
TEST = HERE / 'mocap3d.test.mjs'

# (label, find, replace) - each must make the test suite fail
MUTATIONS = [
    ('forward axis flipped',
     'this._fwd3.crossVectors(upS, sideS)',
     'this._fwd3.crossVectors(sideS, upS)'),
    ('sides not mirrored',
     "const p = side === 'left' ? 'right' : 'left';",
     'const p = side;'),
    ('root reference ignored',
     'clamp((pelvis.z - this.rootRef.z) * scale, -1.6, 1.6) * weight,',
     'clamp(pelvis.z * scale, -1.6, 1.6) * weight,'),
    ('limbs read the wrong joints',
     'SMPL[`${p}Shoulder`], SMPL[`${p}Elbow`]);',
     'SMPL[`${side}Shoulder`], SMPL[`${side}Elbow`]);'),
    ('elbow aim dropped',
     "aimJ(`${side}LowerArm`, `${side}Hand`,",
     "aimJ(`${side}Hand`, `${side}LowerArm`,"),
]


def run() -> bool:
    out = subprocess.run(['node', str(TEST)], cwd=str(WEB),
                         capture_output=True, text=True)
    return out.returncode == 0


def main() -> int:
    raw = SRC.read_bytes()
    stat = SRC.stat()
    text = raw.decode('utf-8')
    if not run():
        print('baseline already failing - fix that first')
        return 1

    survived = []
    try:
        for label, find, replace in MUTATIONS:
            if text.count(find) != 1:
                print(f'  SKIP  {label}: pattern matched {text.count(find)}x')
                survived.append(label)
                continue
            SRC.write_bytes(text.replace(find, replace).encode('utf-8'))
            caught = not run()
            print(f'  {"CAUGHT" if caught else "MISSED"}  {label}')
            if not caught:
                survived.append(label)
    finally:
        SRC.write_bytes(raw)
        os.utime(SRC, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    print(f'\n{len(MUTATIONS) - len(survived)}/{len(MUTATIONS)} mutations caught')
    return 1 if survived else 0


if __name__ == '__main__':
    sys.exit(main())
