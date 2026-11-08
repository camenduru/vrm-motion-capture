/**
 * Unit tests for the tracker -> VRM conversion.
 *
 * The point of this file is that none of the conversion needs a camera, a
 * browser or an avatar: landmarks, ARKit scores and a head matrix go in, VRM
 * weights and a gaze vector come out. `face.js` only wraps MediaPipe around it.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  faceFrameToVrm,
  gazeFromLandmarks,
  mapBlendshapes,
  quaternionFromMatrix,
  readBlendshapes,
} from '../src/face-retarget.js';

/** A 478-point landmark array with the two eyes positioned by hand. */
function syntheticFace({ irisX = 0.55, irisY = 0.52 } = {}) {
  const points = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const set = (index, x, y) => { points[index] = { x, y, z: 0 }; };

  // left eye: corners at 0.4 / 0.6, mid 0.5, half width 0.1
  set(33, 0.6, 0.5);
  set(133, 0.4, 0.5);
  set(468, irisX, irisY);
  // right eye: corners mirrored around the same mid
  set(263, 0.4, 0.5);
  set(362, 0.6, 0.5);
  set(473, irisX, irisY);
  return points;
}

test('readBlendshapes accepts a Map and a plain object', () => {
  const map = new Map([['jawOpen', 0.4]]);
  assert.equal(readBlendshapes(map)('jawOpen'), 0.4);
  assert.equal(readBlendshapes({ jawOpen: 0.2 })('jawOpen'), 0.2);
  assert.equal(readBlendshapes({})('missing'), 0);
  assert.equal(readBlendshapes(null)('jawOpen'), 0);
});

test('a resting face produces no expression', () => {
  // every channel sits at the ARKit noise floor
  const blend = new Map([
    ['jawOpen', 0.03], ['mouthSmileLeft', 0.04], ['mouthSmileRight', 0.05],
    ['browDownLeft', 0.02], ['browDownRight', 0.03], ['eyeBlinkLeft', 0.01],
  ]);
  const out = mapBlendshapes(blend, { mouthRef: 0.3 });
  for (const [name, value] of Object.entries(out)) {
    assert.ok(value < 0.06, `${name} should be at rest, got ${value}`);
  }
});

test('jaw open drives the mouth presets and not the emotion presets', () => {
  const out = mapBlendshapes({ jawOpen: 0.3 }, { mouthRef: 0.3 });
  assert.ok(out.aa > 0.8, `aa ${out.aa}`);
  assert.ok(out.happy < 0.01 && out.angry < 0.01);
  assert.ok(out.oh < 0.2, 'a relaxed jaw should not round the mouth');
});

test('pucker and funnel shape the jaw into ou / oh', () => {
  const out = mapBlendshapes({
    jawOpen: 0.2, mouthPucker: 0.8, mouthFunnel: 0.7,
  }, { mouthRef: 0.3 });
  assert.ok(out.ou > out.aa, `ou ${out.ou} vs aa ${out.aa}`);
  assert.ok(out.oh > 0.2, `oh ${out.oh}`);
});

test('smile, brow and eye scores reach the VRM emotion presets', () => {
  const out = mapBlendshapes({
    mouthSmileLeft: 0.8, mouthSmileRight: 0.8,
    browDownLeft: 0.8, browDownRight: 0.8,
    browInnerUp: 0.9, eyeWideLeft: 0.4, eyeWideRight: 0.4,
  }, { mouthRef: 0.3 });
  assert.ok(out.happy > 0.5, `happy ${out.happy}`);
  assert.ok(out.angry > 0.5, `angry ${out.angry}`);
  assert.ok(out.surprised > 0.5, `surprised ${out.surprised}`);
});

test('blink is reported per eye and averaged', () => {
  const out = mapBlendshapes({ eyeBlinkLeft: 1, eyeBlinkRight: 0.2 });
  assert.equal(out.blinkLeft, 1);
  assert.ok(out.blinkRight > 0.1 && out.blinkRight < 0.2);
  assert.ok(out.blink > 0.5 && out.blink < 0.6);
});

test('mouth presets are normalised so they cannot overdrive the lips', () => {
  const out = mapBlendshapes({
    jawOpen: 1, mouthPucker: 1, mouthFunnel: 1,
    mouthStretchLeft: 1, mouthStretchRight: 1,
  }, { mouthRef: 0.18 });
  const total = out.aa + out.ih + out.ou + out.ee + out.oh;
  assert.ok(total <= 1.1500001, `mouth total ${total}`);
  for (const value of [out.aa, out.ih, out.ou, out.ee, out.oh]) {
    assert.ok(value >= 0 && value <= 1);
  }
});

test('gaze reads iris offset in image direction and survives aspect', () => {
  const centred = gazeFromLandmarks(syntheticFace({ irisX: 0.5, irisY: 0.5 }));
  assert.ok(Math.abs(centred.x) < 1e-9 && Math.abs(centred.y) < 1e-9);

  const right = gazeFromLandmarks(syntheticFace({ irisX: 0.55, irisY: 0.52 }));
  assert.ok(right.x > 0.4, `x ${right.x}`);
  assert.ok(right.y > 0.3, `y ${right.y}`);

  // A 16:9 frame stretches x, so an eye that is just as tall in pixels reads
  // as narrower vertically once x is measured in the same units.
  const wide = gazeFromLandmarks(syntheticFace({ irisX: 0.55, irisY: 0.52 }),
    16 / 9);
  assert.ok(Math.abs(wide.y) < Math.abs(right.y));
});

test('gaze on a missing face is neutral, not NaN', () => {
  const out = gazeFromLandmarks([], 1.777);
  assert.deepEqual(out, { x: 0, y: 0, left: null, right: null });
});

test('quaternionFromMatrix extracts identity and a known rotation', () => {
  const identity = quaternionFromMatrix([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
  assert.ok(Math.abs(identity[3] - 1) < 1e-9);

  // column-major 90 degrees about +Z
  const half = Math.SQRT1_2;
  const z90 = quaternionFromMatrix([
    0, 1, 0, 0,
    -1, 0, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
  assert.ok(Math.abs(z90[2] - half) < 1e-9, `z ${z90[2]}`);
  assert.ok(Math.abs(z90[3] - half) < 1e-9, `w ${z90[3]}`);
});

test('a tracked frame forwards blendshapes, gaze and head pose', () => {
  const frame = {
    landmarks: syntheticFace({ irisX: 0.6 }),
    blendshapes: new Map([['jawOpen', 0.5]]),
    transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    aspect: 1,
    gaze: { x: 0.5, y: 0.1 },
  };
  const target = faceFrameToVrm(frame);
  assert.equal(target.blend, frame.blendshapes);
  assert.deepEqual(target.gaze, frame.gaze);
  assert.ok(Math.abs(target.q[3] - 1) < 1e-9);
  assert.equal(target.aspect, 1);
});
