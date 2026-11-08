/**
 * Face tracking -> VRM face, with no SDK and no three.js.
 *
 * This is the piece the GNM webcam puppet (../gnm-webcam-puppet) stops at:
 * MediaPipe reports 478 landmarks, 52 ARKit blendshape scores and a 4x4 head
 * pose, and the puppet feeds those to its own parametric head. A VRM avatar
 * wants the same evidence expressed the other way round:
 *
 *   landmarks  -> gaze            (iris offset normalised by the eye's width)
 *   blendshapes-> expression      (ARKit scores -> VRM preset weights)
 *   transform  -> head quaternion (the head bone follows the face)
 *
 * Everything here is pure and runs under `node --test`, so the conversion can
 * be checked without a camera, a browser or an avatar. `face.js` wraps the
 * MediaPipe runtime; `mocap.js` applies the result to the VRM.
 *
 * The mapping is deliberately conservative. ARKit's 52 channels overlap, and
 * naively forwarding them to the 11 VRM mouth/emotion presets double-drives
 * the jaw and fights the blink channel, which reads as a twitching face.
 */

/** Landmark indices of the two iris rings and the eye corners they sit in. */
export const EYE_LANDMARKS = {
  left: { irisCentre: 468, outerCorner: 33, innerCorner: 133 },
  right: { irisCentre: 473, outerCorner: 263, innerCorner: 362 },
};

/** ARKit coefficients idle around 0.02-0.08 on a resting face. */
const FLOOR = 0.06;

export function clamp(value, min, max) {
  return value < min ? min : (value > max ? max : value);
}

/** Normalises a Map, a plain object, or null into a name -> score getter. */
export function readBlendshapes(source) {
  if (!source) return () => 0;
  if (typeof source.get === 'function') {
    return (name) => source.get(name) ?? 0;
  }
  return (name) => source[name] ?? 0;
}

function soft(value) {
  return value <= FLOOR ? 0 : Math.min(1, (value - FLOOR) / (1 - FLOOR));
}

/**
 * Gaze for one eye.
 *
 * The iris offset is divided by the eye's own half-width, so the readout
 * survives the head moving towards or away from the camera: offset and width
 * scale together. `x` is positive towards the image's right, `y` positive
 * towards the image's bottom (image coordinates, like the landmarks).
 *
 * `aspect` (width / height) is needed because MediaPipe normalises x by width
 * and y by height; without it a wide camera would read a horizontal glance as
 * a much smaller angle than it is.
 */
function eyeGaze(landmarks, indices, aspect) {
  const iris = landmarks[indices.irisCentre];
  const outer = landmarks[indices.outerCorner];
  const inner = landmarks[indices.innerCorner];
  if (!iris || !outer || !inner) return null;

  const midX = (outer.x + inner.x) / 2;
  const midY = (outer.y + inner.y) / 2;
  const dx = (outer.x - inner.x) * aspect;
  const dy = outer.y - inner.y;
  const halfWidth = Math.hypot(dx, dy) / 2;
  if (!(halfWidth > 1e-6)) return null;

  return {
    x: clamp(((iris.x - midX) * aspect) / halfWidth, -1.5, 1.5),
    // An eye is far wider than it is tall, so the same normaliser would make
    // vertical gaze read as enormous. Halving it keeps the axes comparable.
    y: clamp((iris.y - midY) / (halfWidth * 0.5), -1.5, 1.5),
  };
}

/**
 * Average gaze over both eyes, in image-normalised units.
 * Returns `{x, y, left, right}`; `x`/`y` are the mean of whatever eyes exist.
 */
export function gazeFromLandmarks(landmarks, aspect = 1) {
  if (!landmarks || landmarks.length < 474) {
    return { x: 0, y: 0, left: null, right: null };
  }
  const left = eyeGaze(landmarks, EYE_LANDMARKS.left, aspect);
  const right = eyeGaze(landmarks, EYE_LANDMARKS.right, aspect);
  const eyes = [left, right].filter(Boolean);
  if (eyes.length === 0) return { x: 0, y: 0, left, right };
  const mean = eyes.reduce(
    (sum, eye) => ({ x: sum.x + eye.x, y: sum.y + eye.y }),
    { x: 0, y: 0 },
  );
  return {
    x: mean.x / eyes.length,
    y: mean.y / eyes.length,
    left,
    right,
  };
}

/**
 * Column-major 4x4 rotation -> quaternion `[x, y, z, w]`.
 *
 * MediaPipe's facial transformation matrix is column-major (the same layout
 * as glTF), so this is a direct Shepperd extraction with no transpose. The
 * result is left in the tracker's camera frame, which is the frame the rest
 * of this app already treats the server's face quaternion as living in.
 */
export function quaternionFromMatrix(matrix) {
  if (!matrix || matrix.length < 16) return null;
  const m00 = matrix[0], m10 = matrix[1], m20 = matrix[2];
  const m01 = matrix[4], m11 = matrix[5], m21 = matrix[6];
  const m02 = matrix[8], m12 = matrix[9], m22 = matrix[10];

  const trace = m00 + m11 + m22;
  let x, y, z, w;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    w = 0.25 * s;
    x = (m21 - m12) / s;
    y = (m02 - m20) / s;
    z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / s;
    x = 0.25 * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25 * s;
    z = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25 * s;
  }

  const length = Math.hypot(x, y, z, w) || 1;
  return [x / length, y / length, z / length, w / length];
}

/**
 * ARKit's 52 scores -> VRM preset weights.
 *
 * `mouthRef` is the speaker's own rolling wide-open jaw value: a webcam shout
 * peaks near 0.6 while MuseTalk clips peak near 0.25, so normalising against
 * the observed range keeps the mouth moving in both cases instead of pinned
 * shut or permanently wide.
 *
 * Every returned weight is a VRM 1.0 preset name. `blinkLeft`/`blinkRight`
 * are also returned separately; the caller should prefer them over `blink`
 * when the avatar has both, so a wink and an asymmetric blink survive.
 */
export function mapBlendshapes(source, { mouthRef = 0.3 } = {}) {
  const g = readBlendshapes(source);
  const avg = (a, b) => (g(a) + g(b)) * 0.5;

  const jawRaw = g('jawOpen');
  const ref = clamp(mouthRef, 0.18, 0.7);
  const open = clamp((jawRaw - 0.025) / (ref * 0.75), 0, 1);

  const smile = soft(avg('mouthSmileLeft', 'mouthSmileRight'));
  const frown = soft(avg('mouthFrownLeft', 'mouthFrownRight'));
  const pucker = soft(g('mouthPucker'));
  const funnel = soft(g('mouthFunnel'));
  const stretch = soft(avg('mouthStretchLeft', 'mouthStretchRight'));
  const browDown = soft(avg('browDownLeft', 'browDownRight'));
  const browIn = soft(g('browInnerUp'));
  const browOut = soft(avg('browOuterUpLeft', 'browOuterUpRight'));
  const eyeWide = soft(avg('eyeWideLeft', 'eyeWideRight'));
  const squint = soft(avg('eyeSquintLeft', 'eyeSquintRight'));

  const blinkLeft = soft(g('eyeBlinkLeft'));
  const blinkRight = soft(g('eyeBlinkRight'));

  // Mouth shapes share the same jaw evidence, so they are solved together and
  // then normalised: five presets each at 0.9 would distort the lips more than
  // a single 1.0 would.
  let aa = open * (1 - 0.55 * pucker) * (1 - 0.35 * funnel);
  let oh = clamp(open * funnel * 1.7 + funnel * 0.2, 0, 1);
  let ou = clamp(pucker * (0.7 + 0.3 * open), 0, 1);
  let ee = clamp(stretch * (0.75 + 0.25 * (1 - open)), 0, 1);
  let ih = clamp(stretch * 0.45 + smile * (1 - open) * 0.15, 0, 1);

  const total = aa + oh + ou + ee + ih;
  if (total > 1.15) {
    const scale = 1.15 / total;
    aa *= scale; oh *= scale; ou *= scale; ee *= scale; ih *= scale;
  }

  return {
    aa,
    ih,
    ou,
    ee,
    oh,
    blink: Math.min(1, (blinkLeft + blinkRight) * 0.5),
    blinkLeft,
    blinkRight,
    happy: Math.min(1, smile * (1 - 0.6 * open)),
    angry: Math.min(1, browDown * 0.9 + frown * 0.3),
    sad: clamp(frown * 0.7 + browIn * 0.35 - browOut * 0.2, 0, 1),
    surprised: Math.min(1, browIn * 0.75 + eyeWide * 0.55),
    relaxed: Math.min(1, squint * 0.3),
  };
}

/**
 * One tracked frame -> the shape `MocapRetargeter.updateFace` consumes.
 *
 * Keeping `blend` (rather than pre-mapped weights) means the mouth calibration
 * still sees the raw `jawOpen` and can follow the speaker's own range.
 */
export function faceFrameToVrm(frame, { mouthRef = 0.3 } = {}) {
  if (!frame) return null;
  const blendshapes = frame.blendshapes || {};
  const gaze = frame.gaze ||
    gazeFromLandmarks(frame.landmarks, frame.aspect || 1);
  return {
    blend: blendshapes,
    q: quaternionFromMatrix(frame.transform),
    gaze,
    // forwarded for diagnostics / future drivers
    landmarks: frame.landmarks || null,
    transform: frame.transform || null,
    aspect: frame.aspect || 1,
    at: typeof performance !== 'undefined' ? performance.now() : 0,
    mouthRef,
  };
}

/** Expression names a tracked frame would like to write. */
export const FACE_PRESETS = [
  'aa', 'ih', 'ou', 'ee', 'oh',
  'blink', 'blinkLeft', 'blinkRight',
  'happy', 'angry', 'sad', 'relaxed', 'surprised',
];
