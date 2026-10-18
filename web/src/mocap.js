import * as THREE from 'three';

/**
 * Retargets COCO-17 keypoints from a mirrored webcam image onto a VRM
 * humanoid. The mirror means the person's right side appears on the right of
 * the screen, which is the avatar's *left* side (it faces the viewer), so
 * sides are swapped below. The keypoints arrive already corrected for the
 * mirror by the pose engine (it flips the frame before inference, so the raw
 * labels would otherwise describe the wrong side), which is why the swap here
 * is by index name and not by screen position.
 *
 * Joint rotation is solved by aiming each bone at the next joint in the
 * chain: rotate the bone so its current direction to the child matches the
 * direction measured in the image. Depth is recovered from limb
 * foreshortening, which makes arms reach toward the camera when the projected
 * segment shortens.
 */

const KP = {
  nose: 0,
  leftEye: 1,
  rightEye: 2,
  leftEar: 3,
  rightEar: 4,
  leftShoulder: 5,
  rightShoulder: 6,
  leftElbow: 7,
  rightElbow: 8,
  leftWrist: 9,
  rightWrist: 10,
  leftHip: 11,
  rightHip: 12,
  leftKnee: 13,
  rightKnee: 14,
  leftAnkle: 15,
  rightAnkle: 16,
};

// bone name -> [joint A, joint B] indices (A~B is the bone direction)
const ARM = {
  left: [KP.rightShoulder, KP.rightElbow, KP.rightWrist], // person right
  right: [KP.leftShoulder, KP.leftElbow, KP.leftWrist],
};
const LEG = {
  left: [KP.rightHip, KP.rightKnee, KP.rightAnkle],
  right: [KP.leftHip, KP.leftKnee, KP.leftAnkle],
};

const JOINT_CONF = 0.3;
const TAU_HEAD = 0.08;
const TAU_ROOT = 0.09;
const TAU_BLEND = 0.2;
const TAU_FACE = 0.045;      // expression weight smoothing
const TAU_FACE_HEAD = 0.07;  // facial head pose smoothing

// One Euro filter bands for the two fast, noisy signals: joint positions
// (pixels) and limb directions (radians). A fixed time constant has to choose
// between lag on a fast move and jitter when still; these raise the cutoff
// with the signal's own speed, so a still limb is smoothed hard and a swinging
// one is followed closely.
const POS_FILTER = { minCutoff: 1.2, beta: 0.04 };
const DIR_FILTER = { minCutoff: 1.0, beta: 0.7 };

// How fast the per limb foreshortening reference may grow (fraction of itself
// per frame) and how far it is allowed to stray from this avatar's own bone.
const REF_MAX = { rise: 0.01, decay: 0.9995, floor: 0.6, cap: 1.2 };

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const damp = (dt, tau) => 1 - Math.exp(-dt / tau);

/** One Euro blend factor: more smoothing while `speed` is low, less while high. */
function oneEuro(dt, speed, { minCutoff, beta }) {
  const cutoff = minCutoff + beta * Math.max(0, speed);
  return 1 - Math.exp(-dt * 2 * Math.PI * cutoff);
}

const AXIS_Y = new THREE.Vector3(0, 1, 0);
const AXIS_Z = new THREE.Vector3(0, 0, 1);

export class MocapRetargeter {
  constructor(vrm, root) {
    this.vrm = vrm;
    this.root = root;

    const get = (name) => vrm.humanoid.getNormalizedBoneNode(name);
    this.b = {
      hips: get('hips'),
      spine: get('spine'),
      chest: get('chest'),
      upperChest: get('upperChest'),
      neck: get('neck'),
      head: get('head'),
      leftUpperArm: get('leftUpperArm'),
      leftLowerArm: get('leftLowerArm'),
      leftHand: get('leftHand'),
      rightUpperArm: get('rightUpperArm'),
      rightLowerArm: get('rightLowerArm'),
      rightHand: get('rightHand'),
      leftUpperLeg: get('leftUpperLeg'),
      leftLowerLeg: get('leftLowerLeg'),
      leftFoot: get('leftFoot'),
      leftToes: get('leftToes'),
      rightUpperLeg: get('rightUpperLeg'),
      rightLowerLeg: get('rightLowerLeg'),
      rightFoot: get('rightFoot'),
      rightToes: get('rightToes'),
    };
    if (!this.b.chest && this.b.upperChest) this.b.chest = this.b.upperChest;
    if (!this.b.chest) this.b.chest = this.b.spine;

    this.smoothPx = new Float32Array(17 * 2);
    this.scores = new Float32Array(17);
    this.hasJoint = new Uint8Array(17);
    this.blend = 0;
    this.head = { yaw: 0, roll: 0 };
    // Facial layer: MediaPipe gives a head quaternion (world space, since the
    // frame it sees is already mirrored like the preview) and ARKit-style
    // blendshape weights, which are mapped onto the VRM expression presets.
    this.face = {
      q: new THREE.Quaternion(),
      target: new THREE.Quaternion(),
      headW: 0,
      weights: Object.create(null),
      // rolling "wide open" reference for this speaker, so the mouth opens
      // fully at whatever jaw range the face/camera actually produces
      mouthRef: 0.25,
      fresh: false,
    };
    this._faceNames = null;
    this.facing = 1; // +1 person faces the camera, -1 they turned away
    this.dirState = new Map();
    // longest each limb has been seen to project, relative to this avatar's
    // own bone: the reference foreshortening is measured against
    this.limbRef = new Map();
    this.torsoPx = 0;
    // on the instance so the bands can be retuned without touching the maths
    this.filter = { pos: { ...POS_FILTER }, dir: { ...DIR_FILTER } };
    this.refMax = { ...REF_MAX };

    // Captured before any animation runs: the head is driven relative to its
    // rest orientation so a bad idle clip can never flip it around.
    this.restHeadLocal = this.b.head
      ? this.b.head.quaternion.clone()
      : null;
    this.rest = this.measure();
    this._v1 = new THREE.Vector3();
    this._v2 = new THREE.Vector3();
    this._q1 = new THREE.Quaternion();
    this._q2 = new THREE.Quaternion();
    this._q3 = new THREE.Quaternion();
    this._q4 = new THREE.Quaternion();
    this._before = new THREE.Quaternion();
    this._headTarget = new THREE.Quaternion();
    this._target = new THREE.Vector3();

    this.enabled = true;
  }

  measure() {
    const vrm = this.vrm;
    vrm.scene.updateMatrixWorld(true);
    const p = (node) => node.getWorldPosition(new THREE.Vector3());
    const fallback = { torso: 0.5, upperArm: 0.27, lowerArm: 0.25,
      upperLeg: 0.42, lowerLeg: 0.42 };

    try {
      const lSh = p(this.b.leftUpperArm);
      const rSh = p(this.b.rightUpperArm);
      const lHip = p(this.b.leftUpperLeg);
      const rHip = p(this.b.rightUpperLeg);
      const shMid = lSh.clone().add(rSh).multiplyScalar(0.5);
      const hipMid = lHip.clone().add(rHip).multiplyScalar(0.5);
      return {
        torso: shMid.distanceTo(hipMid) || fallback.torso,
        upperArm: p(this.b.leftUpperArm).distanceTo(p(this.b.leftLowerArm)) ||
          fallback.upperArm,
        lowerArm: p(this.b.leftLowerArm).distanceTo(p(this.b.leftHand)) ||
          fallback.lowerArm,
        upperLeg: p(this.b.leftUpperLeg).distanceTo(p(this.b.leftLowerLeg)) ||
          fallback.upperLeg,
        lowerLeg: p(this.b.leftLowerLeg).distanceTo(p(this.b.leftFoot)) ||
          fallback.lowerLeg,
      };
    } catch (err) {
      console.warn('[mocap] could not measure avatar, using defaults', err);
      return fallback;
    }
  }

  ingest(keypoints, scores, width, height, dt) {
    if (!Array.isArray(keypoints) || keypoints.length < 17) return;
    for (let i = 0; i < 17; i++) {
      const kp = keypoints[i];
      if (!kp) continue;
      const score = scores?.[i] ?? 0;
      const x = kp[0] * width;
      const y = kp[1] * height;
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const j = i * 2;
      if (!this.hasJoint[i]) {
        this.smoothPx[j] = x;
        this.smoothPx[j + 1] = y;
        this.hasJoint[i] = 1;
      } else if (score >= 0.2) {
        const speed = Math.hypot(x - this.smoothPx[j],
          y - this.smoothPx[j + 1]) / dt;
        const alpha = oneEuro(dt, speed, this.filter.pos);
        this.smoothPx[j] += (x - this.smoothPx[j]) * alpha;
        this.smoothPx[j + 1] += (y - this.smoothPx[j + 1]) * alpha;
      }
      this.scores[i] = score;
    }
  }

  px(i) {
    return { x: this.smoothPx[i * 2], y: this.smoothPx[i * 2 + 1] };
  }

  smoothDir(key, target, dt) {
    let current = this.dirState.get(key);
    if (!current) {
      current = target.clone();
      this.dirState.set(key, current);
    } else {
      const mag = current.length() * target.length();
      const angle = mag > 1e-9
        ? Math.acos(clamp(current.dot(target) / mag, -1, 1))
        : 0;
      current.lerp(target, oneEuro(dt, angle / dt, this.filter.dir));
      if (current.lengthSq() < 1e-5) current.copy(target);
      current.normalize();
    }
    return current;
  }

  /** 2D direction between two joints in avatar-world units (y up). */
  dir2(a, b, k) {
    return new THREE.Vector3((b.x - a.x) * k, -(b.y - a.y) * k, 0);
  }

  /**
   * 2D direction plus a depth component from foreshortening. When the
   * projected limb is shorter than the limb itself, it must be angled
   * toward/away from the camera; we assume toward (+Z).
   *
   * The reference is the longest the limb has ever projected relative to the
   * torso, not this avatar's own bone: the person and the avatar rarely have
   * the same proportions, and measuring against the avatar's bone would push
   * perfectly straight limbs at the camera. `key` names the limb the
   * calibration is shared by, so both sides agree.
   */
  limbDir(a, b, boneLength, k, key) {
    const v = this.dir2(a, b, k);
    const projected = v.length();
    const reference = this.limbReference(key, projected, boneLength, k);
    let z = 0;
    if (reference > 1e-4 && projected < reference) {
      const ratio = projected / reference;
      z = boneLength * Math.sqrt(Math.max(0, 1 - ratio * ratio)) * 0.55;
    }
    v.z = z;
    if (v.lengthSq() < 1e-8) v.set(0, -1, 0);
    return v.normalize();
  }

  /**
   * Calibrated reference length of one limb, in metres.
   *
   * The limb is measured as a fraction of the torso and compared with the same
   * fraction on this avatar, so camera distance and body size cancel. The
   * reference then creeps toward the longest that limb has ever projected:
   * the rise is rate limited, because a keypoint thrown off by the tracker
   * must not convince us the arm is three times its real length and leave
   * every frame afterwards leaning at the camera.
   */
  limbReference(key, projected, boneLength, k) {
    const cap = this.refMax;
    let scale = this.limbRef.get(key);
    if (scale === undefined) scale = cap.floor;
    if (this.torsoPx > 1 && k > 1e-6) {
      const seen = (projected / k) / this.torsoPx /
        (boneLength / this.rest.torso);
      if (Number.isFinite(seen)) {
        if (seen > scale && seen < 2) {
          scale += Math.min(seen - scale, cap.rise);
        } else {
          scale *= cap.decay;
        }
        scale = clamp(scale, cap.floor, cap.cap);
        this.limbRef.set(key, scale);
      }
    }
    return boneLength * scale;
  }

  /**
   * Rotate `bone` so the direction to `child` matches `dir` in world space.
   * `weight` < 1 blends back toward the pose the animation mixer supplied.
   */
  aim(bone, child, dir, weight) {
    if (!bone || !child || weight <= 0.001) return;
    bone.getWorldPosition(this._v1);
    child.getWorldPosition(this._v2);
    this._v2.sub(this._v1);
    if (this._v2.lengthSq() < 1e-10 || dir.lengthSq() < 1e-10) return;
    this._v2.normalize();

    this._q1.setFromUnitVectors(this._v2, this._target.copy(dir).normalize());

    bone.parent.getWorldQuaternion(this._q2);
    bone.parent.getWorldQuaternion(this._q3);
    this._q3.invert().multiply(this._q1).multiply(this._q2);

    this._before.copy(bone.quaternion);
    bone.quaternion.premultiply(this._q3);
    if (weight < 1) bone.quaternion.slerp(this._before, 1 - weight);
    bone.updateWorldMatrix(false, true);
  }

  aimBone(boneName, childName, dir, weight) {
    this.aim(this.b[boneName], this.b[childName], dir, weight);
  }

  update(pose, dt) {
    if (!this.enabled || !this.b.hips) return;
    dt = clamp(dt, 0.001, 0.1);

    // the face layer runs even when no body is in frame (a close-up), so it
    // must happen before the body early-returns below
    this.updateFace(pose, dt);

    const hasPose = !!pose && Array.isArray(pose.keypoints) &&
      pose.keypoints.length >= 17;
    if (hasPose) {
      this.ingest(pose.keypoints, pose.scores, pose.width || 1280,
        pose.height || 720, dt);
    }

    const confident = hasPose &&
      this.scores[KP.leftShoulder] >= JOINT_CONF &&
      this.scores[KP.rightShoulder] >= JOINT_CONF &&
      this.scores[KP.leftHip] >= JOINT_CONF &&
      this.scores[KP.rightHip] >= JOINT_CONF;

    const targetBlend = confident ? 1 : 0;
    this.blend += (targetBlend - this.blend) * damp(dt, TAU_BLEND);
    if (!hasPose && this.blend < 0.002) {
      this.blend = 0;
      this.root.position.lerp(this._target.set(0, 0, 0), damp(dt, TAU_ROOT));
      return;
    }

    const weight = this.blend;
    this.vrm.scene.updateMatrixWorld(true);

    const lSh = this.px(5);
    const rSh = this.px(6);
    const lHip = this.px(11);
    const rHip = this.px(12);
    const shMid = { x: (lSh.x + rSh.x) / 2, y: (lSh.y + rSh.y) / 2 };
    const hipMid = { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 };
    const torsoPx = Math.hypot(shMid.x - hipMid.x, shMid.y - hipMid.y);
    if (torsoPx < 8) return;
    const k = this.rest.torso / torsoPx;
    this.torsoPx = torsoPx;

    // --- which way the person is facing --------------------------------
    // The hip and shoulder lines are *lateral* pairs: they reverse sign when
    // the person turns their back to the camera, and every angle built from
    // such a pair (pelvis up, head roll) has a 180 degree ambiguity. The
    // evidence is only trusted when it is clear - in profile both lines
    // collapse and the previous value is held.
    const shDx = this.px(6).x - this.px(5).x;
    const evidence = shDx + (this.px(12).x - this.px(11).x);
    const scale = Math.max(40, Math.abs(shDx));
    if (Math.abs(evidence) > 0.6 * scale) this.facing = evidence > 0 ? 1 : -1;

    // --- torso ---------------------------------------------------------
    const hipLine = this.dir2(this.px(11), this.px(12), k);
    hipLine.z = 0;
    if (hipLine.lengthSq() > 1e-8) {
      hipLine.normalize();
      const up = new THREE.Vector3(-hipLine.y, hipLine.x, 0).normalize();
      // Pin the perpendicular to the upward half space: a person never stands
      // on their head, and without this the pelvis aims straight down and
      // folds the avatar in half as soon as the person faces away.
      if (up.y < 0) up.negate();
      const upDir = this.smoothDir('up', up, dt);
      this.aimBone('hips', 'spine', upDir, weight);

      const torso = this.smoothDir('torso',
        this.dir2(hipMid, shMid, k).normalize(), dt);
      if (this.b.spine && this.b.chest) {
        const blendDir = upDir.clone().lerp(torso, 0.6).normalize();
        this.aimBone('spine', 'chest', blendDir, weight);
      }
      if (this.b.chest && this.b.neck) {
        this.aimBone('chest', 'neck', torso, weight);
      }
    }

    // --- head ----------------------------------------------------------
    this.updateHead(dt, weight, k);

    // --- arms ----------------------------------------------------------
    for (const side of ['left', 'right']) {
      const [jS, jE, jW] = ARM[side];
      const shoulder = this.px(jS);
      const elbow = this.px(jE);
      const wrist = this.px(jW);
      const upper = side === 'left' ? 'leftUpperArm' : 'rightUpperArm';
      const lower = side === 'left' ? 'leftLowerArm' : 'rightLowerArm';
      const hand = side === 'left' ? 'leftHand' : 'rightHand';
      const dUpper = this.smoothDir(`${side}Upper`,
        this.limbDir(shoulder, elbow, this.rest.upperArm, k, 'upperArm'), dt);
      this.aimBone(upper, lower, dUpper, weight);
      const dLower = this.smoothDir(`${side}Lower`,
        this.limbDir(elbow, wrist, this.rest.lowerArm, k, 'lowerArm'), dt);
      this.aimBone(lower, hand, dLower, weight);
    }

    // --- legs ----------------------------------------------------------
    for (const side of ['left', 'right']) {
      const [jH, jK, jA] = LEG[side];
      const hip = this.px(jH);
      const knee = this.px(jK);
      const ankle = this.px(jA);
      const upper = side === 'left' ? 'leftUpperLeg' : 'rightUpperLeg';
      const lower = side === 'left' ? 'leftLowerLeg' : 'rightLowerLeg';
      const foot = side === 'left' ? 'leftFoot' : 'rightFoot';
      const toes = side === 'left' ? 'leftToes' : 'rightToes';
      const dUpper = this.smoothDir(`${side}UpperLeg`,
        this.limbDir(hip, knee, this.rest.upperLeg, k, 'upperLeg'), dt);
      this.aimBone(upper, lower, dUpper, weight);
      const dLower = this.smoothDir(`${side}LowerLeg`,
        this.limbDir(knee, ankle, this.rest.lowerLeg, k, 'lowerLeg'), dt);
      this.aimBone(lower, foot, dLower, weight);
      if (this.b[foot] && this.b[toes]) {
        const footDir = this.smoothDir(`${side}Foot`,
          new THREE.Vector3(0, -0.18, 1).normalize(), dt);
        this.aimBone(foot, toes, footDir, weight);
      }
    }

    // --- root translation ---------------------------------------------
    const nx = hipMid.x / (pose?.width || 1280);
    const ny = hipMid.y / (pose?.height || 720);
    this._target.set(
      clamp((nx - 0.5) * 1.8, -0.9, 0.9) * weight,
      clamp((0.6 - ny) * 1.2, -0.15, 0.15) * weight,
      0,
    );
    this.root.position.lerp(this._target, damp(dt, TAU_ROOT));
  }

  /** VRM expression names this avatar actually has, resolved once. */
  faceExpressions() {
    if (this._faceNames) return this._faceNames;
    const em = this.vrm && this.vrm.expressionManager;
    if (!em) return null;
    const map = em.expressionMap || {};
    const pick = (...names) => names.find((n) => n in map) || null;
    this._faceNames = {
      blink: pick('blink'),
      blinkLeft: pick('blinkLeft', 'blink_l'),
      blinkRight: pick('blinkRight', 'blink_r'),
      aa: pick('aa', 'a'),
      ih: pick('ih', 'i'),
      ou: pick('ou', 'u'),
      ee: pick('ee', 'e'),
      oh: pick('oh', 'o'),
      happy: pick('happy'),
      angry: pick('angry'),
      sad: pick('sad'),
      relaxed: pick('relaxed'),
      surprised: pick('surprised'),
    };
    return this._faceNames;
  }

  /**
   * Drive the face from `pose.face` (head quaternion + blendshapes).
   * Falls back to zero weights when the face is lost, so expressions ease out
   * instead of sticking.
   */
  updateFace(pose, dt) {
    const face = pose && pose.face ? pose.face : null;
    const hasQ = !!(face && Array.isArray(face.q) && face.q.length === 4);
    const fresh = !!(face && face.blend);
    this.face.fresh = fresh;

    // --- head ------------------------------------------------------------
    if (hasQ) {
      this.face.target.set(face.q[0], face.q[1], face.q[2], face.q[3]);
      if (this.face.target.lengthSq() < 1e-6) this.face.target.identity();
      else this.face.target.normalize();
    }
    this.face.headW += ((hasQ ? 1 : 0) - this.face.headW) *
      damp(dt, TAU_FACE_HEAD);

    const head = this.b.head;
    if (head && this.restHeadLocal && this.face.headW > 0.001) {
      this.face.q.slerp(this.face.target, damp(dt, TAU_FACE_HEAD));
      this.vrm.scene.updateMatrixWorld(true);
      head.parent.getWorldQuaternion(this._q3);
      this._q4.copy(this._q3).invert()
        .multiply(this.face.q).multiply(this._q3);
      this._headTarget.copy(this.restHeadLocal).premultiply(this._q4);
      head.quaternion.slerp(this._headTarget, this.face.headW);
    }

    // --- expressions -----------------------------------------------------
    const em = this.vrm && this.vrm.expressionManager;
    const names = this.faceExpressions();
    if (!em || !names) return;

    const g = (name) => (fresh ? (face.blend[name] || 0) : 0);
    const avg = (a, b) => (g(a) + g(b)) * 0.5;
    // ARKit coefficients idle around 0.02-0.08; the soft floor keeps a resting
    // face from twitching its brows and mouth
    const s = (v) => (v <= 0.06 ? 0 : Math.min(1, (v - 0.06) / 0.94));

    const jaw = s(g('jawOpen'));
    // MuseTalk clips peak around jawOpen 0.25 while a shouted webcam pose
    // reaches 0.6; normalising against the speaker's own range keeps the mouth
    // visibly moving instead of pinned shut or permanently wide
    const jawRaw = g('jawOpen');
    this.face.mouthRef += (jawRaw - this.face.mouthRef) *
      (jawRaw > this.face.mouthRef ? 0.35 : 0.008);
    this.face.mouthRef = clamp(this.face.mouthRef, 0.18, 0.7);
    const open = clamp(
      (jawRaw - 0.025) / (this.face.mouthRef * 0.75), 0, 1);
    const smile = s(avg('mouthSmileLeft', 'mouthSmileRight'));
    const frown = s(avg('mouthFrownLeft', 'mouthFrownRight'));
    const pucker = s(g('mouthPucker'));
    const funnel = s(g('mouthFunnel'));
    const stretch = s(avg('mouthStretchLeft', 'mouthStretchRight'));
    const browDown = s(avg('browDownLeft', 'browDownRight'));
    const browIn = s(g('browInnerUp'));
    const browOut = s(avg('browOuterUpLeft', 'browOuterUpRight'));
    const eyeWide = s(avg('eyeWideLeft', 'eyeWideRight'));
    const squint = s(avg('eyeSquintLeft', 'eyeSquintRight'));

    const targets = {
      blink: s(avg('eyeBlinkLeft', 'eyeBlinkRight')),
      aa: open,
      ou: pucker,
      oh: Math.min(1, open * funnel * 1.7 + funnel * 0.2),
      ee: stretch,
      // no direct ARKit source for a closed "ih" mouth, and deriving it from
      // the smile just double-drives `happy`, so it stays unused
      ih: 0,
      happy: Math.min(1, smile * (1 - 0.6 * jaw)),
      angry: Math.min(1, browDown * 0.9 + frown * 0.3),
      sad: clamp(frown * 0.7 + browIn * 0.35 - browOut * 0.2, 0, 1),
      surprised: Math.min(1, browIn * 0.75 + eyeWide * 0.55),
      relaxed: Math.min(1, squint * 0.3),
    };

    const a = damp(dt, TAU_FACE);
    for (const key of Object.keys(targets)) {
      const name = names[key];
      if (!name) continue;
      const want = targets[key];
      const cur = this.face.weights[key] || 0;
      const next = cur + (want - cur) * a;
      this.face.weights[key] = next;
      if (next > 0.0005 || cur > 0.0005) em.setValue(name, next);
    }
  }

  updateHead(dt, weight, k) {
    const head = this.b.head;
    if (!head) return;
    // when the facial tracker has a lock it owns the head pose outright
    if (this.face.headW > 0.001) return;
    let roll = this.head.roll;
    let yaw = this.head.yaw;

    const earsOk = this.scores[KP.leftEar] > JOINT_CONF &&
      this.scores[KP.rightEar] > JOINT_CONF;
    const eyesOk = this.scores[KP.leftEye] > JOINT_CONF &&
      this.scores[KP.rightEye] > JOINT_CONF;

    if (earsOk) {
      // mirrored view: person's right ear is the avatar's left ear. Turning
      // away swaps which ear lands on which side of the image, so the whole
      // head-frame angle flips with `facing`.
      const left = this.px(KP.rightEar);
      const right = this.px(KP.leftEar);
      const dx = (left.x - right.x) * k * this.facing;
      const dy = -(left.y - right.y) * k * this.facing;
      const width = Math.hypot(dx, dy);
      if (width > 0.01) {
        roll = clamp(Math.atan2(dy, dx), -0.7, 0.7);
        if (this.scores[KP.nose] > 0.4) {
          const nose = this.px(KP.nose);
          const midX = (left.x + right.x) / 2;
          yaw = clamp(Math.atan2((nose.x - midX) * k, width) * 1.4,
            -0.8, 0.8);
        }
      }
    } else if (eyesOk) {
      const left = this.px(KP.rightEye);
      const right = this.px(KP.leftEye);
      roll = clamp(Math.atan2(-(left.y - right.y) * this.facing,
        Math.abs(left.x - right.x)), -0.5, 0.5);
    }

    const a = damp(dt, TAU_HEAD);
    this.head.roll += (roll - this.head.roll) * a;
    this.head.yaw += (yaw - this.head.yaw) * a;

    if (weight <= 0.001 || !this.restHeadLocal) return;

    // Face angles are applied in world space on top of the rest head pose,
    // which is carried along by the (mocap driven) neck.
    this._q1.setFromAxisAngle(AXIS_Y, this.head.yaw);
    this._q2.setFromAxisAngle(AXIS_Z, this.head.roll);
    this._q1.multiply(this._q2);

    head.parent.getWorldQuaternion(this._q3);
    this._q4.copy(this._q3).invert().multiply(this._q1).multiply(this._q3);
    this._headTarget.copy(this.restHeadLocal).premultiply(this._q4);
    head.quaternion.slerp(this._headTarget, weight);
  }

  reset() {
    this.blend = 0;
    this.hasJoint.fill(0);
    this.dirState.clear();
    this.limbRef.clear();
    this.torsoPx = 0;
    this.facing = 1;
    this.head.roll = 0;
    this.head.yaw = 0;
    this.face.q.identity();
    this.face.target.identity();
    this.face.headW = 0;
    this.face.mouthRef = 0.25;
    this.face.weights = Object.create(null);
    this.root.position.set(0, 0, 0);
  }
}
