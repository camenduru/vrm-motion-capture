/**
 * 3D retarget check (FlowHMR path).
 *
 * Builds a T-pose stand-in for the avatar - the retargeter only ever asks for
 * world positions, world quaternions and the normalized bone nodes, so a plain
 * THREE hierarchy with the same bone names is enough - then feeds it SMPL-H
 * joint sets whose answer is known by hand, and measures where the avatar's
 * bones actually ended up.
 *
 *   node web/test/mocap3d.test.mjs
 */
import assert from 'node:assert/strict';
import * as THREE from 'three';

import { MocapRetargeter } from '../src/mocap.js';

const SMPL = {
  pelvis: 0, leftHip: 1, rightHip: 2, spine: 3,
  leftKnee: 4, rightKnee: 5, spine1: 6,
  leftAnkle: 7, rightAnkle: 8, spine2: 9,
  leftToe: 10, rightToe: 11, neck: 12, head: 15,
  leftShoulder: 16, rightShoulder: 17,
  leftElbow: 18, rightElbow: 19, leftWrist: 20, rightWrist: 21,
};

/** A T-pose rig with VRM humanoid bone names, in metres. */
function fakeAvatar() {
  const nodes = {};
  const add = (name, parent, pos) => {
    const node = new THREE.Object3D();
    node.name = name;
    node.position.set(...pos);
    (parent ? nodes[parent] : null)?.add(node) ?? null;
    nodes[name] = node;
    return node;
  };

  const scene = new THREE.Object3D();
  scene.add(nodes.hips = new THREE.Object3D());
  nodes.hips.name = 'hips';
  nodes.hips.position.set(0, 1.0, 0);
  add('spine', 'hips', [0, 0.1, 0]);
  add('chest', 'spine', [0, 0.1, 0]);
  add('upperChest', 'chest', [0, 0.1, 0]);
  add('neck', 'upperChest', [0, 0.15, 0]);
  add('head', 'neck', [0, 0.15, 0]);

  add('leftUpperArm', 'upperChest', [0.18, 0.1, 0]);
  add('leftLowerArm', 'leftUpperArm', [0.27, 0, 0]);
  add('leftHand', 'leftLowerArm', [0.25, 0, 0]);
  add('rightUpperArm', 'upperChest', [-0.18, 0.1, 0]);
  add('rightLowerArm', 'rightUpperArm', [-0.27, 0, 0]);
  add('rightHand', 'rightLowerArm', [-0.25, 0, 0]);

  add('leftUpperLeg', 'hips', [0.1, -0.05, 0]);
  add('leftLowerLeg', 'leftUpperLeg', [0, -0.4, 0]);
  add('leftFoot', 'leftLowerLeg', [0, -0.43, 0]);
  add('leftToes', 'leftFoot', [0, -0.06, 0.12]);
  add('rightUpperLeg', 'hips', [-0.1, -0.05, 0]);
  add('rightLowerLeg', 'rightUpperLeg', [0, -0.4, 0]);
  add('rightFoot', 'rightLowerLeg', [0, -0.43, 0]);
  add('rightToes', 'rightFoot', [0, -0.06, 0.12]);

  scene.updateMatrixWorld(true);
  const vrm = {
    scene,
    humanoid: { getNormalizedBoneNode: (name) => nodes[name] || null },
    expressionManager: null,
  };
  return { vrm, nodes, root: scene };
}

/** SMPL-H body joints (0-21) in a T-pose, metres, y up, person facing +Z. */
function tPoseJoints() {
  const j = Array.from({ length: 22 }, () => [0, 0, 0]);
  j[SMPL.pelvis] = [0, 1.0, 0];
  j[SMPL.spine] = [0, 1.1, 0];
  j[SMPL.spine1] = [0, 1.2, 0];
  j[SMPL.spine2] = [0, 1.3, 0];
  j[SMPL.neck] = [0, 1.45, 0];
  j[SMPL.head] = [0, 1.6, 0];
  // the person's right is -X when they face +Z
  j[SMPL.leftHip] = [0.1, 0.95, 0];
  j[SMPL.rightHip] = [-0.1, 0.95, 0];
  j[SMPL.leftKnee] = [0.1, 0.55, 0];
  j[SMPL.rightKnee] = [-0.1, 0.55, 0];
  j[SMPL.leftAnkle] = [0.1, 0.12, 0];
  j[SMPL.rightAnkle] = [-0.1, 0.12, 0];
  j[SMPL.leftToe] = [0.1, 0.06, 0.12];
  j[SMPL.rightToe] = [-0.1, 0.06, 0.12];
  j[SMPL.leftShoulder] = [0.18, 1.4, 0];
  j[SMPL.rightShoulder] = [-0.18, 1.4, 0];
  j[SMPL.leftElbow] = [0.45, 1.4, 0];
  j[SMPL.rightElbow] = [-0.45, 1.4, 0];
  j[SMPL.leftWrist] = [0.7, 1.4, 0];
  j[SMPL.rightWrist] = [-0.7, 1.4, 0];
  return j;
}

const round = (v, n = 3) => Number(v.toFixed(n));

/** World direction from one avatar bone to its child. */
function boneDir(nodes, bone, child) {
  const a = nodes[bone].getWorldPosition(new THREE.Vector3());
  const b = nodes[child].getWorldPosition(new THREE.Vector3());
  return b.sub(a).normalize();
}

function run(joints, { frames = 90, fps = 30, rotateY = 0, driftZ = 0,
  offsetZ = 0 } = {}) {
  const { vrm, nodes, root } = fakeAvatar();
  const rt = new MocapRetargeter(vrm, root);
  const turned = joints.map((p) => {
    if (!rotateY) return p;
    const c = Math.cos(rotateY);
    const s = Math.sin(rotateY);
    return [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c];
  });
  for (let i = 0; i < frames; i++) {
    const shift = offsetZ + driftZ * (i / Math.max(1, frames - 1));
    const data = turned.map((p) => [p[0], p[1], p[2] + shift]);
    rt.update({ t: i / fps, fps, detected: true, joints3d: data,
      root: [0, 0, 0] }, 1 / fps);
  }
  return { rt, nodes, root };
}

const results = [];
function check(name, fn) {
  try {
    fn();
    results.push(`  PASS  ${name}`);
  } catch (err) {
    results.push(`  FAIL  ${name}\n        ${err.message}`);
    process.exitCode = 1;
  }
}

const near = (v, x, y, z, tol = 0.06) => {
  assert.ok(Math.abs(v.x - x) < tol && Math.abs(v.y - y) < tol &&
    Math.abs(v.z - z) < tol,
  `direction (${round(v.x)}, ${round(v.y)}, ${round(v.z)}) ` +
    `is not near (${x}, ${y}, ${z})`);
};

// ---------------------------------------------------------------- T-pose
check('T-pose input leaves the arms out to the sides', () => {
  const { nodes } = run(tPoseJoints());
  near(boneDir(nodes, 'leftUpperArm', 'leftLowerArm'), 1, 0, 0);
  near(boneDir(nodes, 'rightUpperArm', 'rightLowerArm'), -1, 0, 0);
  near(boneDir(nodes, 'leftUpperLeg', 'leftLowerLeg'), 0, -1, 0);
});

// ------------------------------------------------------- mirrored sides
check("person's right arm up drives the avatar's LEFT arm up", () => {
  const j = tPoseJoints();
  j[SMPL.rightElbow] = [-0.18, 1.65, 0];
  j[SMPL.rightWrist] = [-0.18, 1.9, 0];
  const { nodes } = run(j);
  near(boneDir(nodes, 'leftUpperArm', 'leftLowerArm'), 0, 1, 0);
  near(boneDir(nodes, 'leftLowerArm', 'leftHand'), 0, 1, 0);
  // the other arm must not move
  near(boneDir(nodes, 'rightUpperArm', 'rightLowerArm'), -1, 0, 0);
});

check("person's left arm up drives the avatar's RIGHT arm up", () => {
  const j = tPoseJoints();
  j[SMPL.leftElbow] = [0.18, 1.65, 0];
  j[SMPL.leftWrist] = [0.18, 1.9, 0];
  const { nodes } = run(j);
  near(boneDir(nodes, 'rightUpperArm', 'rightLowerArm'), 0, 1, 0);
  near(boneDir(nodes, 'leftUpperArm', 'leftLowerArm'), 1, 0, 0);
});

// ------------------------------------------------------------- depth
check('reaching toward the camera puts the hand toward the viewer', () => {
  const j = tPoseJoints();
  j[SMPL.rightElbow] = [-0.18, 1.4, 0.15];
  j[SMPL.rightWrist] = [-0.18, 1.4, 0.4];
  const { nodes } = run(j);
  near(boneDir(nodes, 'leftUpperArm', 'leftLowerArm'), 0, 0, 1);
});

check('a bent elbow keeps the forearm bent the same way', () => {
  const j = tPoseJoints();
  // upper arm out to the side, forearm up
  j[SMPL.rightElbow] = [-0.45, 1.4, 0];
  j[SMPL.rightWrist] = [-0.45, 1.7, 0];
  const { nodes } = run(j);
  near(boneDir(nodes, 'leftUpperArm', 'leftLowerArm'), 1, 0, 0);
  near(boneDir(nodes, 'leftLowerArm', 'leftHand'), 0, 1, 0);
});

// ------------------------------------------------------------- legs
check('a raised knee drives the matching leg', () => {
  const j = tPoseJoints();
  j[SMPL.rightKnee] = [-0.1, 0.75, 0.3];
  j[SMPL.rightAnkle] = [-0.1, 0.4, 0.3];
  const { nodes } = run(j);
  const d = boneDir(nodes, 'leftUpperLeg', 'leftLowerLeg');
  assert.ok(d.z > 0.6 && d.y < -0.2, `knee direction ${round(d.x)}, ` +
    `${round(d.y)}, ${round(d.z)} should be forward and down`);
});

// --------------------------------------------------- frame independence
check('the same pose rotated 90 degrees produces the same avatar pose', () => {
  const plain = run(tPoseJoints());
  const turned = run(tPoseJoints(), { rotateY: Math.PI / 2 });
  for (const [bone, child] of [
    ['leftUpperArm', 'leftLowerArm'],
    ['rightUpperArm', 'rightLowerArm'],
    ['leftUpperLeg', 'leftLowerLeg'],
    ['spine', 'chest'],
  ]) {
    const a = boneDir(plain.nodes, bone, child);
    const b = boneDir(turned.nodes, bone, child);
    assert.ok(a.distanceTo(b) < 0.08,
      `${bone} differs by ${round(a.distanceTo(b))} after rotating the input`);
  }
});

// ------------------------------------------------------- root motion
check('pelvis motion becomes root translation, scaled by the torso', () => {
  const { root } = run(tPoseJoints(), { driftZ: 0.3 });
  assert.ok(root.position.z > 0.2,
    `root z is ${round(root.position.z)}, expected > 0.2`);
  assert.ok(Math.abs(root.position.x) < 0.05,
    `root x drifted to ${round(root.position.x)}`);
});

check('a person standing off-centre does not teleport the avatar', () => {
  // the first frame is the reference: a constant offset is not motion
  const { root } = run(tPoseJoints(), { offsetZ: 0.5 });
  assert.ok(Math.abs(root.position.z) < 0.05,
    `root z is ${round(root.position.z)}, expected ~0`);
});

// ------------------------------------------------------- reset
check('reset drops the root reference so the next clip starts centred', () => {
  const { rt, root } = run(tPoseJoints());
  rt.reset();
  assert.equal(root.position.length(), 0);
  assert.equal(rt.rootRef, null);
});

console.log(results.join('\n'));
console.log(process.exitCode ? '\n3D retarget: FAILED' : '\n3D retarget: ok');
