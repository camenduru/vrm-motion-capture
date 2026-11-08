import fs from 'node:fs';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMAnimationLoaderPlugin } from '@pixiv/three-vrm-animation';

const path = process.argv[2];
if (!path) {
  console.error('usage: node check_vrma.mjs <file.vrma>');
  process.exit(2);
}
const buf = fs.readFileSync(path);
const arr = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);

const loader = new GLTFLoader();
loader.register((parser) => new VRMAnimationLoaderPlugin(parser));
loader.parse(
  arr,
  '',
  (gltf) => {
    const anim = gltf.userData.vrmAnimations?.[0];
    if (!anim) {
      console.error('FAIL: no vrmAnimations parsed');
      process.exit(1);
    }
    console.log(JSON.stringify({
      duration: anim.duration,
      rotationBones: [...anim.humanoidTracks.rotation.keys()],
      translationBones: [...anim.humanoidTracks.translation.keys()],
      expressions: [...anim.expressionTracks.preset.keys()],
    }, null, 2));
    const hipsRot = anim.humanoidTracks.rotation.get('hips');
    console.log('hips rotation keys', hipsRot?.times.length, 'first quat', Array.from(hipsRot?.values.slice(0, 4) ?? []));
    const hipsTr = anim.humanoidTracks.translation.get('hips');
    console.log('hips translation keys', hipsTr?.times.length, 'first pos', Array.from(hipsTr?.values.slice(0, 3) ?? []));
  },
  (err) => {
    console.error('FAIL: parse error', err);
    process.exit(1);
  },
);
