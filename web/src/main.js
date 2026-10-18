import './style.css';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import {
  createVRMAnimationClip,
  VRMAnimationLoaderPlugin,
} from '@pixiv/three-vrm-animation';

import { MocapRetargeter } from './mocap.js';

const FALLBACK_VRM = '/vrm/AvatarSample_B.vrm';
const FALLBACK_VRMA = '/vrm/idle_loop.vrma';

// The avatar files can be swapped on disk; ask the server for versioned URLs
// so a rebuilt .vrm is never served out of the browser cache.
async function avatarUrls() {
  try {
    const response = await fetch('/api/avatar', { cache: 'no-store' });
    const data = await response.json();
    return {
      vrm: data.vrm?.url ?? FALLBACK_VRM,
      vrma: data.vrma?.url ?? FALLBACK_VRMA,
    };
  } catch {
    return { vrm: FALLBACK_VRM, vrma: FALLBACK_VRMA };
  }
}

/* ------------------------------------------------------------------ dom */
const canvas = document.getElementById('scene');
const loaderEl = document.getElementById('loader');
const loaderText = document.getElementById('loader-text');
const loaderBar = document.getElementById('loader-bar');
const statusPill = document.getElementById('status-pill');
const statusText = document.getElementById('status-text');
const feedPanel = document.getElementById('feed-panel');
const feedImg = document.getElementById('feed');
const feedMeta = document.getElementById('feed-meta');
const statRender = document.getElementById('stat-render');
const statPose = document.getElementById('stat-pose');
const statModel = document.getElementById('stat-model');
const toastEl = document.getElementById('toast');
const btnReset = document.getElementById('btn-reset');
const btnFeed = document.getElementById('btn-feed');
const btnFull = document.getElementById('btn-fullscreen');
const sourceSelect = document.getElementById('source-select');
const deviceSelect = document.getElementById('device-select');
const btnUpload = document.getElementById('btn-upload');
const fileInput = document.getElementById('file-input');

/* --------------------------------------------------------------- scene */
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07080c);
scene.fog = new THREE.Fog(0x07080c, 5.5, 14);

// Framing: keep the whole avatar in shot for any viewport shape, from a wide
// desktop window to the narrow side panel.
const FRAME = { height: 2.0, width: 1.3, centerY: 0.92 };
const DEFAULT_DIR = new THREE.Vector3(0, 0.44, 3.05).normalize();

const camera = new THREE.PerspectiveCamera(
  32,
  window.innerWidth / window.innerHeight,
  0.1,
  60,
);
camera.position.copy(DEFAULT_DIR).multiplyScalar(3.1).add(
  new THREE.Vector3(0, FRAME.centerY, 0),
);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, FRAME.centerY, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
// left-drag orbits, right-drag slides the avatar around the frame, wheel zooms
controls.enablePan = true;
controls.screenSpacePanning = true;
controls.panSpeed = 1;
controls.mouseButtons = {
  LEFT: THREE.MOUSE.ROTATE,
  MIDDLE: THREE.MOUSE.DOLLY,
  RIGHT: THREE.MOUSE.PAN,
};
controls.minDistance = 1.2;
controls.maxDistance = 12;
controls.minPolarAngle = 0.55;
controls.maxPolarAngle = Math.PI / 2 - 0.02;
controls.update();

// right-drag is a pan gesture, so the browser menu must not steal it
canvas.addEventListener('contextmenu', (event) => event.preventDefault());

let userMovedCamera = false;
controls.addEventListener('start', () => {
  userMovedCamera = true;
});

function fitCamera(force = false) {
  // a zero-sized viewport during a panel resize must never poison the camera
  const ok = Number.isFinite(camera.position.x) &&
    Number.isFinite(controls.target.x);
  if (!ok) {
    camera.position.copy(DEFAULT_DIR).multiplyScalar(3).add(
      new THREE.Vector3(0, FRAME.centerY, 0),
    );
    controls.target.set(0, FRAME.centerY, 0);
    force = true;
  }
  if (userMovedCamera && !force) return;

  const width = Math.max(1, window.innerWidth);
  const height = Math.max(1, window.innerHeight);
  const aspect = Math.min(8, Math.max(0.15, width / height));
  camera.aspect = aspect;

  const halfFov = THREE.MathUtils.degToRad(camera.fov) / 2;
  const distance = Math.min(
    20,
    Math.max(
      FRAME.height / 2 / Math.tan(halfFov),
      FRAME.width / 2 / (Math.tan(halfFov) * aspect),
    ),
  );

  const direction = DEFAULT_DIR.clone();
  if (!force) {
    direction.copy(camera.position).sub(controls.target);
    if (direction.lengthSq() < 1e-6) direction.copy(DEFAULT_DIR);
    else direction.normalize();
  }
  controls.target.set(0, FRAME.centerY, 0);
  camera.position.copy(controls.target)
    .addScaledVector(direction, distance);
  camera.updateProjectionMatrix();
  controls.update();
  controls.saveState();
}

fitCamera(true);

scene.add(new THREE.HemisphereLight(0x9dc8ff, 0x0a0c12, 1.15));

const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
keyLight.position.set(2.4, 4.2, 3.2);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.left = -2.5;
keyLight.shadow.camera.right = 2.5;
keyLight.shadow.camera.top = 4;
keyLight.shadow.camera.bottom = -1;
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 12;
keyLight.shadow.bias = -0.0006;
scene.add(keyLight);

const rimLight = new THREE.DirectionalLight(0x62f2d2, 1.7);
rimLight.position.set(-3.2, 2.4, -3.4);
scene.add(rimLight);

const warmLight = new THREE.PointLight(0xff7aa8, 9, 7, 2);
warmLight.position.set(1.7, 1.1, 1.4);
scene.add(warmLight);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(3.8, 96),
  new THREE.MeshStandardMaterial({
    color: 0x0a0d13,
    roughness: 0.92,
    metalness: 0.15,
  }),
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(9, 36, 0x1c2433, 0x10151e);
grid.position.y = 0.004;
grid.material.transparent = true;
grid.material.opacity = 0.55;
scene.add(grid);

const ring = new THREE.Mesh(
  new THREE.RingGeometry(1.0, 1.035, 96),
  new THREE.MeshBasicMaterial({
    color: 0x62f2d2,
    transparent: true,
    opacity: 0.22,
    side: THREE.DoubleSide,
  }),
);
ring.rotation.x = -Math.PI / 2;
ring.position.y = 0.006;
scene.add(ring);

// soft contact shadow that follows the avatar's root motion
function makeBlobShadow() {
  const size = 256;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const grad = ctx.createRadialGradient(
    size / 2, size / 2, 0,
    size / 2, size / 2, size / 2,
  );
  grad.addColorStop(0, 'rgba(0,0,0,0.55)');
  grad.addColorStop(0.55, 'rgba(0,0,0,0.22)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(c);
  texture.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(1.6, 1.6),
    new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
    }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.008;
  return mesh;
}

const blob = makeBlobShadow();
scene.add(blob);

const avatarRoot = new THREE.Group();
scene.add(avatarRoot);

/* --------------------------------------------------------------- state */
const state = {
  vrm: null,
  mixer: null,
  retargeter: null,
  pose: null,
  poseAt: 0,
  poseFps: 0,
  detected: false,
  mode: '—',
  source: 'camera:0',
  resolution: '',
  wsOpen: false,
  engineError: '',
  toastError: false,
  feedOn: true,
};

let blinkExpression = null;
let blinkTimer = 2 + Math.random() * 3;
let blinkPhase = 0;
let toastTimer = 0;

/* ------------------------------------------------------------------ ui */
function setProgress(frac) {
  loaderBar.style.width = `${Math.round(Math.min(1, frac) * 100)}%`;
}

function showToast(message, { sticky = false, error = false } = {}) {
  if (!message) return;
  toastEl.textContent = message;
  toastEl.classList.toggle('error', error);
  toastEl.classList.add('show');
  state.toastError = error;
  clearTimeout(toastTimer);
  if (!sticky) {
    toastTimer = window.setTimeout(() => {
      toastEl.classList.remove('show');
      state.toastError = false;
    }, 6000);
  }
}

function hideToast() {
  clearTimeout(toastTimer);
  toastEl.classList.remove('show');
  state.toastError = false;
}

/* --------------------------------------------------------------- avatar */
function loadGLTF(url, plugin, onProgress) {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    if (plugin) loader.register(plugin);
    loader.load(url, resolve, onProgress, reject);
  });
}

async function loadAvatar() {
  loaderText.textContent = 'Loading avatar…';
  const urls = await avatarUrls();
  const gltf = await loadGLTF(
    urls.vrm,
    (parser) => new VRMLoaderPlugin(parser),
    (event) => {
      if (event.total) setProgress((event.loaded / event.total) * 0.85);
    },
  );

  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error('VRM data missing from file');

  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.rotateVRM0(vrm);
  vrm.scene.traverse((object) => {
    if (object.isMesh) {
      object.castShadow = true;
      object.receiveShadow = false;
      object.frustumCulled = false;
    }
  });

  avatarRoot.add(vrm.scene);
  avatarRoot.position.set(0, 0, 0);
  vrm.scene.updateMatrixWorld(true);

  state.vrm = vrm;
  state.retargeter = new MocapRetargeter(vrm, avatarRoot);

  if (vrm.lookAt) {
    vrm.lookAt.target = camera;
  }
  blinkExpression = vrm.expressionManager?.getExpression('blink') ?? null;

  setProgress(0.9);
  loaderText.textContent = 'Loading idle motion…';
  await loadIdle(vrm, urls.vrma);
  setProgress(1);
  loaderEl.classList.add('done');
}

async function loadIdle(vrm, url) {
  try {
    const gltf = await loadGLTF(
      url,
      (parser) => new VRMAnimationLoaderPlugin(parser),
    );
    const vrmAnimation = gltf.userData.vrmAnimations?.[0];
    if (!vrmAnimation) return;
    const clip = createVRMAnimationClip(vrmAnimation, vrm);
    const mixer = new THREE.AnimationMixer(vrm.scene);
    mixer.clipAction(clip).play();
    state.mixer = mixer;
  } catch (err) {
    console.warn('[idle] animation unavailable', err);
  }
}

/* ------------------------------------------------------------- websocket */
function connect() {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${proto}://${window.location.host}/ws/pose`);

  socket.addEventListener('open', () => {
    state.wsOpen = true;
  });

  socket.addEventListener('close', () => {
    state.wsOpen = false;
    window.setTimeout(connect, 1500);
  });

  socket.addEventListener('error', () => socket.close());

  socket.addEventListener('message', (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      return;
    }
    if (message.type === 'hello') {
      state.mode = message.mode ?? state.mode;
      state.source = message.source ?? state.source;
    } else if (message.type === 'pose') {
      state.pose = message;
      state.poseAt = performance.now();
      state.poseFps = message.fps ?? 0;
      state.detected = !!message.detected;
      if (message.width && message.height) {
        state.resolution = `${message.width}×${message.height}`;
      }
    }
  });
}

async function pollStatus() {
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    const status = await response.json();
    if (status.mode) state.mode = status.mode;
    if (status.device) state.device = status.device;
    if (status.device_pref && status.state !== 'switching' &&
        deviceSelect.value !== status.device_pref) {
      deviceSelect.value = status.device_pref;
    }
    const cudaOption = deviceSelect.querySelector('option[value="cuda"]');
    if (cudaOption && status.gpu_available !== undefined) {
      cudaOption.disabled = status.gpu_available === false;
      cudaOption.textContent = status.gpu_available === false
        ? 'gpu (unavailable)'
        : 'gpu (cuda)';
    }
    if (status.source) {
      state.source = status.source;
      if (sourceSelect.value && sourceSelect.value !== status.source &&
          status.state !== 'switching') {
        sourceSelect.value = status.source;
      }
    }
    const error = status.state === 'error' ? (status.error || 'Source unavailable') : '';
    state.engineError = error;
    if (error) showToast(error, { sticky: true, error: true });
    else if (state.toastError) hideToast();
  } catch {
    /* server restarting */
  }
}

async function loadSources() {
  try {
    const response = await fetch('/api/sources', { cache: 'no-store' });
    const data = await response.json();
    sourceSelect.innerHTML = '';
    for (const item of data.sources) {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = item.label;
      sourceSelect.append(option);
    }
    if (data.current) sourceSelect.value = data.current;
  } catch {
    /* the picker stays empty until the server answers */
  }
}

async function switchSource(source) {
  try {
    const response = await fetch('/api/source', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source }),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.error || `HTTP ${response.status}`);
    }
    state.source = source;
    state.poseAt = 0;
    startFeed();
    return true;
  } catch (err) {
    showToast(`Could not switch source: ${err.message}`, { error: true });
    return false;
  }
}

sourceSelect.addEventListener('change', () => switchSource(sourceSelect.value));

async function switchDevice(device) {
  try {
    const response = await fetch('/api/device', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ device }),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.error || `HTTP ${response.status}`);
    }
    state.device = device;
    state.poseAt = 0;
    showToast(device === 'cpu'
      ? 'Rebuilding on CPU (light models)…'
      : 'Rebuilding on the GPU (accurate models)…');
    return true;
  } catch (err) {
    showToast(`Could not switch device: ${err.message}`, { error: true });
    return false;
  }
}

deviceSelect.addEventListener('change', () => switchDevice(deviceSelect.value));

btnUpload.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  showToast(`Uploading ${file.name}…`, { sticky: true });
  try {
    const response = await fetch(
      `/api/upload?name=${encodeURIComponent(file.name)}`,
      { method: 'POST', body: file },
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    await loadSources();
    sourceSelect.value = data.source;
    state.source = data.source;
    state.poseAt = 0;
    startFeed();
    hideToast();
    showToast(`Playing ${data.name} as live pose`);
  } catch (err) {
    showToast(`Upload failed: ${err.message}`, { sticky: true, error: true });
  } finally {
    fileInput.value = '';
  }
});

/* ------------------------------------------------------------------ feed */
function startFeed() {
  feedImg.src = `/video?t=${Date.now()}`;
}

feedImg.addEventListener('load', () => feedImg.classList.add('live'));
feedImg.addEventListener('error', () => {
  feedImg.classList.remove('live');
  if (state.feedOn) window.setTimeout(startFeed, 2500);
});

/* --------------------------------------------------------------- controls */
btnReset.addEventListener('click', () => {
  userMovedCamera = false;
  fitCamera(true);
});

btnFeed.addEventListener('click', () => {
  state.feedOn = !state.feedOn;
  feedPanel.classList.toggle('hidden', !state.feedOn);
  btnFeed.classList.toggle('active', state.feedOn);
  if (state.feedOn) {
    startFeed();
  } else {
    feedImg.removeAttribute('src');
    feedImg.classList.remove('live');
  }
});

btnFull.addEventListener('click', () => {
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else {
    document.documentElement.requestFullscreen?.();
  }
});

/* ------------------------------------------------------------------ hud */
let hudAccum = 0;
function updateHud(dt, renderFps) {
  hudAccum += dt;
  if (hudAccum < 0.12) return;
  hudAccum = 0;

  const fresh = state.pose && performance.now() - state.poseAt < 800;
  const tracking = fresh && state.detected;
  statusPill.classList.toggle('live', tracking);
  statusPill.classList.toggle('offline', !!state.engineError || !state.wsOpen);

  if (state.engineError) {
    statusText.textContent = 'camera offline';
  } else if (!state.wsOpen) {
    statusText.textContent = 'reconnecting';
  } else if (tracking) {
    statusText.textContent = 'tracking';
  } else {
    statusText.textContent = 'searching for pose';
  }

  statRender.textContent = `${Math.round(renderFps)} fps`;
  statPose.textContent = fresh ? `${state.poseFps.toFixed(0)} fps` : '—';
  statModel.textContent = state.device
    ? `${state.mode} · ${state.device}`
    : state.mode;
  feedMeta.textContent = [state.source, state.resolution]
    .filter(Boolean)
    .join(' · ');

  if (!fresh) blob.material.opacity = 0.35;
  else blob.material.opacity = 1;
}

/* ------------------------------------------------------------------ loop */
let lastFrame = performance.now();
let renderFps = 60;
let elapsed = 0;

function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.1, Math.max(0.001, (now - lastFrame) / 1000));
  lastFrame = now;
  elapsed += dt;
  renderFps += (1 / dt - renderFps) * 0.06;

  if (state.mixer) state.mixer.update(dt);

  const fresh = state.pose && performance.now() - state.poseAt < 800;
  if (state.retargeter) {
    state.retargeter.update(fresh ? state.pose : null, dt);
  }

  updateBlink(dt);

  if (state.vrm) state.vrm.update(dt);

  blob.position.x = avatarRoot.position.x;
  controls.update();
  renderer.render(scene, camera);
  updateHud(dt, renderFps);
}

function updateBlink(dt) {
  if (!state.vrm?.expressionManager || !blinkExpression) {
    return;
  }
  blinkTimer -= dt;
  if (blinkTimer <= 0) {
    blinkPhase = 0.15;
    blinkTimer = 2 + Math.random() * 4;
  }
  let value = 0;
  if (blinkPhase > 0) {
    blinkPhase = Math.max(0, blinkPhase - dt);
    value = Math.sin((blinkPhase / 0.15) * Math.PI);
  }
  state.vrm.expressionManager.setValue('blink', value);
}

/* ----------------------------------------------------------------- init */
window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  fitCamera();
});

// handy for debugging from the console / automation
window.mocap = { state, scene, camera, controls, avatarRoot, renderer };

requestAnimationFrame(animate);

(async () => {
  try {
    await loadAvatar();
  } catch (err) {
    console.error(err);
    loaderText.textContent = 'Could not load the VRM avatar';
    setProgress(1);
    showToast(`VRM load failed: ${err.message}`, { sticky: true, error: true });
  }
  startFeed();
  connect();
  loadSources();
  pollStatus();
  window.setInterval(pollStatus, 3000);
})();
