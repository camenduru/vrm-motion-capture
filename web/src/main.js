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
import { FaceTracker } from './face.js';
import { faceFrameToVrm } from './face-retarget.js';

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
const feedImg = document.getElementById('feed');
const feedMeta = document.getElementById('feed-meta');
const statRender = document.getElementById('stat-render');
const statPose = document.getElementById('stat-pose');
const statModel = document.getElementById('stat-model');
const toastEl = document.getElementById('toast');
const btnReset = document.getElementById('btn-reset');
const btnFull = document.getElementById('btn-fullscreen');
const btnSound = document.getElementById('btn-sound');
const sourceAudio = document.getElementById('source-audio');
const sourceSelect = document.getElementById('source-select');
const deviceSelect = document.getElementById('device-select');
const btnUpload = document.getElementById('btn-upload');
const btnGenerate = document.getElementById('btn-generate');
const busyEl = document.getElementById('busy');
const busyText = document.getElementById('busy-text');
const btnLoadVrma = document.getElementById('btn-load-vrma');
const vrmaInput = document.getElementById('vrma-input');
const btnRealtime = document.getElementById('btn-realtime');
const btnFaceTrack = document.getElementById('btn-face-track');
const statFace = document.getElementById('stat-face');
const fileInput = document.getElementById('file-input');
const lipsyncMode = document.getElementById('lipsync-mode');
const generateWith = document.getElementById('generate-with');
const flowhmrInterval = document.getElementById('flowhmr-interval');
const flowhmrIntervalPicker =
  document.getElementById('flowhmr-interval-picker');
const exmrCamera = document.getElementById('exmr-camera');
const exmrCameraPicker = document.getElementById('exmr-camera-picker');
const btnA2f = document.getElementById('btn-a2f');
const a2fInput = document.getElementById('a2f-input');
const btnA2fMic = document.getElementById('btn-a2f-mic');
const btnA2fPlay = document.getElementById('btn-a2f-play');
const a2fAudio = document.getElementById('a2f-audio');

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
  source: 'none',
  lipsyncMode: 'landmarks',
  // which offline generator "Generate VRMA…" runs: the 2D tracker this app
  // ships, or a 3D reconstructor in its own environment (no face track)
  generateWith: 'rtmpose',
  flowhmrReady: null,
  exmrReady: null,
  resolution: '',
  wsOpen: false,
  engineError: '',
  toastError: false,
  audioUrl: '',
  soundOn: localStorage.getItem('mocap.sound') !== 'off',
  recording: null,
  animation: null,
  animStart: 0,
  liveDrive: false,
  vrmaMixer: null,
  uploadIntent: 'play',
  lipJob: false,
  busy: null,
  busyIdle: 0,
  // Audio2Face-3D: the baked track (frames of ARKit weights + eye degrees)
  // and whether its audio clock is currently driving the face
  a2fTrack: null,
  a2fPlaying: false,
  // In-page MediaPipe face tracker (GNM webcam puppet pipeline). It reads the
  // already-running MJPEG preview, so it never opens a second camera, and it
  // drives the VRM face directly (`retargeter.faceOverride`).
  faceTracker: null,
  faceOn: false,
  faceFps: 0,
  faceDetected: false,
  faceCanvas: null,
  faceLoop: 0,
  faceAt: 0,
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
  // the Audio2Face gaze offsets are measured against the camera direction
  state.retargeter.camera = camera;

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
    // while a bake runs the engine is playing the file being recorded; keep
    // the UI on the user's own source and stay silent
    if (status.source && !state.busy) {
      state.source = status.source;
      if (sourceSelect.value && sourceSelect.value !== status.source &&
          status.state !== 'switching') {
        sourceSelect.value = status.source;
      }
      syncFeed(status.source);
    }
    if (!state.busy && status.audio_url !== state.audioUrl) {
      state.audioUrl = status.audio_url || '';
      applySourceAudio();
    }

    // The generate overlay is driven by the server's job state, not only by a
    // local timer chain (background tabs throttle timers), so it can never sit
    // on a stale label.
    if (state.busy) {
      try {
        const lip = await (await fetch('/api/lipsync',
          { cache: 'no-store' })).json();
        if (lip.state === 'running') {
          state.busyIdle = 0;
          updateBusy(`MuseTalk: ${lip.phase || 'working'}… `
            + `${Math.round(lip.elapsed || 0)}s`);
        } else {
          const job = await (await fetch('/api/bake',
            { cache: 'no-store' })).json();
          if (job.state === 'running') {
            state.busyIdle = 0;
            updateBusy(`Generating VRMA… ${Math.round(job.elapsed || 0)}s`);
          } else if (++state.busyIdle >= 2) {
            hideBusy();
          }
        }
      } catch {
        /* server busy; try again next poll */
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
    // a new source ends any baked playback; a running recording is kept,
    // because starting a bake switches the source first
    state.animation = null;
    startFeed();
    return true;
  } catch (err) {
    showToast(`Could not switch source: ${err.message}`, { error: true });
    return false;
  }
}

sourceSelect.addEventListener('change', () => {
  // the select is also synced from /api/status; only act on a real change
  if (sourceSelect.value === state.source) return;
  switchSource(sourceSelect.value);
});

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

btnUpload.addEventListener('click', () => {
  state.uploadIntent = 'play';
  fileInput.click();
});

fileInput.addEventListener('change', async () => {
  const file = fileInput.files?.[0];
  const intent = state.uploadIntent || 'play';
  state.uploadIntent = 'play';
  if (!file) return;
  const generate = intent === 'generate';
  showToast(`Uploading ${file.name}…`, { sticky: true });
  try {
    const response = await fetch(
      `/api/upload?name=${encodeURIComponent(file.name)}` +
        (generate ? '&select=0' : ''),
      { method: 'POST', body: file },
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    await loadSources();
    hideToast();
    if (generate) {
      // generating is offline: do not take over the live source
      await startGenerate(data.source);
      return;
    }
    sourceSelect.value = data.source;
    state.source = data.source;
    state.poseAt = 0;
    startFeed();
    showToast(`Playing ${data.name} as live pose`);
  } catch (err) {
    showToast(`Upload failed: ${err.message}`, { sticky: true, error: true });
  } finally {
    fileInput.value = '';
  }
});

/* --------------------------------------------------------------- lip sync */
let lipPollTimer = 0;

lipsyncMode.addEventListener('change', () => setLipMode(lipsyncMode.value));

function setLipMode(mode) {
  const wanted = mode === 'musetalk' ? 'musetalk' : 'landmarks';
  if (wanted === 'landmarks') {
    stopLipSync();
    state.lipsyncMode = 'landmarks';
    showToast('Mouth driven by face landmarks');
    return;
  }

  // MuseTalk is a purely offline driver: arming it needs no source and must
  // not start a job or touch realtime tracking. Generate VRMA… is what runs it.
  state.lipsyncMode = 'musetalk';
  state.liveDrive = false;
  syncRealtimeButton();
  showToast('MuseTalk armed - press Generate VRMA… to bake');
}

/** Stop the running MuseTalk job; it saturates the GPU while it runs. */
function stopLipSync() {
  window.clearTimeout(lipPollTimer);
  lipPollTimer = 0;
  fetch('/api/lipsync/cancel', { method: 'POST' }).catch(() => {});
}

/* ------------------------------------------------------- generate backend */
generateWith.addEventListener('change', () => setGenerateWith(generateWith.value));

const GENERATE_BACKENDS = [
  ['flowhmr', 'flowhmr (3d)'],
  ['exmr', 'exmr (3d)'],
];

/** Probe one 3D backend; null means unknown, not broken. */
async function probeBackend(url) {
  try {
    const data = await (await fetch(url, { cache: 'no-store' })).json();
    return data.available || null;
  } catch {
    return null;
  }
}

/**
 * Ask the server whether the 3D backends can run and say so on the picker
 * itself, so a missing checkout/checkpoint is visible before the button is
 * pressed rather than after a long wait.
 */
async function checkGenerateBackends() {
  const [flowhmr, exmr] = await Promise.all([
    probeBackend('/api/flowhmr'),
    probeBackend('/api/exmr'),
  ]);
  state.flowhmrReady = flowhmr;
  state.exmrReady = exmr;
  syncBackendOptions();
  return { flowhmr, exmr };
}

function backendReady(backend) {
  if (backend === 'flowhmr') return state.flowhmrReady;
  if (backend === 'exmr') return state.exmrReady;
  return null;
}

function syncBackendOptions() {
  for (const [value, label] of GENERATE_BACKENDS) {
    const option = generateWith.querySelector(`option[value="${value}"]`);
    const ready = backendReady(value);
    if (option) {
      option.textContent = ready && !ready.ok
        ? `${label} - not ready`
        : label;
    }
  }
  const selected = backendReady(state.generateWith);
  generateWith.title = selected && !selected.ok ? selected.reason : '';
}

function setGenerateWith(backend) {
  state.generateWith = GENERATE_BACKENDS.some(([value]) => value === backend)
    ? backend
    : 'rtmpose';
  // the camera detail only means anything to FlowHMR; ExMR has its own
  // static/moving switch
  flowhmrIntervalPicker.classList.toggle(
    'hidden', state.generateWith !== 'flowhmr');
  exmrCameraPicker.classList.toggle(
    'hidden', state.generateWith !== 'exmr');
  syncGenerateButton();
  syncBackendOptions();
  if (state.generateWith === 'rtmpose') {
    showToast('Generating from the 2D tracker');
    return;
  }
  const label = state.generateWith === 'exmr' ? 'ExMR' : 'FlowHMR';
  showToast(`${label} selected - 3D motion, no face track`);
  checkGenerateBackends().then((ready) => {
    const backend = ready[state.generateWith];
    if (backend && !backend.ok) {
      showToast(`${label} not ready: ${backend.reason}`,
        { sticky: true, error: true });
    }
  });
}

/* --------------------------------------------------------- VRMA (offline) */
function showBusy(text) {
  state.busy = text || 'Generating VRMA…';
  busyText.textContent = state.busy;
  busyEl.classList.remove('hidden');
  syncGenerateButton();
}

function updateBusy(text) {
  if (text) state.busy = text;
  busyText.textContent = state.busy;
}

function hideBusy() {
  state.busy = null;
  busyEl.classList.add('hidden');
  syncGenerateButton();
}

/**
 * Offline: bake a VRM Animation from a video the user picks.
 *
 * This is a separate pipeline from the live one: the server extracts the video
 * on its own tracker (no source switch, no preview, no audio), the browser then
 * replays those frames through the retargeter and stores the result. The screen
 * is blanked for the whole thing.
 */
btnGenerate.addEventListener('click', async () => {
  if (state.busy) {
    await Promise.all([
      fetch('/api/bake/cancel', { method: 'POST' }).catch(() => {}),
      fetch('/api/lipsync/cancel', { method: 'POST' }).catch(() => {}),
      fetch('/api/flowhmr/cancel', { method: 'POST' }).catch(() => {}),
      fetch('/api/exmr/cancel', { method: 'POST' }).catch(() => {}),
      fetch('/api/a2f/cancel', { method: 'POST' }).catch(() => {}),
    ]);
    state.lipJob = false;
    hideBusy();
    return;
  }
  state.uploadIntent = 'generate';
  fileInput.click();
});

function syncGenerateButton() {
  const busy = !!state.busy;
  btnGenerate.classList.toggle('active', busy);
  btnGenerate.textContent = busy ? 'Stop' : 'Generate VRMA…';
  const backendLabel = {
    flowhmr: 'FlowHMR',
    exmr: 'ExMR',
  }[state.generateWith];
  btnGenerate.title = busy
    ? 'Stop generating'
    : (backendLabel
      ? `Pick a video and reconstruct its 3D motion with ${backendLabel}`
      : 'Pick a video and bake it into a .vrma');
}

async function startGenerate(source) {
  if (!source || source === 'none' || source === 'demo' ||
      source.startsWith('camera')) {
    showToast('Pick a video first', { sticky: true, error: true });
    return;
  }

  // The 3D reconstructors run in their own environments. They have no face
  // track, so the lip-sync driver does not apply to these paths.
  if (state.generateWith === 'flowhmr' || state.generateWith === 'exmr') {
    const backend = state.generateWith;
    const ready = (await checkGenerateBackends())[backend];
    if (ready && !ready.ok) {
      const label = backend === 'exmr' ? 'ExMR' : 'FlowHMR';
      showToast(`${label} not ready: ${ready.reason}`,
        { sticky: true, error: true });
      return;
    }
    if (backend === 'exmr') await bakeExMR(source);
    else await bakeFlowHMR(source);
    return;
  }

  if (state.lipsyncMode === 'musetalk') {
    // MuseTalk dubs the video first; its clip is then baked the same way
    state.lipJob = true;
    showBusy('MuseTalk: processing the video…');
    startLipSync(source, source);
    return;
  }

  await bakeVideo(source);
}

async function bakeVideo(source) {
  const stem = source.split('/').pop().replace(/\.[^.]+$/, '');
  showBusy('Generating VRMA…');
  try {
    const start = await fetch('/api/bake', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source }),
    });
    const job = await start.json().catch(() => ({}));
    if (!start.ok) throw new Error(job.error || `HTTP ${start.status}`);

    for (;;) {
      await new Promise((r) => setTimeout(r, 400));
      const status = await (await fetch('/api/bake',
        { cache: 'no-store' })).json();
      if (status.state !== 'running') {
        // Stop was pressed: the job was cancelled on purpose, so there is
        // nothing to report and the toast would only accuse the user
        if (status.state === 'cancelled') return;
        if (status.state !== 'done') {
          throw new Error(status.error || 'bake failed');
        }
        break;
      }
      updateBusy(`Generating VRMA… ${status.elapsed}s`);
    }

    const full = await (await fetch('/api/bake?full=1',
      { cache: 'no-store' })).json();
    const frames = full.frames || [];
    if (frames.length < 2) throw new Error('nothing to bake');
    await recordFrames(frames, `${stem}.vrma`);
    showToast(`VRMA saved: animations/${stem}.vrma`);
  } catch (err) {
    showToast(`VRMA failed: ${err.message}`, { sticky: true, error: true });
  } finally {
    hideBusy();
  }
}

/**
 * Offline: reconstruct 3D motion from a video with FlowHMR, then bake it.
 *
 * FlowHMR runs in its own Python environment (see app/flowhmr_engine.py) and
 * answers with SMPL-H 3D joints rather than 2D keypoints, so the same browser
 * replay that bakes a 2D clip drives the rig from exact directions instead of
 * guessed depth.
 */
async function bakeFlowHMR(source) {
  const stem = source.split('/').pop().replace(/\.[^.]+$/, '');
  showBusy('FlowHMR: loading models…');
  try {
    const start = await fetch('/api/flowhmr', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // VGGT-Omega is the memory hog: every frame peaks near 30 GB, so the
      // picker defaults to every 10th frame with the camera interpolated
      body: JSON.stringify({
        source,
        vggt_interval: Number(flowhmrInterval.value) || 1,
      }),
    });
    const job = await start.json().catch(() => ({}));
    if (!start.ok) throw new Error(job.error || `HTTP ${start.status}`);

    for (;;) {
      await new Promise((r) => setTimeout(r, 500));
      const status = await (await fetch('/api/flowhmr',
        { cache: 'no-store' })).json();
      if (status.state !== 'running') {
        if (status.state === 'cancelled') return;
        if (status.state !== 'done') {
          throw new Error(status.error || 'FlowHMR failed');
        }
        break;
      }
      updateBusy(`FlowHMR: ${status.message || status.phase}… ${status.elapsed}s`);
    }

    const full = await (await fetch('/api/flowhmr?full=1',
      { cache: 'no-store' })).json();
    const frames = full.frames || [];
    if (frames.length < 2) throw new Error('nothing to bake');
    updateBusy(`Retargeting ${frames.length} frames…`);
    // let the overlay paint before the replay blocks the main thread
    await new Promise((r) => setTimeout(r, 30));
    await recordFrames(frames, `${stem}.vrma`);
    showToast(`VRMA saved: animations/${stem}.vrma`);
  } catch (err) {
    showToast(`VRMA failed: ${err.message}`, { sticky: true, error: true });
  } finally {
    hideBusy();
  }
}

/**
 * Offline: reconstruct 3D body/hand/foot motion from a video with ExMR, then
 * bake it.
 *
 * ExMR runs in its own Python environment (see app/exmr_engine.py) and
 * answers with refined SMPL-X 3D joints rather than 2D keypoints. The same
 * browser replay that bakes a 2D clip drives the rig from measured directions.
 */
async function bakeExMR(source) {
  const stem = source.split('/').pop().replace(/\.[^.]+$/, '');
  showBusy('ExMR: loading models…');
  try {
    const start = await fetch('/api/exmr', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source,
        static_cam: exmrCamera.value !== 'moving',
      }),
    });
    const job = await start.json().catch(() => ({}));
    if (!start.ok) throw new Error(job.error || `HTTP ${start.status}`);

    for (;;) {
      await new Promise((r) => setTimeout(r, 500));
      const status = await (await fetch('/api/exmr',
        { cache: 'no-store' })).json();
      if (status.state !== 'running') {
        if (status.state === 'cancelled') return;
        if (status.state !== 'done') {
          throw new Error(status.error || 'ExMR failed');
        }
        break;
      }
      updateBusy(`ExMR: ${status.message || status.phase}… ${status.elapsed}s`);
    }

    const full = await (await fetch('/api/exmr?full=1',
      { cache: 'no-store' })).json();
    const frames = full.frames || [];
    if (frames.length < 2) throw new Error('nothing to bake');
    updateBusy(`Retargeting ${frames.length} frames…`);
    // let the overlay paint before the replay blocks the main thread
    await new Promise((r) => setTimeout(r, 30));
    await recordFrames(frames, `${stem}.vrma`);
    showToast(`VRMA saved: animations/${stem}.vrma`);
  } catch (err) {
    showToast(`VRMA failed: ${err.message}`, { sticky: true, error: true });
  } finally {
    hideBusy();
  }
}

/** Replay extracted frames through the retargeter and store the .vrma. */
async function recordFrames(frames, name) {
  const rt = state.retargeter;
  if (!rt) throw new Error('avatar not loaded');
  const fps = frames[0].fps || 30;
  const out = [];
  for (const frame of frames) {
    rt.update(frame, 1 / fps);
    const captured = captureFrame(frame.t);
    if (captured) out.push(captured);
  }
  rt.reset();
  if (out.length < 2) throw new Error('nothing was tracked in that video');

  const anim = { fps, duration: out[out.length - 1].t, frames: out };
  const save = await fetch(`/api/animation?name=${encodeURIComponent(name)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(anim),
  });
  const data = await save.json().catch(() => ({}));
  if (!save.ok) throw new Error(data.error || `HTTP ${save.status}`);
  return data;
}

/**
 * The realtime button is only meaningful for the landmark driver: MuseTalk is
 * the offline path, so selecting it leaves this button inactive.
 */
function syncRealtimeButton() {
  const recording = !!state.busy;
  const blocked = state.lipsyncMode === 'musetalk' || recording;
  const live = !blocked && state.liveDrive && !state.animation &&
    !state.vrmaMixer;
  btnRealtime.disabled = blocked;
  btnRealtime.title = state.lipsyncMode === 'musetalk'
    ? 'MuseTalk is the offline driver - switch lip sync to face landmarks'
    : (recording
      ? 'Recording a VRMA - stop it first'
      : (live ? 'Click to stop tracking the input'
              : 'Drive the avatar live from face landmarks'));
  btnRealtime.classList.toggle('active', live);
  btnRealtime.textContent = live ? 'Realtime drive on' : 'Start realtime drive';
}

btnRealtime.addEventListener('click', () => {
  if (state.lipsyncMode === 'musetalk') return;
  if (state.liveDrive) {
    // switch tracking off: the avatar stops following and eases to idle
    state.liveDrive = false;
    showToast('Tracking stopped');
  } else {
    state.liveDrive = true;
    state.animation = null;
    state.animStart = 0;
    // hand the bones back from any loaded .vrma to the live retargeter
    if (state.vrmaMixer) {
      state.vrmaMixer.stopAllAction();
      state.vrmaMixer = null;
    }
    showToast('Driving the avatar live from face landmarks');
  }
  syncRealtimeButton();
});

/* ------------------------------------------------- browser face tracker */
/*
 * The GNM webcam puppet's tracker, pointed at the VRM face. MediaPipe runs in
 * this page against the server's already-mirrored MJPEG preview, so it never
 * competes with the Python engine for the camera, and it also works when the
 * source is an uploaded clip.
 *
 * It feeds `retargeter.faceOverride`, which sits above the server's face layer
 * because it carries gaze and per-eye blink that the current WS payload does
 * not. Body tracking is untouched: a close-up with no body in frame still
 * animates the whole face.
 */
const FACE_TRACK_FPS = 30;

function setFaceButton() {
  btnFaceTrack.classList.toggle('active', state.faceOn);
  btnFaceTrack.textContent = state.faceOn ? 'Face tracker on' : 'Face tracker';
  btnFaceTrack.title = state.faceOn
    ? 'Stop the in-page face tracker'
    : 'Track your face in the browser and drive VRM expressions, gaze and ' +
      'head pose';
  if (!state.faceOn && statFace) statFace.textContent = '—';
}

async function startFaceTracker() {
  if (state.faceOn) return;
  if (!feedImg.naturalWidth || state.source === 'none') {
    showToast('Pick a camera or video source first — the face tracker reads ' +
      'the live preview', { error: true });
    return;
  }

  btnFaceTrack.disabled = true;
  btnFaceTrack.textContent = 'Loading face…';
  try {
    const tracker = await FaceTracker.create({
      wasmPath: 'mediapipe/wasm',
      modelPath: 'face_landmarker.task',
    });
    state.faceTracker = tracker;
    state.faceOn = true;
    state.faceDetected = false;
    state.faceAt = 0;
    state.faceFps = 0;
    state.faceCanvas = document.createElement('canvas');
    const canvas = state.faceCanvas;
    const context = canvas.getContext('2d');

    let lastDetect = 0;
    let lastTick = performance.now();
    const loop = () => {
      if (!state.faceOn) return;
      state.faceLoop = requestAnimationFrame(loop);

      const source = feedImg;
      const width = source.naturalWidth;
      const height = source.naturalHeight;
      if (!width || !height) return;

      const now = performance.now();
      if (now - lastDetect < 1000 / FACE_TRACK_FPS) return;
      lastDetect = now;

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      context.drawImage(source, 0, 0, width, height);

      let frame;
      try {
        frame = tracker.detect(canvas, now, width / height);
      } catch (err) {
        // A MediaPipe failure must not leave a half-running rAF loop behind.
        console.error('[face]', err);
        stopFaceTracker();
        showToast(`Face tracker stopped: ${err.message}`,
          { sticky: true, error: true });
        return;
      }
      state.faceDetected = !!frame;
      if (frame) {
        state.faceAt = now;
        if (state.retargeter) {
          state.retargeter.setFaceOverride(faceFrameToVrm(frame));
        }
      } else if (state.retargeter && state.faceAt &&
                 now - state.faceAt > 700) {
        // a lost face must release the last pose rather than freeze it
        state.retargeter.setFaceOverride(null);
      }

      const dt = Math.max(1, now - lastTick) / 1000;
      lastTick = now;
      state.faceFps += (1 / dt - state.faceFps) * 0.15;
    };
    state.faceLoop = requestAnimationFrame(loop);
    setFaceButton();
    showToast('Face tracker on — expressions, blink, gaze and head pose ' +
      'drive the VRM');
  } catch (err) {
    state.faceOn = false;
    state.faceTracker = null;
    setFaceButton();
    showToast(`Face tracker failed: ${err.message}`,
      { sticky: true, error: true });
  } finally {
    btnFaceTrack.disabled = false;
  }
}

function stopFaceTracker() {
  state.faceOn = false;
  state.faceDetected = false;
  if (state.faceLoop) cancelAnimationFrame(state.faceLoop);
  state.faceLoop = 0;
  if (state.faceTracker) state.faceTracker.close();
  state.faceTracker = null;
  state.faceCanvas = null;
  if (state.retargeter) state.retargeter.clearFaceOverride();
  setFaceButton();
  showToast('Face tracker stopped');
}

btnFaceTrack.addEventListener('click', () => {
  if (state.faceOn) stopFaceTracker();
  else startFaceTracker();
});

async function startLipSync(audioRef, faceRef) {
  try {
    const payload = { audio: audioRef };
    if (faceRef) payload.face = faceRef;
    const start = await fetch('/api/lipsync', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const job = await start.json().catch(() => ({}));
    if (!start.ok) throw new Error(job.error || `HTTP ${start.status}`);
    state.lipJob = true;
    showToast('MuseTalk: starting…', { sticky: true });
    pollLipSync();
  } catch (err) {
    showToast(`Lip sync failed: ${err.message}`, { sticky: true, error: true });
  }
}

async function pollLipSync() {
  if (state.lipsyncMode !== 'musetalk') return;
  let status;
  try {
    const response = await fetch('/api/lipsync', { cache: 'no-store' });
    status = await response.json();
  } catch {
    lipPollTimer = window.setTimeout(pollLipSync, 2000);
    return;
  }
  if (status.state === 'running') {
    updateBusy(`MuseTalk: ${status.phase || 'working'}… `
      + `${Math.round(status.elapsed || 0)}s`);
    lipPollTimer = window.setTimeout(pollLipSync, 1500);
    return;
  }
  if (status.state === 'error') {
    state.lipJob = false;
    hideBusy();
    showToast(`Lip sync failed: ${status.error}`,
      { sticky: true, error: true });
    return;
  }
  if (status.state === 'cancelled') {
    state.lipJob = false;
    hideBusy();
    showToast('MuseTalk stopped');
    return;
  }
  if (status.state === 'done') {
    state.lipJob = false;
    updateBusy('Generating VRMA…');
    // MuseTalk's clip goes through the same offline bake (no source switch)
    await bakeVideo(status.output);
  }
}

async function checkLipSync() {
  try {
    const response = await fetch('/api/lipsync', { cache: 'no-store' });
    const data = await response.json();
    const ready = data.available && data.available.ok;
    const option = lipsyncMode.querySelector('option[value="musetalk"]');
    if (option) {
      option.disabled = !ready;
      option.textContent = ready ? 'musetalk' : 'musetalk (unavailable)';
      option.title = ready ? '' : (data.available?.reason || '');
    }
    if (!ready && lipsyncMode.value === 'musetalk') {
      lipsyncMode.value = 'landmarks';
      state.lipsyncMode = 'landmarks';
    }
  } catch {
    /* the server may still be starting */
  }
}

/* ------------------------------------------------------------------ feed */
function startFeed() {
  feedImg.src = `/video?t=${Date.now()}`;
}

feedImg.addEventListener('load', () => feedImg.classList.add('live'));
feedImg.addEventListener('error', () => {
  feedImg.classList.remove('live');
  window.setTimeout(startFeed, 2500);
});

/** With no source the stream goes idle; drop the stale frame. */
function syncFeed(source) {
  const idle = !source || source === 'none';
  if (idle) {
    if (feedImg.getAttribute('src')) {
      feedImg.removeAttribute('src');
      feedImg.classList.remove('live');
    }
  } else if (!feedImg.getAttribute('src')) {
    startFeed();
  }
}

/* --------------------------------------------------------------- controls */
btnReset.addEventListener('click', () => {
  userMovedCamera = false;
  fitCamera(true);
});

btnFull.addEventListener('click', () => {
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else {
    document.documentElement.requestFullscreen?.();
  }
});

/* --------------------------------------------------------------- load VRMA */
/**
 * Play a .vrma on the avatar with three-vrm-animation, exactly like the idle
 * loop, so generated and hand-made animations behave the same.
 */
async function playVRMA(url) {
  if (!state.vrm) throw new Error('avatar not loaded');
  const gltf = await loadGLTF(
    url,
    (parser) => new VRMAnimationLoaderPlugin(parser),
  );
  const vrmAnimation = gltf.userData.vrmAnimations?.[0];
  if (!vrmAnimation) throw new Error('not a VRM animation');

  const clip = createVRMAnimationClip(vrmAnimation, state.vrm);
  if (state.vrmaMixer) state.vrmaMixer.stopAllAction();
  const mixer = new THREE.AnimationMixer(state.vrm.scene);
  mixer.clipAction(clip).play();
  state.vrmaMixer = mixer;
  // replay takes over from live tracking / baked playback
  state.animation = null;
  stopA2f();
  state.liveDrive = false;
  syncRealtimeButton();
  return clip;
}

btnLoadVrma.addEventListener('click', () => vrmaInput.click());

/* -------------------------------------------------------- Audio2Face-3D */
/**
 * Audio-driven facial performance: the server runs the NVIDIA Audio2Face-3D
 * model over an audio file and returns a 60 fps track of ARKit blendshape
 * weights plus eyeball rotations. Playback samples that track against the
 * audio element's clock and feeds the retargeter's face layer, so the avatar
 * lip-syncs (and blinks, brows, gaze) to the recording.
 */

/** Nearest frame of the columnar track for a playback time. */
function sampleA2f(track, t) {
  const i = Math.max(0,
    Math.min(track.frames - 1, Math.round(t * track.fps)));
  const blend = {};
  for (const [name, values] of Object.entries(track.shapes)) {
    const v = values[i];
    if (v > 0) blend[name] = v;
  }
  return { blend, eye: track.eyes[i] };
}

function stopA2f() {
  state.a2fPlaying = false;
  a2fAudio.pause();
  if (state.retargeter) state.retargeter.clearExternalFace();
  btnA2fPlay.textContent = 'Play face';
}

btnA2fPlay.addEventListener('click', () => {
  if (state.a2fPlaying) {
    stopA2f();
    return;
  }
  if (!state.a2fTrack || !state.retargeter) return;
  a2fAudio.currentTime = 0;
  a2fAudio.play().catch(() => {
    showToast('Could not start the audio', { error: true });
  });
  state.a2fPlaying = true;
  btnA2fPlay.textContent = 'Stop face';
});

a2fAudio.addEventListener('ended', () => stopA2f());

/** Upload an audio blob, then ask the server to bake a face track. */
async function processA2fAudio(blob, name) {
  stopA2f();
  showBusy(`Audio2Face: uploading ${name}…`);
  try {
    const upload = await fetch(
      `/api/audio?name=${encodeURIComponent(name)}`,
      { method: 'POST', body: blob });
    const saved = await upload.json().catch(() => ({}));
    if (!upload.ok) throw new Error(saved.error || `HTTP ${upload.status}`);

    const start = await fetch('/api/a2f', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio: saved.audio }),
    });
    const started = await start.json().catch(() => ({}));
    if (!start.ok) throw new Error(started.error || `HTTP ${start.status}`);

    // poll until the bake finishes; progress comes from the status endpoint
    for (;;) {
      await new Promise((r) => setTimeout(r, 600));
      const poll = await (await fetch('/api/a2f',
        { cache: 'no-store' })).json();
      if (poll.state === 'running') {
        updateBusy(`Audio2Face: baking ${Math.round((poll.progress || 0) *
          100)}%`);
        continue;
      }
      if (poll.state === 'error') throw new Error(poll.error || 'bake failed');
      if (poll.state !== 'done') continue;   // cancelled back to idle

      const full = await (await fetch('/api/a2f?full=1',
        { cache: 'no-store' })).json();
      const response = await fetch(full.animation_url,
        { cache: 'no-store' });
      state.a2fTrack = await response.json();
      btnA2fPlay.classList.remove('hidden');
      hideBusy();
      showToast(`Face track ready (${full.model || 'a2f'}, ` +
        `${Number(full.duration).toFixed(1)}s) - press Play face`);
      return;
    }
  } catch (err) {
    hideBusy();
    showToast(`Audio2Face failed: ${err.message}`,
      { sticky: true, error: true });
  }
}

btnA2f.addEventListener('click', () => a2fInput.click());

a2fInput.addEventListener('change', () => {
  const file = a2fInput.files?.[0];
  a2fInput.value = '';
  if (file) processA2fAudio(file, file.name);
});

/** Record from the microphone, then bake a face track from the take. */
let a2fRecorder = null;

btnA2fMic.addEventListener('click', async () => {
  if (a2fRecorder) {
    a2fRecorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    const chunks = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data);
    };
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      a2fRecorder = null;
      btnA2fMic.textContent = 'Record audio';
      btnA2fMic.classList.remove('active');
      if (!chunks.length) return;
      const type = recorder.mimeType || 'audio/webm';
      const ext = type.includes('ogg') ? '.ogg' : '.webm';
      processA2fAudio(new Blob(chunks, { type }), `take_${Date.now()}${ext}`);
    };
    recorder.start();
    a2fRecorder = recorder;
    btnA2fMic.textContent = 'Stop recording';
    btnA2fMic.classList.add('active');
    showToast('Recording - press again to bake the take');
  } catch (err) {
    showToast(`Microphone unavailable: ${err.message}`, { error: true });
  }
});

vrmaInput.addEventListener('change', async () => {
  const file = vrmaInput.files?.[0];
  vrmaInput.value = '';
  if (!file) return;
  showToast(`Loading ${file.name}…`, { sticky: true });
  try {
    const upload = await fetch(
      `/api/vrma?name=${encodeURIComponent(file.name)}`,
      { method: 'POST', body: file },
    );
    const data = await upload.json().catch(() => ({}));
    if (!upload.ok) throw new Error(data.error || `HTTP ${upload.status}`);
    await playVRMA(data.animation);
    showToast(`Playing ${file.name}`);
  } catch (err) {
    showToast(`Could not load VRMA: ${err.message}`,
      { sticky: true, error: true });
  }
});

/* ------------------------------------------------------- animation baking */
const ANIM_EXPRESSIONS = [
  'happy', 'angry', 'sad', 'relaxed', 'surprised',
  'aa', 'ih', 'ou', 'ee', 'oh', 'blink', 'blinkLeft', 'blinkRight',
];

/** Snapshot the avatar's pose so it can be replayed without any model. */
function captureFrame(t) {
  const rt = state.retargeter;
  const em = state.vrm && state.vrm.expressionManager;
  if (!rt || !em) return null;

  const bones = {};
  for (const [name, bone] of Object.entries(rt.b)) {
    if (!bone) continue;
    const q = bone.quaternion;
    bones[name] = [
      Math.round(q.x * 1e4) / 1e4,
      Math.round(q.y * 1e4) / 1e4,
      Math.round(q.z * 1e4) / 1e4,
      Math.round(q.w * 1e4) / 1e4,
    ];
  }
  const expr = {};
  for (const name of ANIM_EXPRESSIONS) {
    const value = em.getValue(name);
    if (value > 0.001) expr[name] = Math.round(value * 1000) / 1000;
  }
  return { t: Math.round(t * 1000) / 1000, bones, expr };
}

/** Bake the pose stream that a clip is producing into a stored animation. */
function applyAnimation(anim, t) {
  const frames = anim.frames;
  if (!frames || frames.length < 2) return;
  const duration = frames[frames.length - 1].t || 1;
  const tt = ((t % duration) + duration) % duration;

  // frames are sorted by time; walk from the middle
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t < tt) lo = mid + 1;
    else hi = mid;
  }
  const frame = frames[Math.max(0, lo)];

  const rt = state.retargeter;
  if (rt) {
    for (const [name, q] of Object.entries(frame.bones)) {
      const bone = rt.b[name];
      if (bone) bone.quaternion.set(q[0], q[1], q[2], q[3]);
    }
  }
  const em = state.vrm && state.vrm.expressionManager;
  if (em) {
    for (const name of ANIM_EXPRESSIONS) {
      em.setValue(name, frame.expr[name] || 0);
    }
  }
}

/* ------------------------------------------------------------------ sound */
/** Play the source clip's audio track; the pose stream itself is silent. */
function applySourceAudio() {
  btnSound.disabled = !state.audioUrl;
  btnSound.classList.toggle('active', state.soundOn);
  if (!state.audioUrl) {
    sourceAudio.pause();
    sourceAudio.removeAttribute('src');
    sourceAudio.load();
    return;
  }
  if (sourceAudio.getAttribute('src') !== state.audioUrl) {
    sourceAudio.src = state.audioUrl;
    sourceAudio.currentTime = 0;
  }
  sourceAudio.muted = !state.soundOn;
  sourceAudio.volume = state.soundOn ? 1 : 0;
  if (state.soundOn) {
    sourceAudio.play().catch(() => {
      // autoplay policy: needs a gesture, which the Sound button provides
      state.soundOn = false;
      btnSound.classList.remove('active');
      showToast('Click Sound to hear the clip');
    });
  }
}

btnSound.addEventListener('click', () => {
  state.soundOn = !state.soundOn;
  localStorage.setItem('mocap.sound', state.soundOn ? 'on' : 'off');
  btnSound.classList.toggle('active', state.soundOn);
  sourceAudio.muted = !state.soundOn;
  sourceAudio.volume = state.soundOn ? 1 : 0;
  if (state.soundOn && state.audioUrl) {
    sourceAudio.play().catch(() => {
      showToast('Could not start audio', { error: true });
    });
  }
});

/* ------------------------------------------------------------------ hud */
let hudAccum = 0;
function updateHud(dt, renderFps) {
  hudAccum += dt;
  if (hudAccum < 0.12) return;
  hudAccum = 0;

  const fresh = state.pose && performance.now() - state.poseAt < 800;
  // the pill reports what the avatar is doing, not what the camera sees
  const driving = state.liveDrive && !state.animation && !state.busy;
  const tracking = driving && fresh && state.detected;
  // the face layer can be locked on while the body is out of frame (close-up)
  const faceLive = driving && fresh && !!(state.pose && state.pose.face &&
    state.pose.face.blend);
  const browserFace = state.faceOn && state.faceDetected &&
    performance.now() - state.faceAt < 500;
  statusPill.classList.toggle('live', tracking || faceLive || browserFace);
  statusPill.classList.toggle('offline', !!state.engineError || !state.wsOpen);

  if (state.engineError) {
    statusText.textContent = 'camera offline';
  } else if (!state.wsOpen) {
    statusText.textContent = 'reconnecting';
  } else if (state.busy) {
    statusText.textContent = 'recording';
  } else if (state.a2fPlaying) {
    statusText.textContent = 'audio2face';
  } else if (state.animation || state.vrmaMixer) {
    statusText.textContent = 'replay';
  } else if (browserFace) {
    statusText.textContent = 'face tracker';
  } else if (state.faceOn) {
    statusText.textContent = 'searching for face';
  } else if (!state.liveDrive) {
    statusText.textContent = 'paused';
  } else if (tracking) {
    statusText.textContent = 'tracking';
  } else if (faceLive) {
    statusText.textContent = 'face only';
  } else {
    statusText.textContent = 'searching for pose';
  }

  statRender.textContent = `${Math.round(renderFps)} fps`;
  statPose.textContent = fresh ? `${state.poseFps.toFixed(0)} fps` : '—';
  statModel.textContent = state.device
    ? `${state.mode} · ${state.device}`
    : state.mode;
  if (statFace) {
    statFace.textContent = !state.faceOn
      ? '—'
      : (browserFace ? `${state.faceFps.toFixed(0)} fps` : 'no face');
  }
  feedMeta.textContent = [state.source, state.resolution]
    .filter(Boolean)
    .join(' · ');

  if (!fresh || !driving) blob.material.opacity = 0.35;
  else blob.material.opacity = 1;

  syncRealtimeButton();
  syncGenerateButton();
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

  if (state.vrmaMixer) state.vrmaMixer.update(dt);
  else if (state.mixer) state.mixer.update(dt);

  const fresh = state.pose && performance.now() - state.poseAt < 800;
  if (state.animation) {
    // baked clip: drive the avatar from recorded data + the audio clock
    const t = (state.audioUrl && !sourceAudio.paused)
      ? sourceAudio.currentTime
      : (performance.now() / 1000 - state.animStart);
    applyAnimation(state.animation, t);
  } else if (state.retargeter) {
    // live mirror only; a generate run drives the rig itself, off-screen
    const drive = state.liveDrive && !state.busy;
    state.retargeter.update(drive && fresh ? state.pose : null, dt);
  }

  // Audio2Face playback: the baked track replaces the face source while its
  // own audio clock runs (the track carries blinks, so the procedural one
  // stands down below)
  if (state.a2fPlaying && state.retargeter && state.a2fTrack) {
    state.retargeter.externalFace =
      sampleA2f(state.a2fTrack, a2fAudio.currentTime);
  }

  if (!state.animation && !state.busy && !state.a2fPlaying) updateBlink(dt);

  // keep the clip's own audio aligned with the frames the engine is showing
  const mediaT = state.pose && state.pose.media_t;
  if (state.audioUrl && !sourceAudio.paused &&
      typeof mediaT === 'number' &&
      performance.now() - state.poseAt < 600 &&
      Math.abs(sourceAudio.currentTime - mediaT) > 0.3) {
    sourceAudio.currentTime = Math.max(0, mediaT);
  }

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
  // A live face source owns the eyelids. The browser tracker writes per-eye
  // blink; the server's face layer writes the averaged preset. Only fall back
  // to the procedural blink when neither has a fresh lock.
  const browserBlink = state.faceOn &&
    performance.now() - state.faceAt < 500;
  const serverBlink = state.liveDrive && !state.animation &&
    !!(state.pose && state.pose.face && state.pose.face.blend);
  if (browserBlink || serverBlink) return;

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
  checkLipSync();
  checkGenerateBackends();
  setFaceButton();
  window.setInterval(pollStatus, 3000);
})();
