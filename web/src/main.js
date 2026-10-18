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
const fileInput = document.getElementById('file-input');
const lipsyncMode = document.getElementById('lipsync-mode');

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
  btnGenerate.title = busy
    ? 'Stop generating'
    : 'Pick a video and bake it into a .vrma';
}

async function startGenerate(source) {
  if (!source || source === 'none' || source === 'demo' ||
      source.startsWith('camera')) {
    showToast('Pick a video first', { sticky: true, error: true });
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
    showToast(
      `MuseTalk: ${status.phase || 'working'}… ${Math.round(status.elapsed || 0)}s`,
      { sticky: true },
    );
    lipPollTimer = window.setTimeout(pollLipSync, 1500);
    return;
  }
  if (status.state === 'error') {
    state.lipJob = false;
    showToast(`Lip sync failed: ${status.error}`,
      { sticky: true, error: true });
    return;
  }
  if (status.state === 'cancelled') {
    state.lipJob = false;
    showToast('MuseTalk stopped');
    return;
  }
  if (status.state === 'done') {
    state.lipJob = false;
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
  state.liveDrive = false;
  syncRealtimeButton();
  return clip;
}

btnLoadVrma.addEventListener('click', () => vrmaInput.click());

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
  statusPill.classList.toggle('live', tracking || faceLive);
  statusPill.classList.toggle('offline', !!state.engineError || !state.wsOpen);

  if (state.engineError) {
    statusText.textContent = 'camera offline';
  } else if (!state.wsOpen) {
    statusText.textContent = 'reconnecting';
  } else if (state.busy) {
    statusText.textContent = 'recording';
  } else if (state.animation || state.vrmaMixer) {
    statusText.textContent = 'replay';
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

  if (!state.animation && !state.busy) updateBlink(dt);

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
  window.setInterval(pollStatus, 3000);
})();
