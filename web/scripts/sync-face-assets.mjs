/**
 * Vendors MediaPipe's face-tracking runtime into `public/` so the page is
 * self-contained.
 *
 * Pointing FilesetResolver at a CDN is the usual shortcut. Vendoring instead
 * keeps the app working offline, pins the WASM to the same npm package the
 * JS runtime came from -- a mismatch fails at runtime, not at build -- and
 * means no third-party host sees a request from a page that is about to read
 * a camera or video frame.
 *
 * The model is the same `face_landmarker.task` the Python face layer uses, so
 * the browser tracker and the server tracker are looking at one face model.
 *
 * Run from web/:  npm run sync-face-assets
 */

import { copyFile, cp, mkdir, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const WASM_SOURCE = join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const WASM_TARGET = join(root, 'public', 'mediapipe', 'wasm');

const MODEL_SOURCE = join(root, '..', 'models', 'face_landmarker.task');
const MODEL_TARGET = join(root, 'public', 'face_landmarker.task');

// Download URL printed when the model is missing; keeping it out of the code
// path means a fresh clone still builds.
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/' +
  'face_landmarker/face_landmarker/float16/1/face_landmarker.task';

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  if (!(await exists(WASM_SOURCE))) {
    throw new Error(`Missing ${WASM_SOURCE}. Run npm install first.`);
  }
  await mkdir(WASM_TARGET, { recursive: true });
  await cp(WASM_SOURCE, WASM_TARGET, { recursive: true });
  console.log(`wasm    -> ${WASM_TARGET}`);

  if (!(await exists(MODEL_SOURCE))) {
    console.warn(
      `[face] missing ${MODEL_SOURCE}\n` +
      `[face] the browser face tracker will be unavailable until it is ` +
      `downloaded:\n[face] ${MODEL_URL}\n` +
      '[face] or: python scripts/download_models.py --set face',
    );
    return;
  }
  await mkdir(dirname(MODEL_TARGET), { recursive: true });
  await copyFile(MODEL_SOURCE, MODEL_TARGET);
  const { size } = await stat(MODEL_TARGET);
  console.log(`model   -> ${MODEL_TARGET} (${(size / 1e6).toFixed(1)} MB)`);
}

await main();
