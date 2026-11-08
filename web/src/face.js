/**
 * MediaPipe face tracking, wrapped so the rest of the app never sees the SDK.
 *
 * Ported from the GNM webcam puppet (`gnm-webcam-puppet/web_puppet/src/
 * tracking.ts`) and pointed at a VRM avatar instead of a parametric head.
 * The landmarker asks for all three outputs it can produce, because they fail
 * in different ways and a VRM face needs all of them:
 *
 *   landmarks    478 points, including the two 5-point iris rings. Geometric,
 *                so gaze can be read straight off them.
 *   blendshapes  52 ARKit-style scores from a trained classifier. Far more
 *                reliable than eyelid geometry for blink.
 *   transform    a 4x4 head pose in camera space, which becomes the VRM head
 *                bone rotation without a per-frame pose solve.
 *
 * The tracker accepts any MediaPipe image source. In this app it is fed the
 * server's mirrored MJPEG preview, so the browser never opens a second camera
 * (the Python pose engine already owns it) and a video-file source works too.
 * The preview is already selfie-mirrored, exactly like the frames the server's
 * own face layer sees.
 */

import {
  FaceLandmarker,
  FilesetResolver,
} from '@mediapipe/tasks-vision';

import { gazeFromLandmarks } from './face-retarget.js';

export class FaceTracker {
  constructor(landmarker) {
    this.landmarker = landmarker;
    this.lastTimestamp = -1;
    this.delegate = 'GPU';
  }

  /**
   * Loads the WASM runtime and the face landmarker.
   *
   * @param {object} options
   * @param {string} [options.wasmPath] Directory holding the vendored WASM.
   * @param {string} [options.modelPath] The `face_landmarker.task` bundle.
   * @param {'GPU'|'CPU'} [options.delegate] Preferred MediaPipe delegate.
   * @returns {Promise<FaceTracker>}
   */
  static async create({
    wasmPath = 'mediapipe/wasm',
    modelPath = 'face_landmarker.task',
    delegate = 'GPU',
  } = {}) {
    const fileset = await FilesetResolver.forVisionTasks(wasmPath);

    const build = async (preferred) => FaceLandmarker.createFromOptions(
      fileset,
      {
        baseOptions: { modelAssetPath: modelPath, delegate: preferred },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      },
    );

    let landmarker;
    let used = delegate;
    try {
      landmarker = await build(delegate);
    } catch (error) {
      // Some browsers expose WebGL but refuse the GPU delegate for wasm. The
      // CPU path is slower, not broken, so fall back rather than losing the
      // face layer entirely.
      if (delegate === 'CPU') throw error;
      console.warn('[face] GPU delegate unavailable, using CPU', error);
      landmarker = await build('CPU');
      used = 'CPU';
    }

    const tracker = new FaceTracker(landmarker);
    tracker.delegate = used;
    return tracker;
  }

  /**
   * Runs detection on one image.
   *
   * @param {HTMLVideoElement|HTMLCanvasElement|ImageBitmap} source
   * @param {number} timestampMs Monotonically increasing timestamp.
   * @param {number} [aspect] Source width / height.
   * @returns {object|null} `{landmarks, blendshapes, transform, gaze, aspect}`
   */
  detect(source, timestampMs, aspect = 1) {
    // MediaPipe's video mode rejects a timestamp that does not advance, which
    // happens whenever the render loop outruns the source frame rate.
    if (timestampMs <= this.lastTimestamp) return null;
    this.lastTimestamp = timestampMs;

    const result = this.landmarker.detectForVideo(source, timestampMs);
    const landmarks = result.faceLandmarks?.[0];
    if (!landmarks || landmarks.length === 0) return null;

    const blendshapes = new Map();
    for (const category of result.faceBlendshapes?.[0]?.categories ?? []) {
      blendshapes.set(category.categoryName, category.score);
    }

    const matrix = result.facialTransformationMatrixes?.[0];
    return {
      landmarks,
      blendshapes,
      transform: matrix ? Float32Array.from(matrix.data) : null,
      gaze: gazeFromLandmarks(landmarks, aspect),
      aspect,
    };
  }

  close() {
    try {
      this.landmarker.close();
    } catch {
      /* already closed */
    }
  }
}
