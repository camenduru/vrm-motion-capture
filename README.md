# VRM Motion Capture

Real-time webcam motion capture that drives a **VRM avatar** with **rtmlib**
(RTMPose) pose estimation, rendered in the browser with **three.js** +
`@pixiv/three-vrm`.

```
webcam / video file ──► rtmlib PoseTracker (ONNX Runtime) ──► COCO-17 keypoints
        │                              │
        │ MJPEG preview                │ websocket (~30 Hz)
        ▼                              ▼
   browser HUD  ◄──────────────  retarget 2D → VRM humanoid bones → three.js
```

No camera? Every clip in [`videos/`](videos) can be used as a *live pose
source*, and any video file can be added straight from the UI. The recommended
test clip is `videos/green screen dancing girl 01.mp4`.

## Quick start

```shell
# 1. Python environment
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt      # Windows
# .venv/bin/python -m pip install -r requirements.txt            # macOS / Linux

# 2. Pose models (~200 MB, downloaded once into models/)
.venv\Scripts\python.exe scripts\download_models.py

# 3. Frontend (three.js bundle, served by the Python server)
cd web && npm install && npm run build && cd ..

# 4. Run it (webcam 0 by default)
start.bat
# or: .venv\Scripts\python.exe -m app.server
```

Only the app's own source is tracked (see [`.gitignore`](.gitignore)): the
virtualenv, `web/node_modules/`, the built frontend, the ONNX models and both
third-party checkouts are ignored. A fresh clone therefore needs them fetched
before step 1:

```shell
git clone https://github.com/Tau-J/rtmlib.git rtmlib   # pose library, pip -e ./rtmlib
# RTMPose/ is only a local scratch checkout — nothing in app/ reads it.
```

Then open <http://127.0.0.1:8000>.

`start.bat` opens the browser and forwards any arguments to the server:

```shell
start.bat --source "videos/green screen dancing girl 01.mp4"   # live pose from a clip
start.bat --source demo                                        # synthetic skeleton
start.bat --source camera:1 --mode balanced                    # second webcam, best accuracy
```

## Using it

| Control | What it does |
| --- | --- |
| **pose source** | `None` (default), webcams, the synthetic demo, and every clip in `videos/` - switched without restarting |
| **compute** | rebuild the models on `auto` / `gpu (cuda)` / `cpu` without restarting; `auto` uses the accurate models on a GPU and the light ones on CPU |
| **Add video…** | upload a clip (`.mp4 .webm .mov .mkv`, max 512 MB) — it is stored in `videos/` and used as the live pose source |
| **Generate VRMA…** | bakes the current clip into a `.vrma`; with the lip-sync driver set to `musetalk (offline)` it runs MuseTalk first |
| **Load VRMA…** | plays a `.vrma` file on the avatar (generated elsewhere or hand-made) |
| **Start realtime drive** | toggles live landmark tracking on/off (off by default; disabled while MuseTalk is the driver) |
| **Feed / Reset view / Fullscreen** | hide the pose preview, re-frame the avatar, go fullscreen |
| left-drag / right-drag / scroll | orbit · slide the avatar around the frame · zoom |

The avatar mirrors you: raise your right hand and the arm on the right of the
screen moves. Tracking runs while your shoulders and hips are visible; when
tracking is lost the avatar falls back to the `idle_loop.vrma` animation and
fades back in when you are detected again. The face is driven from the same
frames — expressions and head pose — so a close-up alone still animates the
avatar and the status pill reads **face only**. A small HUD shows render fps,
pose fps and the active model preset.

## Command line

```shell
python -m app.server [options]

--source    none | camera:0 | demo | videos/<clip> | path/to/video-or-image   (default none)
--mode      auto | performance | balanced | lightweight   (default auto: performance on GPU, lightweight on CPU)
--device    auto | cpu | cuda | mps         (default auto: CUDA when available)
--det-frequency  N   run person detection every N frames (default 6)
--face-every    N   run the face model every N frames (default 2)
--face / --no-face  facial expressions + head pose (default on)
--host / --port      default 127.0.0.1:8000
```

**`auto`** (the default) runs `performance` when the models are on a GPU and
`lightweight` on CPU:

| preset | detector | pose model | measured on RTX 3090 |
| --- | --- | --- | --- |
| `performance` | YOLOX-x 640×640 | RTMPose-x 384×288 | 16.7 ms (60 fps) |
| `balanced` | YOLOX-m 640×640 | RTMPose-m 256×192 | 9.0 ms (111 fps) |
| `lightweight` | YOLOX-tiny 416×416 | RTMPose-s 256×192 | 6.7 ms (150 fps) |

Those are full per-frame costs at 1280×720 with detection included; the same
`lightweight` pair on CPU is about 16 ms. Every preset clears 30 fps on a GPU,
so the engine paces itself to the source and reports its real rate.

`performance` is worth it when accuracy matters: on the same frames all 17
joints resolve, and the 384×288 pose model holds up much better than 256×192
when the person is small in frame or moving fast.

### GPU inference

`onnxruntime-gpu` needs a CUDA/cuDNN runtime that it does not bundle, and the
CUDA major version is fixed by the ORT build. The PyPI build of
`onnxruntime-gpu` 1.26 is CUDA 12.8 + cuDNN 9; 1.27+ switched to CUDA 13,
which has no Windows wheels yet:

```shell
.venv\Scripts\python.exe -m pip uninstall -y onnxruntime
.venv\Scripts\python.exe -m pip install "onnxruntime-gpu>=1.26,<1.27" ^
    nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12
```

The engine puts those DLLs on the search path itself (cuDNN 9 is split across
several DLLs, and ORT only preloads the main one) and logs what it settled on:

```
[pose] ready: balanced on cuda (CUDAExecutionProvider, CPUExecutionProvider)
```

If the GPU runtime is broken it falls back to CPU automatically instead of
failing, and the HUD shows the active preset and device (`balanced · cuda`).

`--det-frequency` trades detector cost against how often the tracker re-finds
you: the pose model runs every frame, person detection only every Nth frame.

### Face tracking

The face layer runs MediaPipe's FaceLandmarker on the same (already mirrored)
frames. With every pose it sends the head-pose quaternion and the 52
ARKit-style blendshape coefficients, which the web layer maps onto the VRM
expressions:

| blendshapes | expression |
| --- | --- |
| `eyeBlinkLeft` / `Right` | `blink` |
| `jawOpen` | `aa` |
| `mouthPucker` | `ou` |
| `jawOpen` × `mouthFunnel` | `oh` |
| `mouthStretchLeft` / `Right` | `ee` |
| `mouthSmileLeft` / `Right` | `happy` |
| `browDownLeft` / `Right` | `angry` |
| `mouthFrown*` + `browInnerUp` | `sad` |
| `browInnerUp` + `eyeWide*` | `surprised` |

While a face is locked the head bone follows the measured head pose; when the
face is lost the body keypoints take the head back over. A close-up with no
body still animates the face — the status pill then reads **face only**.

The model is a separate download:

```shell
python scripts/download_models.py --set face
```

Face inference costs ~9 ms a frame, so it runs every `--face-every` frames
(default 2) and reuses the last result in between — that keeps a 30 fps source
at 30 fps. `--no-face` turns the layer off completely.

### VRM animation generation (MuseTalk v1.5)

There are two distinct paths, and MuseTalk is **only** used by the offline one.

**Realtime** — the **Start realtime drive** button (off by default) drives the
avatar continuously from face landmarks. The status pill reads `tracking` while
it follows the input and `paused` when it is off.

**Offline** — **Generate VRMA…** bakes the current clip into a VRM Animation
(`animations/<clip>.vrma`: a glTF binary with `VRMC_vrm_animation`, carrying
every humanoid bone rotation and expression weight per frame). **Load VRMA…**
plays any `.vrma` on the avatar. The **lip sync** dropdown chooses what drives
the mouth while baking:

| driver | what Generate VRMA… does |
| --- | --- |
| `face landmarks` | bakes straight from the clip's own face |
| `musetalk (offline)` | runs MuseTalk v1.5 on the clip's audio first, then bakes its result |

MuseTalk never runs on its own — selecting it only arms the offline driver, and
the realtime button is disabled while it is armed. The baked animation replays
against the clip's audio clock, so it needs no models or GPU. MuseTalk's face
preprocessing is cached per identity; because it processes every identity frame
(~0.5 s each), an identity video is capped at 900 frames.

MuseTalk needs its own interpreter: this app runs Python 3.13 while MuseTalk's
torch/diffusers stack lives on 3.10/3.11. One-time setup:

```shell
scripts\setup_musetalk.bat
```

It records the interpreter in `MuseTalk/.musetalk_python` and downloads the
v1.5 weights (~4 GB) into `MuseTalk/models`. Point `MUSETALK_PYTHON` at a
different interpreter if you keep one elsewhere. When MuseTalk or its weights
are missing the option is disabled and its tooltip says why — the rest of the
app is unaffected.

A few local patches make the checkout work with current dependencies: no
`mmpose`/`mmcv`/`mmdet` (the face crop box is computed from MediaPipe
landmarks instead of DWPose), `np.int` → `int`, and `weights_only=False` for
the legacy checkpoints under torch ≥ 2.6.

## API

| endpoint | purpose |
| --- | --- |
| `GET /` | the app (built frontend) |
| `GET /video` | MJPEG pose preview (skeleton drawn server-side) |
| `GET /api/status` | engine state, source, resolution, fps, detection, face on/off, active device + preset |
| `GET /api/sources` | pickable pose sources |
| `POST /api/source` | `{"source": "videos/clip.mp4"}` |
| `POST /api/device` | `{"device": "auto" \| "cuda" \| "cpu"}` — rebuilds the models on that device |
| `POST /api/audio` | raw audio body → saved in `audio/` for lip sync |
| `GET /api/lipsync` | lip-sync job state + whether MuseTalk is installed |
| `POST /api/lipsync` | `{"audio": "source", "face"?: "videos/clip.mp4"}` → starts a MuseTalk job (only used by Generate VRMA…) |
| `POST /api/lipsync/cancel` | stop the running MuseTalk job |
| `POST /api/animation?name=x.vrma` | bakes posted frames into `animations/x.vrma` |
| `POST /api/vrma?name=x.vrma` | raw `.vrma` upload, served from `animations/` |
| `POST /api/upload?name=clip.mp4` | raw video body → saved in `videos/` and selected |
| `GET /api/avatar` | versioned URLs for the `.vrm` / `.vrma` files |
| `WS /ws/pose` | keypoint stream: `{seq, t, fps, detected, keypoints[17][2], scores[17], width, height}` |

Keypoints are normalized `0..1` on the **mirrored** frame, in COCO order:
`nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle`.
Every frame is flipped before inference to give the selfie preview, so the
model's raw left/right labels would land on the opposite side of the body; the
engine swaps those pairs back before publishing, and the indices therefore
always describe the real person (as seen in the mirrored preview).

## How the retargeting works

`web/src/mocap.js` turns 2D keypoints into VRM bone rotations:

* every bone is *aimed*: the vector from a bone to its child joint is rotated
  onto the vector measured in the image, converted to world space and applied
  through the parent's rotation, so any rig rest pose works;
* the mirrored preview means the person's right side maps to the avatar's left
  side (the avatar faces the camera);
* limb depth comes from foreshortening — a projected limb that is shorter than
  the longest that limb has been seen to project (measured as a fraction of the
  torso, so it is independent of the avatar's proportions) pushes the joint
  toward the camera, which is what makes reaching forward work with a single
  camera. The reference self-calibrates per limb, creeping up toward the longest
  projection seen while rate-limiting the rise so a stray keypoint cannot warp
  every frame afterwards; a genuinely foreshortened limb is still recovered;
* hips, spine, chest and head come from the shoulder/hip line, the torso
  direction and the ear/nose geometry; the root follows the hip center;
* the hip and shoulder lines are *lateral* pairs, so they reverse when the
  person turns their back to the camera. The pelvis and head angles derived
  from them are pinned to the half space that keeps the avatar upright, and
  the head's mirroring follows the detected facing, so turning around does not
  fold or flip the avatar;
* every input is speed-adaptively smoothed — a One Euro filter, which smooths
  hard while a joint is still and loosens as it speeds up, so fast swings track
  without the lag a fixed time constant would add. Joints below a confidence
  threshold hold their last value, and everything cross-fades to the idle
  animation when tracking stops.

## Layout

```
app/pose_engine.py   capture thread, rtmlib tracking, preview JPEG, pose snapshots
app/server.py        FastAPI: static frontend, MJPEG, websocket, source switching, uploads
web/src/main.js      three.js scene, VRM + VRMA loading, websocket client, UI
web/src/mocap.js     2D → VRM retargeting
videos/              demo live-pose clips (also where uploads land)
models/              ONNX checkpoints downloaded by scripts/download_models.py
vrm_models/          avatar (.vrm) and idle animation (.vrma)
scripts/             model/video downloaders and the source smoke test
```

## Tests

```shell
# switch every source and report detection rate, confidence and motion
.venv\Scripts\python.exe scripts/check_sources.py --seconds 4

# unit tests of the rtmlib checkout (pose models, trackers, file mirror)
.venv\Scripts\python.exe -m pytest rtmlib/tests -q
```

## Troubleshooting

| symptom | fix |
| --- | --- |
| `Could not open camera index 0` | another app holds the camera, or Windows camera permission is off (Settings → Privacy → Camera). Try `--source camera:1`. |
| Pose fps in single digits | the GPU is not being used — check the startup line `[pose] ready: <mode> on <device>`; if it says `cpu`, see **GPU inference** above. |
| Avatar is stale / old model appears | the avatar URL is versioned by file mtime (`GET /api/avatar`) and `/vrm/*` is served `no-cache`; a hard reload clears anything older. |
| `Missing pose models` | run `python scripts/download_models.py` (add `--set balanced` for the slower, more accurate pair). |
| Uploaded clip will not play | the source must be a real video file the browser can hand over — check the server log, the toast shows the decoder error. |
