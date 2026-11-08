# VRM Motion Capture

https://github.com/user-attachments/assets/8dce8357-1b81-400e-9e33-70f74b9203ce

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
| **Generate VRMA…** | bakes the current clip into a `.vrma`; the **generate with** picker chooses the 2D tracker, FlowHMR or ExMR, and with the lip-sync driver set to `musetalk (offline)` it runs MuseTalk first |
| **Load VRMA…** | plays a `.vrma` file on the avatar (generated elsewhere or hand-made) |
| **Start realtime drive** | toggles live landmark tracking on/off (off by default; disabled while MuseTalk is the driver) |
| **Face tracker** | runs MediaPipe face tracking in the browser and drives the full VRM face — expressions, per-eye blink, gaze and head pose — from the same camera or clip |
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

### Realtime browser face tracker (from the GNM webcam puppet)

**Face tracker** in the actions panel runs a second, independent face layer
entirely in the browser. It is the tracking half of
[`gnm-webcam-puppet`](gnm-webcam-puppet) — MediaPipe's 478 landmarks, 52 ARKit
blendshape scores and the 4×4 head pose — converted to drive **the whole VRM
face** rather than the GNM parametric head:

| tracker output | VRM face channel |
| --- | --- |
| 52 ARKit scores | the 11 expression presets (`aa ih ou ee oh happy angry sad relaxed surprised` + blink) |
| `eyeBlinkLeft` / `eyeBlinkRight` | `blinkLeft` / `blinkRight` separately, so a wink survives |
| 478 landmarks + iris rings | gaze → `vrm.lookAt` target, or the `lookUp/Down/Left/Right` presets on a rig-less avatar |
| 4×4 facial transform | the head bone quaternion, in world space on top of its rest pose |

Two design choices make it coexist with the existing pipeline:

* **It reads the server's MJPEG preview**, not a camera of its own. The Python
  pose engine already owns the webcam, and the preview is mirrored exactly like
  the frames the server's face layer sees — so the browser tracker also works
  when the pose source is an uploaded clip instead of a live camera.
* **It sits above the server face layer only while it has a lock.** `face.js`
  is a thin wrapper around MediaPipe Tasks Vision; the conversion lives in
  `face-retarget.js` with no SDK and no three.js, so `npm test` checks the
  mapping, gaze maths and head-pose extraction without a camera. When the face
  is lost the override expires after 700 ms and expressions ease back to rest.

The runtime is vendored into `web/public/` (gitignored) by
`npm run sync-face-assets`, which also copies the same
`models/face_landmarker.task` the Python layer uses. `npm run dev` and
`npm run build` run it automatically.

```shell
cd web && npm install
npm run sync-face-assets   # once, or whenever the MediaPipe package changes
npm run dev
```

Select a camera or a clip, then press **Face tracker**. The status pill reads
**face tracker** while it is locked; the `face` stat shows its rate. Body
tracking is untouched — a close-up with no shoulders in frame still animates
the face and head.

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

### FlowHMR (3D motion from video)

The **generate with** picker switches the offline generator. `rtmpose (2d)` is
the default path described above — 17 keypoints from the video, depth guessed
from limb foreshortening. `flowhmr (3d)` runs
[FlowHMR](https://github.com/flowhmr/flowhmr) over the same clip instead: it
detects the person, estimates camera motion, extracts body features and
generates **SMPL-H motion** (`poses (T, 52, 3)`, `trans (T, 3)`), so every bone
direction in the bake is measured in 3D rather than inferred.

FlowHMR is a separate environment (Python 3.10 + its own torch stack, plus
YOLOX, VGGT-Omega, SAM 3D Body and the SMPL-H body model - the last three are
gated downloads you have to request). `flowhmr/docs/install.md` has the full
list; the short version, from `flowhmr/`:

```shell
git submodule update --init --recursive
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r tools\web\requirements.txt
.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e third_party\YOLOX
```

Public checkpoints: `yolox_l.pth` (YOLOX release) and
`fafsaf1/flowhmr-0.46B` (`checkpoints/flowhmr_latest/`). Gated ones, after
accepting their licenses and with an `hf` token:
`facebook/VGGT-Omega` (`ckpts/vggt-omega/`) and
`facebook/sam-3d-body-dinov3` (`ckpts/sam-3d-body-dinov3/`). The SMPL-H neutral
model is a manual download from [MANO](https://mano.is.tue.mpg.de/) and belongs
at `assets/body_models/smplh/neutral/model.npz`; the joint regressor is the
public GVHMR asset.

The app finds `<flowhmr>/.venv/Scripts/python.exe` on its own, or an
interpreter written to `flowhmr/.flowhmr_python`:

```shell
set FLOWHMR_PYTHON=C:\path\to\flowhmr\.venv\Scripts\python.exe
set FLOWHMR_DIR=C:\Users\PC\Documents\rtmlib\flowhmr      # default: ./flowhmr
```

`FLOWHMR_CKPT` overrides the checkpoint (default
`checkpoints/flowhmr_latest/flowhmr_latest.ckpt`).

A bake runs the pipeline as a subprocess (`run_v2m_demo_generation.py
--postprocess`), polls its stage (`reading → detecting → camera → features →
generating`), then reads the npz and hands the browser 22 body joints per
frame. The browser
retargets them exactly as it retargets keypoints — same `aim` solver, same VRMA
writer — so a FlowHMR bake and a 2D bake are the same kind of artefact.

Differences worth knowing:

* **No face.** FlowHMR predicts body motion only, so a FlowHMR bake has no
  expressions or head pose, and the lip-sync driver does not apply to it.
* **The person's frame is measured, not assumed.** The rig is aimed in the
  person's own body frame (up = hips → shoulders, right = left → right
  shoulder), mapped so the person's right hand drives the avatar's left. A clip
  that turns around turns the avatar, and the npz's axis convention does not
  matter.
* **Root translation is not baked.** The avatar walks in place: `.vrma` carries
  bone rotations, and `captureFrame` does not store the root position (the 2D
  path has the same limit).
* Feature extraction is cached per video stem in `flowhmr/output/<stem>/`, and
  recomputed with `--overwrite` when the clip is newer than any of its caches.
* A run takes minutes per clip on a GPU and there is no CPU fallback worth
  waiting for; the button turns into **Stop** while it runs.
* **3d camera detail** sets how often VGGT-Omega runs. Every frame is the most
  accurate and peaks near 30 GB of VRAM; every 10th frame (the default, camera
  interpolated in between) is what fits a 24 GB card.

When FlowHMR, its interpreter or its checkpoints are missing, the picker
itself reads **flowhmr (3d) - not ready** with the reason as its tooltip,
selecting it repeats that reason, and the 2D path keeps working. The
availability probe runs at page load, so nothing is uploaded before the app
knows the run cannot start.

### ExMR (refined 3D hands and feet from video)

`exmr (3d)` is the alternative 3D path. [ExMR](https://github.com/twehrbein/ExMR)
reconstructs SMPL-X motion with separate refinement stages for hands (HandMR)
and feet (FootMR), so extremities that a whole-body estimate blurs are measured
from the local image evidence instead. Its smooth, extremity-refined
`joints_global` frames feed the same `update3D` retargeter as FlowHMR.

ExMR is also a separate environment (Python 3.12 + torch + SAM 3D Body +
HaMeR). Setup is ExMR's own `docs/INSTALL.md`; the app finds
`ExMR/.venv/Scripts/python.exe`, or an interpreter written to
`ExMR/.exmr_python`:

```shell
set EXMR_PYTHON=C:\path\to\ExMR\.venv\Scripts\python.exe
set EXMR_DIR=C:\Users\PC\Documents\rtmlib\ExMR      # default: ./ExMR
set EXMR_CHECKPOINT_DIR=C:\path\to\ExMR\checkpoints # optional
```

ExMR needs the ExMR weights, SMPL-X, YOLO11m, Sapiens2, SAM 3D Body and HaMeR
under `ExMR/checkpoints/`; the last two are gated downloads. Runs use ExMR's
`balanced` mode without rendering, write to the app-owned
`ExMR/outputs/_rtmlib/<clip>/`, and are rebuilt when the clip is newer than
that folder. The **3d camera** switch picks `static camera` (default: a fixed
webcam or tripod) or `moving camera`; the camera motion is a model input, so
choose the one that matches the footage.

As with FlowHMR, an ExMR bake has **no face track** (ExMR does not estimate
jaw, eyes or expressions), the person's frame is measured rather than assumed,
and root translation is not stored in the `.vrma`. When ExMR, its interpreter
or its files are missing, the picker reads **exmr (3d) - not ready** with the
reason as its tooltip.

## Docker

The image is self-contained — app code, the built three.js bundle, the pose and
face models, the avatar, and MuseTalk v1.5 with its weights are all baked in.
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` make "no model downloads at
runtime" a guarantee rather than a promise.

```shell
docker build -t camenduru/vrm-motion-capture .
docker run --rm -it --gpus all -p 8000:8000 \
  -v "$PWD/animations:/app/animations" camenduru/vrm-motion-capture
# or simply:  docker compose up --build
```

Then open <http://localhost:8000>. Without `--gpus all` the app still runs —
`--device auto` falls back to CPU with the lightweight models.

To publish it:

```shell
docker login
docker push camenduru/vrm-motion-capture:latest
```

Build from a working tree that has `models/` and `MuseTalk/models/` present:
they are gitignored, so a bare `git clone` does not contain them. Expect a
~12–14 GB image (build context ~5 GB).

Two Python environments live inside: `/opt/app` (onnxruntime-gpu 1.26, i.e.
CUDA 12.8 + cuDNN 9, matching the `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`
base) and `/opt/musetalk` (torch cu126 + diffusers), which
`MuseTalk/.musetalk_python` points at. The `.vrma` files you generate land in
`animations/`; mount that to keep them outside the container.

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
| `GET /api/flowhmr` | FlowHMR job state + whether the checkout/interpreter/checkpoints are ready |
| `POST /api/flowhmr` | `{"source": "videos/clip.mp4"}` → starts a FlowHMR run (3D joints, no face) |
| `POST /api/flowhmr/cancel` | stop the running FlowHMR job |
| `GET /api/exmr` | ExMR job state + whether the checkout/interpreter/checkpoints are ready |
| `POST /api/exmr` | `{"source": "videos/clip.mp4", "static_cam"?: true}` → starts an ExMR run (refined 3D body/hand/foot joints, no face) |
| `POST /api/exmr/cancel` | stop the running ExMR job |
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

Frames that carry `joints3d` (FlowHMR and ExMR) take a parallel path,
`update3D`:

* the person's body frame is measured per frame — `up` is hip midpoint →
  shoulder midpoint, `right` is the person's left → right shoulder — and every
  joint is projected onto it, so the rig is aimed in a frame that travels with
  the person instead of one baked into the npz;
* forward is `up × right`, which is what makes the result a *mirror* rather
  than a rotation: the person's right hand lands on the avatar's left;
* each bone is then aimed at its child exactly, with no foreshortening guess —
  which is the whole point of using a 3D reconstructor;
* the root follows the pelvis delta, scaled by the ratio between the avatar's
  torso and the person's so a short person and a tall avatar cover the same
  ground.

## Layout

```
app/pose_engine.py   capture thread, rtmlib tracking, preview JPEG, pose snapshots
app/flowhmr_engine.py  FlowHMR subprocess + SMPL-H npz -> joints3d frames
app/exmr_engine.py     ExMR subprocess + SMPL-X npz -> joints3d frames
app/server.py        FastAPI: static frontend, MJPEG, websocket, source switching, uploads
web/src/main.js      three.js scene, VRM + VRMA loading, websocket client, UI
web/src/mocap.js     2D → VRM retargeting
web/src/face.js      browser MediaPipe face tracker (GNM webcam puppet port)
web/src/face-retarget.js  landmarks/blendshapes/head pose → VRM face (pure)
web/scripts/         vendored MediaPipe asset sync
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
