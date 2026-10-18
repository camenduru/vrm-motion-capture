"""FastAPI server for the VRM webcam motion-capture app.

Serves the built three.js frontend, the VRM avatar assets, an MJPEG pose
preview stream and a websocket that pushes COCO-17 keypoints at camera rate.

Run:
    python -m app.server                 # starts idle; pick a pose source in the UI
    python -m app.server --source demo   # synthetic skeleton, no camera
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .pose_engine import COCO_KEYPOINTS, VIDEO_EXT, PoseEngine
from . import lipsync, vrma

ROOT = Path(__file__).resolve().parent.parent
WEB_DIST = ROOT / 'web' / 'dist'
VRM_DIR = ROOT / 'vrm_models'
VIDEOS_DIR = ROOT / 'videos'
AUDIO_DIR = ROOT / 'audio'
ANIM_DIR = ROOT / 'animations'
UPLOAD_LIMIT = 512 * 1024 * 1024
AUDIO_EXT = {'.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac', '.opus'}

engine: PoseEngine | None = None
_boundary = 'frame'


def create_app(pose_engine: PoseEngine) -> FastAPI:
    global engine
    engine = pose_engine

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        engine.start()
        try:
            yield
        finally:
            engine.stop()

    app = FastAPI(title='VRM Motion Capture', lifespan=lifespan)
    lip_sync = lipsync.LipSyncJob()

    @app.middleware('http')
    async def revalidate_avatar_assets(request, call_next):
        """Avatar files get swapped in place, so never serve them stale."""
        response = await call_next(request)
        if request.url.path.startswith('/vrm/'):
            response.headers['Cache-Control'] = 'no-cache'
        return response

    @app.get('/api/status')
    async def status():
        data = dict(engine.status())
        # a video source carries its own audio; expose it so the page can play
        # it in sync with the frames (the MJPEG preview is silent)
        path = engine.resolve_source(engine.source)
        data['audio_url'] = (
            f'/videos/{path.name}'
            if path is not None and path.parent == VIDEOS_DIR.resolve()
            else '')
        data['keypoint_names'] = COCO_KEYPOINTS
        return data

    @app.get('/api/avatar')
    async def avatar():
        """Expose the avatar files with a mtime version so swaps are picked up.

        Browsers happily reuse a cached 20+ MB .vrm for hours; the version
        suffix forces a fresh fetch whenever the file changes.
        """

        def describe(path: Path):
            return {
                'name': path.name,
                'url': f'/vrm/{path.name}?v={int(path.stat().st_mtime)}',
            }

        vrm_files = sorted(VRM_DIR.glob('*.vrm'))
        vrma_files = sorted(VRM_DIR.glob('*.vrma'))
        preferred = next((p for p in vrm_files
                          if p.name == 'AvatarSample_B.vrm'), None)
        idle = next((p for p in vrma_files if p.name == 'idle_loop.vrma'),
                    vrma_files[0] if vrma_files else None)
        return {
            'vrm': describe(preferred or vrm_files[0]) if vrm_files else None,
            'vrma': describe(idle) if idle else None,
        }

    @app.post('/api/upload')
    async def upload(name: str, request: Request, select: bool = True):
        """Save an uploaded clip into videos/ and optionally switch to it.

        Byte-identical content reuses the file already stored instead of
        piling up ``_1``/``_2`` copies every time the same clip is added.
        """
        clean = re.sub(r'[^A-Za-z0-9._ -]+', '_', Path(name).name).strip(' .')
        if not clean or Path(clean).suffix.lower() not in VIDEO_EXT:
            return JSONResponse(
                {'error': 'Unsupported file. Use ' +
                          ', '.join(sorted(VIDEO_EXT))},
                status_code=400)

        # stream to scratch space (out of the project) while hashing
        VIDEOS_DIR.mkdir(exist_ok=True)
        tmp = Path(tempfile.gettempdir()) / \
            f'upload_{os.getpid()}_{int(time.time() * 1000)}.part'
        digest = hashlib.sha256()
        written = 0
        try:
            with open(tmp, 'wb') as handle:
                async for chunk in request.stream():
                    written += len(chunk)
                    if written > UPLOAD_LIMIT:
                        raise ValueError('too large')
                    digest.update(chunk)
                    handle.write(chunk)
        except ValueError:
            tmp.unlink(missing_ok=True)
            return JSONResponse(
                {'error': f'File is larger than '
                          f'{UPLOAD_LIMIT // (1024 * 1024)} MB'},
                status_code=413)
        except Exception:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            raise

        if written == 0:
            tmp.unlink(missing_ok=True)
            return JSONResponse({'error': 'Empty upload'}, status_code=400)

        wanted = digest.hexdigest()

        def same_content(path: Path) -> bool:
            if not path.is_file() or path.stat().st_size != written:
                return False
            hasher = hashlib.sha256()
            with open(path, 'rb') as source_file:
                for block in iter(lambda: source_file.read(1 << 20), b''):
                    hasher.update(block)
            return hasher.hexdigest() == wanted

        for existing in sorted(VIDEOS_DIR.iterdir()):
            if existing.suffix.lower() in VIDEO_EXT and same_content(existing):
                tmp.unlink(missing_ok=True)
                source = f'videos/{existing.name}'
                if select:
                    engine.set_source(source)
                print(f'[upload] reusing {existing.name}', flush=True)
                return {'source': source, 'name': existing.name,
                        'bytes': written, 'selected': select, 'reused': True}

        destination = VIDEOS_DIR / clean
        index = 1
        while destination.exists():
            destination = VIDEOS_DIR / f'{Path(clean).stem}_{index}' \
                f'{Path(clean).suffix}'
            index += 1
        shutil.move(str(tmp), str(destination))

        source = f'videos/{destination.name}'
        if select:
            engine.set_source(source)
        print(f'[upload] {destination.name} ({written / 1e6:.1f} MB)', flush=True)
        return {'source': source, 'name': destination.name,
                'bytes': written, 'selected': select}

    @app.get('/api/sources')
    async def sources():
        return {
            'current': engine.source,
            'sources': engine.available_sources(),
        }

    @app.post('/api/source')
    async def set_source(payload: dict):
        source = str(payload.get('source', ''))
        try:
            engine.set_source(source)
        except ValueError as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)
        return {'source': source, 'state': 'switching'}

    @app.post('/api/device')
    async def set_device(payload: dict):
        device = str(payload.get('device', ''))
        try:
            engine.set_device(device)
        except ValueError as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)
        return {'device': device, 'state': 'switching'}

    # ------------------------------------------------------------------ lip sync
    @app.post('/api/audio')
    async def upload_audio(name: str, request: Request):
        """Save an uploaded clip into audio/ for the lip-sync job."""
        clean = re.sub(r'[^A-Za-z0-9._ -]+', '_', Path(name).name).strip(' .')
        if not clean or Path(clean).suffix.lower() not in AUDIO_EXT:
            return JSONResponse(
                {'error': 'Unsupported audio. Use ' +
                          ', '.join(sorted(AUDIO_EXT))},
                status_code=400)

        AUDIO_DIR.mkdir(exist_ok=True)
        destination = AUDIO_DIR / clean
        index = 1
        while destination.exists():
            destination = AUDIO_DIR / f'{Path(clean).stem}_{index}' \
                f'{Path(clean).suffix}'
            index += 1

        written = 0
        try:
            with open(destination, 'wb') as handle:
                async for chunk in request.stream():
                    written += len(chunk)
                    if written > UPLOAD_LIMIT:
                        raise ValueError('too large')
                    handle.write(chunk)
        except ValueError:
            destination.unlink(missing_ok=True)
            return JSONResponse(
                {'error': f'Audio is larger than '
                          f'{UPLOAD_LIMIT // (1024 * 1024)} MB'},
                status_code=413)
        except Exception:  # noqa: BLE001
            destination.unlink(missing_ok=True)
            raise

        if written == 0:
            destination.unlink(missing_ok=True)
            return JSONResponse({'error': 'Empty upload'}, status_code=400)
        return {'audio': f'audio/{destination.name}', 'bytes': written}

    @app.get('/api/lipsync')
    async def lipsync_state():
        return {**lip_sync.status(), 'available': lipsync.availability()}

    @app.post('/api/lipsync/cancel')
    async def lipsync_cancel():
        return lip_sync.cancel()

    @app.post('/api/vrma')
    async def upload_vrma(name: str, request: Request):
        """Store a .vrma the user picked (already a VRM Animation)."""
        clean = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(name).name).strip(' .')
        if not clean:
            return JSONResponse({'error': 'bad name'}, status_code=400)
        if not clean.lower().endswith('.vrma'):
            clean += '.vrma'
        body = await request.body()
        if not body:
            return JSONResponse({'error': 'empty file'}, status_code=400)
        ANIM_DIR.mkdir(exist_ok=True)
        (ANIM_DIR / clean).write_bytes(body)
        print(f'[anim] loaded {clean} ({len(body) / 1e3:.0f} kB)', flush=True)
        return {'animation': f'/animations/{clean}', 'bytes': len(body)}

    @app.post('/api/animation')
    async def save_animation(name: str, request: Request):
        """Store a baked VRM animation as .vrma (or raw .json)."""
        clean = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(name).name).strip(' .')
        if not clean:
            return JSONResponse({'error': 'bad name'}, status_code=400)
        body = await request.body()
        if not body:
            return JSONResponse({'error': 'empty body'}, status_code=400)
        ANIM_DIR.mkdir(exist_ok=True)

        if clean.lower().endswith('.vrma'):
            try:
                data = json.loads(body.decode('utf-8'))
                info = vrma.write_vrma(ANIM_DIR / clean, data)
            except Exception as exc:  # noqa: BLE001
                return JSONResponse({'error': str(exc)}, status_code=400)
            print(f'[anim] saved {clean} ({info["bytes"] / 1e3:.0f} kB, '
                  f'{info["bones"]} bones, {info["expressions"]} expressions)',
                  flush=True)
            return {'animation': f'/animations/{clean}', **info}

        if not clean.lower().endswith('.json'):
            clean += '.json'
        destination = ANIM_DIR / clean
        destination.write_bytes(body)
        print(f'[anim] saved {clean} ({len(body) / 1e3:.0f} kB)', flush=True)
        return {'animation': f'/animations/{clean}', 'bytes': len(body)}

    @app.post('/api/lipsync')
    async def lipsync_start(payload: dict):
        audio_ref = str(payload.get('audio', '')).strip()
        face_ref = str(payload.get('face', '')).strip()
        from_source = audio_ref in ('', 'source')

        if lip_sync.busy():
            return JSONResponse(
                {'error': 'a lip-sync job is already running'},
                status_code=409)
        # scratch space only: nothing from this job lands in the project
        lipsync.reset_work()

        if from_source:
            # "source audio": use the audio track of the clip being played
            source_path = pose_engine.resolve_source(pose_engine.source)
            if source_path is None or source_path.suffix.lower() in \
                    {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}:
                return JSONResponse(
                    {'error': 'the pose source has no audio track - use a '
                              'video, or upload an audio file'},
                    status_code=400)
            try:
                audio = lipsync.extract_source_audio(source_path)
            except Exception as exc:  # noqa: BLE001
                return JSONResponse({'error': str(exc)}, status_code=400)
        else:
            audio = (ROOT / audio_ref).resolve()
            if ROOT not in audio.parents or not audio.exists():
                return JSONResponse({'error': 'audio file not found'},
                                    status_code=400)

        if face_ref:
            face = (ROOT / face_ref).resolve()
            if ROOT not in face.parents or not face.exists():
                return JSONResponse({'error': 'face image not found'},
                                    status_code=400)
            if face.suffix.lower() not in lipsync.IMAGE_EXT:
                # a video identity: trim it to the cap, and if that happened
                # the trimmed clip (which carries the audio) is used for both
                try:
                    face, audio = lipsync.fit_identity(face, audio)
                except Exception as exc:  # noqa: BLE001
                    return JSONResponse({'error': str(exc)}, status_code=400)
        else:
            face = lipsync.work_path(f'face_{int(time.time())}.png')
            if not pose_engine.capture_to(face):
                return JSONResponse(
                    {'error': 'no live frame to use as the face - select a '
                              'pose source first, or pass a face image'},
                    status_code=409)
            # MuseTalk needs an actual face; say so instead of failing later
            if not (pose_engine.pose() or {}).get('face'):
                return JSONResponse(
                    {'error': 'no face in the captured frame - face the '
                              'camera, or pass a face image'},
                    status_code=409)

        try:
            return lip_sync.start(audio, face)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({'error': str(exc)}, status_code=400)

    @app.get('/video')
    async def video():
        async def stream():
            last = -1
            idle = 0
            while True:
                jpeg, seq = engine.snapshot()
                if jpeg is not None and seq != last:
                    last = seq
                    idle = 0
                    yield (b'--' + _boundary.encode() + b'\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'Content-Length: ' + str(len(jpeg)).encode() +
                           b'\r\n\r\n' + jpeg + b'\r\n')
                else:
                    idle += 1
                await asyncio.sleep(1 / 60 if idle < 120 else 0.25)

        return StreamingResponse(
            stream(),
            media_type=f'multipart/x-mixed-replace; boundary={_boundary}',
            headers={'Cache-Control': 'no-store'},
        )

    @app.websocket('/ws/pose')
    async def pose_ws(ws: WebSocket):
        await ws.accept()
        await ws.send_json({
            'type': 'hello',
            'keypoint_names': COCO_KEYPOINTS,
            'mode': engine.mode,
            'source': engine.source,
        })
        last_seq = -1
        try:
            while True:
                pose = engine.pose()
                if pose['seq'] != last_seq:
                    last_seq = pose['seq']
                    await ws.send_json({'type': 'pose', **pose})
                await asyncio.sleep(1 / 90)
        except (WebSocketDisconnect, RuntimeError):
            return

    app.mount('/vrm', StaticFiles(directory=VRM_DIR), name='vrm')
    app.mount('/videos', StaticFiles(directory=VIDEOS_DIR), name='videos')
    ANIM_DIR.mkdir(exist_ok=True)
    app.mount('/animations', StaticFiles(directory=ANIM_DIR), name='animations')

    assets = WEB_DIST / 'assets'
    if assets.exists():
        app.mount('/assets', StaticFiles(directory=assets), name='assets')

    @app.get('/')
    async def index():
        page = WEB_DIST / 'index.html'
        if not page.exists():
            return JSONResponse(
                {'error': 'Frontend not built. Run: cd web && npm install && '
                          'npm run build'},
                status_code=503)
        return FileResponse(page)

    @app.get('/{path:path}')
    async def static_file(path: str):
        target = (WEB_DIST / path).resolve()
        if target.is_file() and WEB_DIST in target.parents:
            return FileResponse(target)
        return FileResponse(WEB_DIST / 'index.html')

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description='VRM webcam motion capture')
    parser.add_argument('--source', default='none',
                        help='none, camera:0, demo, a clip from videos/, or any '
                             'video/image path')
    parser.add_argument('--mode', default='auto',
                        choices=['auto', 'performance', 'balanced',
                                 'lightweight'],
                        help='auto picks performance models on a GPU and '
                             'lightweight on CPU')
    parser.add_argument('--device', default='auto',
                        help='onnxruntime device: auto, cpu, cuda, mps')
    parser.add_argument('--det-frequency', type=int, default=6,
                        help='run person detection every N frames')
    parser.add_argument('--face', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='track facial expressions and head pose '
                             '(default on; --no-face disables)')
    parser.add_argument('--face-every', type=int, default=2,
                        help='run the face model every N frames (default 2)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    if not (WEB_DIST / 'index.html').exists():
        print('[warn] web/dist is missing. Build the frontend first:\n'
              '       cd web && npm install && npm run build', flush=True)

    pose_engine = PoseEngine(source=args.source,
                             mode=args.mode,
                             device=args.device,
                             det_frequency=args.det_frequency,
                             face=args.face,
                             face_every=args.face_every)
    uvicorn.run(create_app(pose_engine),
                host=args.host,
                port=args.port,
                log_level='info')


if __name__ == '__main__':
    main()
