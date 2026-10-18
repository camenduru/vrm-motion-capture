"""FastAPI server for the VRM webcam motion-capture app.

Serves the built three.js frontend, the VRM avatar assets, an MJPEG pose
preview stream and a websocket that pushes COCO-17 keypoints at camera rate.

Run:
    python -m app.server                 # webcam 0, GPU if available
    python -m app.server --source demo   # synthetic skeleton, no camera
"""

from __future__ import annotations

import argparse
import asyncio
import re
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .pose_engine import COCO_KEYPOINTS, VIDEO_EXT, PoseEngine

ROOT = Path(__file__).resolve().parent.parent
WEB_DIST = ROOT / 'web' / 'dist'
VRM_DIR = ROOT / 'vrm_models'
VIDEOS_DIR = ROOT / 'videos'
UPLOAD_LIMIT = 512 * 1024 * 1024

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

    @app.middleware('http')
    async def revalidate_avatar_assets(request, call_next):
        """Avatar files get swapped in place, so never serve them stale."""
        response = await call_next(request)
        if request.url.path.startswith('/vrm/'):
            response.headers['Cache-Control'] = 'no-cache'
        return response

    @app.get('/api/status')
    async def status():
        return {
            **engine.status(),
            'keypoint_names': COCO_KEYPOINTS,
        }

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
    async def upload(name: str, request: Request):
        """Save an uploaded clip into videos/ and switch to it."""
        clean = re.sub(r'[^A-Za-z0-9._ -]+', '_', Path(name).name).strip(' .')
        if not clean or Path(clean).suffix.lower() not in VIDEO_EXT:
            return JSONResponse(
                {'error': 'Unsupported file. Use ' +
                          ', '.join(sorted(VIDEO_EXT))},
                status_code=400)

        VIDEOS_DIR.mkdir(exist_ok=True)
        destination = VIDEOS_DIR / clean
        index = 1
        while destination.exists():
            destination = VIDEOS_DIR / f'{Path(clean).stem}_{index}' \
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
                {'error': f'File is larger than '
                          f'{UPLOAD_LIMIT // (1024 * 1024)} MB'},
                status_code=413)
        except Exception:  # noqa: BLE001
            destination.unlink(missing_ok=True)
            raise

        if written == 0:
            destination.unlink(missing_ok=True)
            return JSONResponse({'error': 'Empty upload'}, status_code=400)

        source = f'videos/{destination.name}'
        engine.set_source(source)
        print(f'[upload] {destination.name} ({written / 1e6:.1f} MB)', flush=True)
        return {'source': source, 'name': destination.name,
                'bytes': written}

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
    parser.add_argument('--source', default='camera:0',
                        help='camera:0, demo, a clip from videos/, or any '
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
