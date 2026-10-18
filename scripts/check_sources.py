"""Smoke test: switch every pose source and report what the engine sees.

Assumes the app is already running (`python -m app.server`). For each source it
switches over HTTP, listens on the pose websocket for a few seconds and prints
detection rate, keypoint confidence and frame rate.

Usage:
    python scripts/check_sources.py [--url http://127.0.0.1:8000] [--seconds 4]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import urllib.request

import websockets

STABLE = [5, 6, 11, 12, 7, 8, 13, 14, 15, 16]


def http_json(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload else None
    request = urllib.request.Request(
        url, data=data,
        headers={'Content-Type': 'application/json'} if data else {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


async def sample(ws_url: str, seconds: float) -> dict:
    frames = 0
    detected = 0
    scores = []
    motion = []
    previous = None
    async with websockets.connect(ws_url, max_size=None) as ws:
        deadline = asyncio.get_event_loop().time() + seconds
        while asyncio.get_event_loop().time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                break
            message = json.loads(raw)
            if message.get('type') != 'pose':
                continue
            frames += 1
            if message.get('detected'):
                detected += 1
                kp = message['keypoints']
                sc = message['scores']
                scores.append(statistics.fmean(sc[i] for i in STABLE))
                if previous is not None:
                    motion.append(statistics.fmean(
                        abs(kp[i][0] - previous[i][0]) +
                        abs(kp[i][1] - previous[i][1]) for i in STABLE))
                previous = kp
            else:
                previous = None
    return {
        'frames': frames,
        'detection_rate': detected / frames if frames else 0.0,
        'confidence': statistics.fmean(scores) if scores else 0.0,
        'motion': statistics.fmean(motion) if motion else 0.0,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--seconds', type=float, default=4.0)
    parser.add_argument('--only', default=None,
                        help='substring filter on the source id')
    args = parser.parse_args()

    ws_url = args.url.replace('http', 'ws', 1) + '/ws/pose'
    sources = http_json(args.url + '/api/sources')['sources']
    if args.only:
        sources = [s for s in sources if args.only in s['id']]

    print(f'{"source":52s} {"detect":>7s} {"conf":>6s} {"motion":>7s}')
    for source in sources:
        try:
            http_json(args.url + '/api/source', {'source': source['id']})
        except urllib.error.HTTPError as exc:
            print(f'{source["id"][:52]:52s}  switch failed: {exc}')
            continue
        await asyncio.sleep(2.5)  # let the source open and models settle
        result = await sample(ws_url, args.seconds)
        print(f'{source["id"][:52]:52s} '
              f'{result["detection_rate"] * 100:6.0f}% '
              f'{result["confidence"]:6.2f} '
              f'{result["motion"]:7.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
