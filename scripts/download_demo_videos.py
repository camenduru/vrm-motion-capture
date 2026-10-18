"""Download small, freely licensed demo clips used as live pose sources.

The app can run the pose pipeline on a video file instead of a webcam, which
is handy for trying the avatar without a camera. Clips come from Wikimedia
Commons; attribution and licence info is written to videos/README-commons.md.

Usage:
    python scripts/download_demo_videos.py
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'videos'
API = 'https://commons.wikimedia.org/w/api.php'
UA = 'vrm-mocap-demo/1.0 (local motion capture demo)'

FILES = [
    'File:Serbian dancing lady video.webm',
    'File:Serpentine Dance (1895) - yt.webm',
    'File:DANCE PRAISE AND WORSHIP PART 2C-02.webm',
    'File:Dance sport in secondary schools.webm',
]


def api_get(params: dict) -> dict:
    url = API + '?' + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={'User-Agent': UA})
    tmp = dest.with_suffix(dest.suffix + '.part')
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get('Content-Length') or 0)
        done = 0
        with open(tmp, 'wb') as f:
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
        print(f'        {done / 1e6:.1f} MB downloaded'
              + (f' of {total / 1e6:.1f} MB' if total else ''))
    tmp.replace(dest)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = api_get({
        'action': 'query',
        'format': 'json',
        'prop': 'imageinfo',
        'iiprop': 'url|size|extmetadata',
        'titles': '|'.join(FILES),
    })

    rows = []
    for page in data['query']['pages'].values():
        info = page.get('imageinfo', [{}])[0]
        source = info.get('url', '').split('?')[0]
        if not source:
            print(f'[skip] {page.get("title")} (no file url)')
            continue
        name = urllib.parse.unquote(source.rsplit('/', 1)[-1])
        name = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
        dest = OUT_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f'[skip] {name} already present')
        else:
            print(f'[get ] {name} ({info.get("size", 0) / 1e6:.1f} MB)')
            fetch(source, dest)
            time.sleep(3)  # be polite to the Commons rate limiter

        meta = info.get('extmetadata', {})
        title = re.sub('<[^>]+>', '', meta.get('ObjectName', {}).get('value', ''))
        artist = re.sub('<[^>]+>', '', meta.get('Artist', {}).get('value', ''))
        rows.append({
            'name': name,
            'title': title or name,
            'artist': artist.strip() or 'unknown',
            'license': meta.get('LicenseShortName', {}).get('value', 'unknown'),
            'page': 'https://commons.wikimedia.org/wiki/'
                    + urllib.parse.quote(page['title'].replace(' ', '_')),
        })

    lines = [
        '# Demo clips',
        '',
        'Short, freely licensed clips from Wikimedia Commons used as live pose',
        'sources. Run the app with one of them instead of a webcam:',
        '',
        '```shell',
        f'python -m app.server --source videos/{rows[0]["name"]}'
        if rows else 'python -m app.server --source videos/<clip>',
        '```',
        '',
        'Every clip in this folder is listed in the pose-source picker in the',
        'web UI (and any clip can be added there with the "Add video…" button).',
        '',
        '| file | title | author | licence | source |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in rows:
        lines.append(
            f'| `{row["name"]}` | {row["title"]} | {row["artist"]} | '
            f'{row["license"]} | [Commons]({row["page"]}) |')
    lines.append('')
    # a separate file so the folder's own README is never overwritten
    (OUT_DIR / 'README-commons.md').write_text('\n'.join(lines),
                                               encoding='utf-8')
    print(f'\nSaved {len(rows)} clip(s) to {OUT_DIR}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
