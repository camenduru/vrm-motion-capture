# Videos — demo live pose

Any clip in this folder shows up in the **pose source** picker in the web UI.
Pick one and the pose pipeline runs on the video exactly as it does on a
webcam: RTMPose keypoints are pushed over the websocket and the VRM avatar
follows them live.

You can also add clips straight from the UI with the **Add video…** button,
which stores the upload here. Accepted formats: `.mp4 .webm .mov .mkv .m4v
.avi` (max 512 MB).

## Clips in this folder

| file | notes |
| --- | --- |
| `green screen dancing girl 01.mp4` | recommended test clip: one person, full body, front facing, chroma key background |
| `Woman Dancing Hip Hop Green Screen -  Chroma Key - No Copyright.mp4` | second single-person dance clip |

Run the app on one of them without touching the camera:

```shell
python -m app.server --source "videos/green screen dancing girl 01.mp4"
```

Or just start the app and choose it in the source dropdown.

`python scripts/download_demo_videos.py` fetches a few extra freely licensed
clips from Wikimedia Commons into this folder and lists them, with attribution,
in `README-commons.md`.
