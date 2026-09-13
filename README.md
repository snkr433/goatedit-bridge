# goatedit-bridge

Runs [yt-dlp](https://github.com/yt-dlp/yt-dlp) on your own machine so the
[GoatEdit](https://ai.goatedit.com) editor can import YouTube videos at full quality —
your IP, your cookies, no server in the middle.

```bash
# Anywhere. Not on PyPI yet, so the wheel comes off a GitHub release.
uvx --from https://github.com/snkr433/goatedit-bridge/releases/download/v0.6.0/goatedit_bridge-0.6.0-py3-none-any.whl goatedit-bridge

# From this checkout.
cd bridge && uv run goatedit-bridge --dev
```

It prints a token. Paste that into **Plugins → yt-dlp Bridge** in the editor and leave
the window open while you download.

Downloads go to a temp dir that is wiped on exit. To keep them, name a folder:

```bash
uv run goatedit-bridge --dir ~/Downloads/GoatEdit
```

Files are then named `Title [videoId].ext` and the bridge never deletes them.

Install ffmpeg too (`brew install ffmpeg`, `sudo apt install ffmpeg`) — without it
YouTube's separate video and audio streams can't be combined, which caps you at
whatever pre-muxed stream is on offer, usually 720p or lower.

## Editing proxies

Some footage has no hardware decoder anywhere on your machine — H.264 4:2:2 or
10-bit (Sony XAVC-I and friends) is the common case. The browser falls back to
decoding it in software, which at 4K costs more than a frame's worth of time per
frame, so the timeline stutters however fast your GPU is.

Point the bridge at the folder and it probes each clip, transcodes only the ones
that actually need it, and writes 960x540 proxies into a `Proxies/` folder beside
the originals:

```bash
uv run goatedit-bridge --allow-dir ~/Footage/my-shoot
```

Then pick **Auto proxy** in the editor's view menu — *Timeline only* for the clips
in use, *All media* for the whole bin. Measured on 4K 50fps XAVC-I: frame p99 66.5ms
to 17.5ms, and stalls from 7 per 9 seconds to none.

Proxies are plain mp4 files. Delete the `Proxies/` folder to reclaim the space; the
editor falls back to the originals on its own.

`--allow-dir` grants read access to that folder and nothing else, and only for files
the bridge itself produced or was asked about.

The server binds `127.0.0.1` only, requires the printed token on every request, and
only answers browser requests coming from the editor origin it was paired with. Full
notes, API and threat model: [`docs/YTDLP_BRIDGE.md`](../docs/YTDLP_BRIDGE.md).

Developing against a checkout:

```bash
cd bridge
uv run goatedit-bridge --dev          # also accepts http://localhost:5173
```
