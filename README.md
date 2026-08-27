# goatedit-bridge

Runs [yt-dlp](https://github.com/yt-dlp/yt-dlp) on your own machine so the
[GoatEdit](https://ai.goatedit.com) editor can import YouTube videos at full quality —
your IP, your cookies, no server in the middle.

```bash
# Anywhere. Not on PyPI yet, so the wheel comes off a GitHub release.
uvx --from https://github.com/snkr433/goatedit-bridge/releases/download/v0.1.5/goatedit_bridge-0.1.5-py3-none-any.whl goatedit-bridge

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

The server binds `127.0.0.1` only, requires the printed token on every request, and
only answers browser requests coming from the editor origin it was paired with. Full
notes, API and threat model: [`docs/YTDLP_BRIDGE.md`](../docs/YTDLP_BRIDGE.md).

Developing against a checkout:

```bash
cd bridge
uv run goatedit-bridge --dev          # also accepts http://localhost:5173
```
