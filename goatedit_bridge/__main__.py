"""CLI entry point: `goatedit-bridge` (or `uvx goatedit-bridge`)."""

from __future__ import annotations

import argparse
import atexit
import os
import secrets
import shutil
import sys
import tempfile

from . import __version__
from .jobs import JobStore, ffmpeg_available
from .localfs import LocalFs
from .server import BridgeConfig, make_server

DEFAULT_PORT = 8765
DEFAULT_ORIGIN = "https://ai.goatedit.com"
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def _banner(
    port: int, token: str, origins: list[str], work_dir: str, keep: bool,
    shared: list[str] | None = None,
) -> None:
    kept = "kept" if keep else "wiped when this process stops"
    print()
    print(f"  GoatEdit bridge {__version__} — listening on http://127.0.0.1:{port}")
    print(f"  Paired with: {', '.join(origins)}")
    print(f"  Downloads land in: {work_dir} ({kept})")
    if not keep:
        print("  Pass --dir <folder> to keep them somewhere of your own.")
    from .render import probe_hardware
    hw = probe_hardware()
    if hw["available"]:
        print(f"  ⚡ Hardware Acceleration: {hw['gpuName']}")
        codec_names = [c["name"] for c in hw["codecs"]]
        print(f"     Codecs unlocked: {', '.join(codec_names)}")
        print(f"     Default export folder: {hw['defaultExportDir']}")
    else:
        print("  ! ffmpeg not found — install ffmpeg for hardware export rendering & 1080p+ downloads.")
    print()
    print("  Paste this token into GoatEdit (yt-dlp panel or Export dialog):")
    print()
    print(f"      {token}")
    print()
    if shared:
        print("  Folders the editor may read (media files only):")
        for folder in shared:
            print(f"      {folder}")
    else:
        print("  Local import is OFF. Pass --allow-dir <folder> to share a folder.")
    print()
    print("  Keep this window open while downloading or exporting. Ctrl-C to stop.")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="goatedit-bridge",
        description="Run yt-dlp locally for the GoatEdit yt-dlp plugin.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"loopback port (default {DEFAULT_PORT})")
    parser.add_argument(
        "--origin", action="append", default=None,
        help=f"editor origin allowed to call this bridge (repeatable, default {DEFAULT_ORIGIN})",
    )
    parser.add_argument("--dev", action="store_true", help="also accept the local Vite dev server")
    parser.add_argument(
        "--dir", dest="work_dir", default=None,
        help="folder to download into and keep files in (default: a temp dir, wiped on exit)",
    )
    parser.add_argument(
        "--allow-dir", dest="allow_dirs", action="append", default=None, metavar="FOLDER",
        help=(
            "share a folder with the editor so it can import media from it (repeatable). "
            "Nothing outside these folders is readable, and only video, audio and image "
            "files inside them are. Omit this and local import stays off."
        ),
    )
    parser.add_argument(
        "--token", default=None,
        help="use a fixed token instead of a fresh random one (handy while developing)",
    )
    parser.add_argument("--version", action="version", version=f"goatedit-bridge {__version__}")
    args = parser.parse_args(argv)

    origins = list(args.origin or [DEFAULT_ORIGIN])
    if args.dev:
        origins += DEV_ORIGINS
    token_cache_path = os.path.expanduser("~/.cache/goatedit-bridge/dev_token")
    if args.token:
        token = args.token
    elif args.dev:
        # In dev mode, reuse a stable token so restarting the terminal doesn't require re-pasting
        token = ""
        if os.path.exists(token_cache_path):
            try:
                with open(token_cache_path, "r", encoding="utf-8") as tf:
                    token = tf.read().strip()
            except Exception:
                token = ""
        if not token:
            token = secrets.token_urlsafe(24)
            try:
                os.makedirs(os.path.dirname(token_cache_path), exist_ok=True)
                with open(token_cache_path, "w", encoding="utf-8") as tf:
                    tf.write(token)
            except Exception:
                pass
    else:
        token = secrets.token_urlsafe(24)

    keep = args.work_dir is not None
    if keep:
        work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
        try:
            os.makedirs(work_dir, exist_ok=True)
        except OSError as exc:
            print(f"Cannot use {work_dir} — {exc}", file=sys.stderr)
            return 1
        if not os.access(work_dir, os.W_OK):
            print(f"Cannot write to {work_dir}", file=sys.stderr)
            return 1
    else:
        work_dir = tempfile.mkdtemp(prefix="goatedit-bridge-")
        atexit.register(shutil.rmtree, work_dir, True)

    # Off unless folders were named. A default of "the home directory" would be
    # the convenient choice and the wrong one: it turns every agent driving the
    # editor into something that can read the whole account's media.
    localfs = LocalFs(
        [os.path.expanduser(d) for d in (args.allow_dirs or [])],
        os.path.join(work_dir, "frames"),
    )
    if args.allow_dirs and not localfs.roots:
        print(f"None of {args.allow_dirs} is a folder that exists.", file=sys.stderr)
        return 1

    config = BridgeConfig(port=args.port, token=token, origins=origins, localfs=localfs)
    try:
        httpd = make_server(config, JobStore(work_dir, keep_files=keep))
    except OSError as exc:
        print(f"Could not bind 127.0.0.1:{args.port} — {exc}", file=sys.stderr)
        print("Another bridge may already be running. Try --port 8766.", file=sys.stderr)
        return 1

    _banner(args.port, token, origins, work_dir, keep, localfs.roots)
    try:
        httpd.serve_forever()
        # serve_forever returning on its own means /shutdown was called, so say
        # so — otherwise the window simply goes quiet and looks like a crash.
        print("\n  Stopped from the editor." if keep
              else "\n  Stopped from the editor. Downloaded files are being cleaned up.")
    except KeyboardInterrupt:
        print(f"\n  Stopping. Files left in {work_dir}." if keep
              else "\n  Stopping. Downloaded files are being cleaned up.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
