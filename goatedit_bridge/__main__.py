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
from .server import BridgeConfig, make_server

DEFAULT_PORT = 8765
DEFAULT_ORIGIN = "https://ai.goatedit.com"
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def _banner(port: int, token: str, origins: list[str], work_dir: str, keep: bool) -> None:
    kept = "kept" if keep else "wiped when this process stops"
    print()
    print(f"  GoatEdit bridge {__version__} — listening on http://127.0.0.1:{port}")
    print(f"  Paired with: {', '.join(origins)}")
    print(f"  Downloads land in: {work_dir} ({kept})")
    if not keep:
        print("  Pass --dir <folder> to keep them somewhere of your own.")
    if not ffmpeg_available():
        print("  ! ffmpeg not found — only already-combined streams (usually ≤720p)")
        print("    will download. Install ffmpeg for 1080p and up.")
    print()
    print("  Paste this token into the yt-dlp panel in GoatEdit:")
    print()
    print(f"      {token}")
    print()
    print("  Keep this window open while you download. Ctrl-C to stop.")
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
        "--token", default=None,
        help="use a fixed token instead of a fresh random one (handy while developing)",
    )
    parser.add_argument("--version", action="version", version=f"goatedit-bridge {__version__}")
    args = parser.parse_args(argv)

    origins = list(args.origin or [DEFAULT_ORIGIN])
    if args.dev:
        origins += DEV_ORIGINS
    token = args.token or secrets.token_urlsafe(24)

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

    config = BridgeConfig(port=args.port, token=token, origins=origins)
    try:
        httpd = make_server(config, JobStore(work_dir, keep_files=keep))
    except OSError as exc:
        print(f"Could not bind 127.0.0.1:{args.port} — {exc}", file=sys.stderr)
        print("Another bridge may already be running. Try --port 8766.", file=sys.stderr)
        return 1

    _banner(args.port, token, origins, work_dir, keep)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n  Stopping. Files left in {work_dir}." if keep
              else "\n  Stopping. Downloaded files are being cleaned up.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
