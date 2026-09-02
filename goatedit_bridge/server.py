"""The loopback HTTP server the plugin panel talks to.

Threat model, because a server on localhost is a server every page in the
browser can try to reach:

  * The socket binds 127.0.0.1, so nothing off this machine can connect.
  * Every request must carry `Authorization: Bearer <token>`. The token is
    generated per run and printed once; the user pastes it into the plugin.
  * Every browser request must carry an `Origin` we were started with
    (https://ai.goatedit.com by default). That is what stops evil.com from
    firing no-preflight requests at us, and — together with the Host check —
    what stops DNS rebinding.
  * The `Host` header must itself be loopback, so a rebound name that resolves
    to 127.0.0.1 still gets turned away.
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from . import __version__
from .jobs import PREVIEW_MAX_DURATION, JobStore, _clock, ffmpeg_available, probe, storyboard

MAX_REQUEST_BODY = 64 * 1024
FILE_CHUNK = 1024 * 1024

ALLOWED_SCHEMES = ("http", "https")


def _section(body: dict[str, Any]) -> tuple[float, float] | None:
    """The (start, end) the panel asked for, or None for the whole video.

    Both ends have to be real numbers and in order, because they are handed to
    ffmpeg as a seek range: a reversed or non-finite pair produces a file that
    is empty or never finishes rather than an error the user can read.
    """
    start_raw, end_raw = body.get("start"), body.get("end")
    if start_raw is None and end_raw is None:
        return None
    try:
        start_at, end_at = float(start_raw or 0), float(end_raw)
    except (TypeError, ValueError):
        raise ValueError("start and end must be numbers of seconds") from None
    if not (math.isfinite(start_at) and math.isfinite(end_at)):
        raise ValueError("start and end must be finite")
    if start_at < 0 or end_at <= start_at:
        raise ValueError("end must be greater than start, and start cannot be negative")
    return (start_at, end_at)


def _max_height(body: dict[str, Any]) -> int:
    """The tallest picture the caller will accept, or 0 for no ceiling.

    A caller that knows the shot is going to be six muted seconds of b-roll can
    say so and skip the 4K master. Rejected rather than clamped when it is not a
    sane number: silently downloading something other than what was asked for is
    how a caller ends up trusting a cap that never applied.
    """
    raw = body.get("maxHeight")
    if raw is None:
        return 0
    try:
        height = int(raw)
    except (TypeError, ValueError):
        raise ValueError("maxHeight must be a number of pixels") from None
    if height < 0:
        raise ValueError("maxHeight cannot be negative")
    return height


class BridgeConfig:
    def __init__(self, port: int, token: str, origins: list[str]) -> None:
        self.port = port
        self.token = token
        self.origins = origins


def _host_is_loopback(host_header: str) -> bool:
    host = host_header.rsplit(":", 1)[0].strip("[]").lower()
    return host in ("127.0.0.1", "localhost", "::1") or host.startswith("127.")


class BridgeHandler(BaseHTTPRequestHandler):
    server_version = f"goatedit-bridge/{__version__}"
    protocol_version = "HTTP/1.1"

    config: BridgeConfig
    jobs: JobStore

    # --- plumbing ------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # One tidy line per request instead of BaseHTTPRequestHandler's stderr spray.
        print(f"  {self.command} {self.path} → {args[1] if len(args) > 1 else ''}")

    def _origin_ok(self) -> str | None:
        """Returns the Origin to echo back, or None when it isn't allowed."""
        origin = self.headers.get("Origin")
        if origin is None:
            # Not a browser request (curl, a health probe). The token still gates it.
            return ""
        return origin if origin in self.config.origins else None

    def _send_json(self, status: int, payload: dict[str, Any], origin: str = "") -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors(origin)
        self.end_headers()
        self.wfile.write(body)

    def _send_cors(self, origin: str) -> None:
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Max-Age", "600")
        # A local downloader has no business being framed or sniffed.
        self.send_header("X-Content-Type-Options", "nosniff")

    def _guard(self) -> str | None:
        """Runs every check a request must pass. Returns the Origin, or None if it was answered with an error."""
        if not _host_is_loopback(self.headers.get("Host", "")):
            self._send_json(403, {"error": "This bridge only answers on 127.0.0.1"})
            return None
        origin = self._origin_ok()
        if origin is None:
            self._send_json(403, {"error": "That origin is not paired with this bridge"})
            return None
        auth = self.headers.get("Authorization", "")
        expected = f"Bearer {self.config.token}"
        # Constant-time: the token is the only secret here.
        if not secrets.compare_digest(auth, expected):
            self._send_json(401, {"error": "Missing or wrong bridge token"}, origin)
            return None
        return origin

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_REQUEST_BODY:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    # --- routes --------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        origin = self._origin_ok()
        if origin is None:
            self._send_json(403, {"error": "That origin is not paired with this bridge"})
            return
        self.send_response(204)
        self._send_cors(origin)
        # Private Network Access: Chrome treats a public page reaching 127.0.0.1
        # as a private-network request and preflights it with this header. Without
        # the matching opt-in the browser drops the request before we ever see it.
        # It is not a weakening of anything — the Origin and token checks below
        # still decide who gets an answer.
        if self.headers.get("Access-Control-Request-Private-Network") == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path

        # /health is the pairing probe: it answers without a token so the panel
        # can say "bridge found, paste your token" instead of "unreachable".
        # It leaks only that a bridge is running, and only to a paired origin.
        if path == "/health":
            origin = self._origin_ok()
            if origin is None:
                self._send_json(403, {"error": "That origin is not paired with this bridge"})
                return
            self._send_json(200, {
                "name": "goatedit-bridge",
                "version": __version__,
                "ffmpeg": ffmpeg_available(),
            }, origin)
            return

        origin = self._guard()
        if origin is None:
            return

        if path.startswith("/job/"):
            job = self.jobs.get(path[len("/job/"):])
            if not job:
                self._send_json(404, {"error": "No such job"}, origin)
                return
            self._send_json(200, job.public(), origin)
            return

        if path.startswith("/file/"):
            self._serve_file(path[len("/file/"):], origin)
            return

        self._send_json(404, {"error": f"No route for {path}"}, origin)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        origin = self._guard()
        if origin is None:
            return

        body = self._read_json()
        url = str(body.get("url") or "").strip()
        if path in ("/resolve", "/download", "/storyboard", "/preview"):
            if not url or urlparse(url).scheme not in ALLOWED_SCHEMES:
                self._send_json(400, {"error": "Give me an http(s) URL to work on"}, origin)
                return

        if path == "/resolve":
            try:
                self._send_json(200, probe(url), origin)
            except Exception as exc:  # noqa: BLE001
                self._send_json(502, {"error": str(exc)[:500]}, origin)
            return

        if path == "/storyboard":
            try:
                self._send_json(200, storyboard(url), origin)
            except Exception as exc:  # noqa: BLE001
                self._send_json(502, {"error": str(exc)[:500]}, origin)
            return

        if path == "/preview":
            # Fetching a small rendition so the panel can play the video before
            # the user commits to a quality or a range. An ordinary job in every
            # respect — the panel polls /job/<id> and then reads it off /file/,
            # exactly as it would a real download; only the reaper treats it
            # differently, deleting it even when downloads are being kept.
            try:
                # Cached from the /resolve the panel already did, so this is a
                # dictionary lookup rather than another minute of player API.
                duration = int(probe(url).get("duration") or 0)
            except Exception as exc:  # noqa: BLE001
                self._send_json(502, {"error": str(exc)[:500]}, origin)
                return
            if duration > PREVIEW_MAX_DURATION:
                self._send_json(413, {
                    "error": (
                        f"Too long to preview — {_clock(duration)} would mean fetching a "
                        "few hundred MB to pick two timestamps. Use the filmstrip."
                    ),
                    "maxDuration": PREVIEW_MAX_DURATION,
                }, origin)
                return
            job = self.jobs.start(url, None, False, None, preview=True)
            self._send_json(202, job.public(), origin)
            return

        if path == "/download":
            format_id = body.get("formatId")
            try:
                section = _section(body)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)}, origin)
                return
            try:
                max_height = _max_height(body)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)}, origin)
                return
            job = self.jobs.start(
                url,
                str(format_id) if format_id else None,
                bool(body.get("audioOnly")),
                section,
                max_height=max_height,
            )
            self._send_json(202, job.public(), origin)
            return

        self._send_json(404, {"error": f"No route for {path}"}, origin)

    # --- file delivery -------------------------------------------

    def _serve_file(self, job_id: str, origin: str) -> None:
        job = self.jobs.get(job_id)
        if not job or job.state != "ready" or not job.filepath:
            self._send_json(404, {"error": "That download is not ready"}, origin)
            return
        path = job.filepath
        # The path was built by us from a hex job id, but re-checking it sits
        # inside the work dir costs nothing and closes the whole class.
        if os.path.dirname(os.path.abspath(path)) != os.path.abspath(self.jobs.work_dir):
            self._send_json(403, {"error": "Refusing to serve a file outside the work directory"}, origin)
            return

        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", "attachment")
        self._send_cors(origin)
        self.end_headers()
        with open(path, "rb") as fh:
            while chunk := fh.read(FILE_CHUNK):
                self.wfile.write(chunk)


def make_server(config: BridgeConfig, jobs: JobStore) -> ThreadingHTTPServer:
    handler = type("BoundBridgeHandler", (BridgeHandler,), {"config": config, "jobs": jobs})
    httpd = ThreadingHTTPServer(("127.0.0.1", config.port), handler)
    httpd.daemon_threads = True
    return httpd
