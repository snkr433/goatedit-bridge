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

    def _send_preview_cors(self) -> None:
        """Headers for the one route a media element fetches for itself.

        `*` rather than the paired origin because the caller is a sandboxed
        frame and sends `Origin: null`; echoing that back would be the same
        permission spelled more alarmingly. The ticket in the path is what
        authorises the read, and the response is never credentialed, so `*`
        widens nothing — anyone who could use it already had the ticket.
        """
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "range")
        self.send_header("Access-Control-Expose-Headers", "content-length, content-range, accept-ranges")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Max-Age", "600")
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
        # A panel frame is sandboxed without allow-same-origin, so its requests
        # carry `Origin: null` and can never match the paired origin. The
        # preview route is built for exactly that caller and authenticates by
        # ticket instead, so its preflight is answered without the origin check
        # the token-bearing routes still get.
        if urlparse(self.path).path.startswith("/preview/"):
            self.send_response(204)
            self._send_preview_cors()
            if self.headers.get("Access-Control-Request-Private-Network") == "true":
                self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

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

        # Ticketed and therefore ahead of the token guard: this is the URL that
        # goes into a <video src>, and a media element sends no Authorization
        # header. The ticket is 192 bits from `secrets`, names one scratch file,
        # and dies with the job.
        if path.startswith("/preview/"):
            self._serve_preview(path[len("/preview/"):])
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

    def do_HEAD(self) -> None:  # noqa: N802
        """Only the preview route. A media element sometimes sizes a stream this
        way before it asks for any of it; everything else here has a body worth
        having and nothing to say without one."""
        path = urlparse(self.path).path
        if path.startswith("/preview/"):
            self._serve_preview(path[len("/preview/"):])
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

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
            # the user commits to a quality or a range. It is an ordinary job —
            # the panel polls /job/<id> for it like any other — but it comes
            # back carrying the URL its bytes will be readable at.
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
            payload = job.public()
            payload["previewUrl"] = f"http://127.0.0.1:{self.config.port}/preview/{job.ticket}"
            self._send_json(202, payload, origin)
            return

        if path == "/download":
            format_id = body.get("formatId")
            try:
                section = _section(body)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)}, origin)
                return
            job = self.jobs.start(
                url,
                str(format_id) if format_id else None,
                bool(body.get("audioOnly")),
                section,
            )
            self._send_json(202, job.public(), origin)
            return

        self._send_json(404, {"error": f"No route for {path}"}, origin)

    # --- file delivery -------------------------------------------

    def _serve_preview(self, ticket: str) -> None:
        """Streams a preview rendition to a <video>, honouring Range.

        Seeking is the whole point of this route, and a media element will not
        offer a scrub bar it cannot seek: it asks with `Range`, and if the answer
        is a flat 200 it treats the stream as unseekable. So a byte range gets a
        real 206 with `Content-Range`, and every answer advertises
        `Accept-Ranges`.
        """
        job = self.jobs.by_ticket(ticket)
        if not job or not job.filepath or not os.path.exists(job.filepath):
            # Deliberately the same answer for a wrong ticket and a job that has
            # expired: a preview URL should not report on tickets it was not given.
            self.send_response(404)
            self._send_preview_cors()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if job.state != "ready":
            self.send_response(409)
            self._send_preview_cors()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        path = job.filepath
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "video/mp4"
        start, end = _parse_range(self.headers.get("Range", ""), size)

        if start is None:
            self.send_response(200)
            length = size
        else:
            if start >= size:
                self.send_response(416)
                self._send_preview_cors()
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")

        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        # Scratch that is about to be reaped; a cached copy would outlive it.
        self.send_header("Cache-Control", "no-store")
        self._send_preview_cors()
        self.end_headers()

        if self.command == "HEAD":
            return
        remaining = length
        with open(path, "rb") as fh:
            if start is not None:
                fh.seek(start)
            while remaining > 0:
                chunk = fh.read(min(FILE_CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # Normal: the element seeked, or the panel moved on. The
                    # socket is gone, and there is nothing left to say on it.
                    return

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


def _parse_range(header: str, size: int) -> tuple[int | None, int]:
    """`Range: bytes=a-b` as (start, end) inclusive, or (None, size - 1).

    Only the single-range form, which is the only one a media element sends.
    Anything malformed, multi-range, or suffix-length past the file falls back
    to the whole file rather than erroring — a 200 is always a truthful answer
    to a range request, just a less useful one.
    """
    header = (header or "").strip().lower()
    if not header.startswith("bytes=") or "," in header:
        return None, size - 1
    spec = header[len("bytes="):].strip()
    first, sep, last = spec.partition("-")
    if not sep:
        return None, size - 1
    try:
        if not first:
            # `bytes=-N`: the final N bytes.
            length = int(last)
            if length <= 0:
                return None, size - 1
            return max(0, size - length), size - 1
        start = int(first)
        end = int(last) if last else size - 1
    except ValueError:
        return None, size - 1
    if start < 0 or end < start:
        return None, size - 1
    return start, min(end, size - 1)


def make_server(config: BridgeConfig, jobs: JobStore) -> ThreadingHTTPServer:
    handler = type("BoundBridgeHandler", (BridgeHandler,), {"config": config, "jobs": jobs})
    httpd = ThreadingHTTPServer(("127.0.0.1", config.port), handler)
    httpd.daemon_threads = True
    return httpd
